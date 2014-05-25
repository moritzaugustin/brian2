#ifndef _SPIKEQUEUE_H_
#define _SPIKEQUEUE_H_

#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include "CudaVector.h"

using namespace std;

//TODO: The data type for indices is currently fixed (int), all floating point
//      variables (delays, dt) are assumed to use the same data type
typedef int32_t DTYPE_int;

template <class scalar>
class CSpikeQueue
{
public:
	CudaVector<DTYPE_int>** synapses_queue;
	CudaVector<DTYPE_int>** pre_neuron_queue;
	CudaVector<DTYPE_int>** post_neuron_queue;
	scalar dt;
	unsigned int offset;
	unsigned int* delays;
	int max_delay;
	int num_parallel;
	int source_start;
	int source_end;
	int32_t* targets;
	CudaVector<int>* synapses;

	__device__ void init(int num_mps, int _source_start, int _source_end)
	{
		source_start = _source_start;
		source_end = _source_end;
		offset = 0;
		dt = 0.0;
		delays = NULL;
		max_delay = 0;
		targets = NULL;
		num_parallel = num_mps;
		synapses_queue = NULL;
		pre_neuron_queue = NULL;
		post_neuron_queue = NULL;
		synapses = NULL;
	};

	__device__ void destroy()
	{
		for(int i = 0; i < max_delay; i++)
		{
			delete [] synapses_queue[i];
			delete [] pre_neuron_queue[i];
			delete [] post_neuron_queue[i];
		}
		if(synapses_queue)
			delete [] synapses_queue;
		if(pre_neuron_queue)
			delete [] pre_neuron_queue;
		if(post_neuron_queue)
			delete [] post_neuron_queue;
		if (delays)
			delete [] delays;
		if(synapses)
			delete [] synapses;
	}

	__device__ int get_maxdelay(unsigned int* delays, int n_delays)
	{
		if(n_delays == 0)
		{
			return 0;
		}
		int max = delays[0];
		for(int i = 1; i < n_delays; i++)
		{
			if(delays[i] > max)
			{
				max = delays[i];
			}
		}
		return max + 1;
	};

	__device__ void prepare(scalar *real_delays, int32_t* sources, int32_t* target, unsigned int n_synapses,
		 double _dt)
	{
		unsigned int newsize = 0;
		targets = target;
		if (delays)
		    delete [] delays;

		delays = new unsigned int[n_synapses];
		synapses = new CudaVector<int>[source_end - source_start];

		//syn connect aufbauen
		for(unsigned int i = 0; i < n_synapses; i++)
		{
			delays[i] =  (int)(real_delays[i] / _dt + 0.5);
			synapses[sources[i] - source_start].push(i);
		}

		max_delay = get_maxdelay(delays, n_synapses);
		//Get maximal delay of new and old spikes
		max_delay = (max_delay > newsize) ? max_delay : newsize;

		synapses_queue = new CudaVector<DTYPE_int>*[max_delay];
		pre_neuron_queue = new CudaVector<DTYPE_int>*[max_delay];
		post_neuron_queue = new CudaVector<DTYPE_int>*[max_delay];
		for(int i = 0; i < max_delay; i++)
		{
			synapses_queue[i] = new CudaVector<DTYPE_int>[num_parallel];
			pre_neuron_queue[i] = new CudaVector<DTYPE_int>[num_parallel];
			post_neuron_queue[i] = new CudaVector<DTYPE_int>[num_parallel];
		}
	}

	__device__ void push(int mpid, int* spikespace, int size_spikespace)
	{
		//each kernel works on consecutive elements of the spikespace
		for(int idx_spike = mpid*(size_spikespace/num_parallel); idx_spike < (mpid + 1)*(size_spikespace/num_parallel); idx_spike++)
		{
			const int idx_neuron = spikespace[idx_spike] - source_start;
			if(idx_neuron != -1)
			{
				//insert all connected synapses
				for(unsigned int idx_indices = 0; idx_indices < synapses[idx_neuron].size(); idx_indices++)
				{
					const int synaptic_index = synapses[idx_neuron].get(idx_indices);
					const unsigned int delay = delays[synaptic_index];
					// insert the synaptic, pre/post neuron indices into the correct queue
					synapses_queue[(offset+delay)%max_delay][mpid].push(synaptic_index);
					pre_neuron_queue[(offset+delay)%max_delay][mpid].push(idx_neuron);
					post_neuron_queue[(offset+delay)%max_delay][mpid].push(targets[synaptic_index]);
				}
			}
		}
	}

	__device__  void peek(CudaVector<DTYPE_int>** _synapses_queue,
		CudaVector<DTYPE_int>** _pre_neuron_queue, CudaVector<DTYPE_int>** _post_neuron_queue)
	{
		*(_synapses_queue) =  &(synapses_queue[offset][0]);
		*(_pre_neuron_queue) =  &(pre_neuron_queue[offset][0]);
		*(_post_neuron_queue) =  &(post_neuron_queue[offset][0]);
	}

	__device__ void advance(int tid)
	{
		synapses_queue[offset][tid].reset();
		pre_neuron_queue[offset][tid].reset();
		post_neuron_queue[offset][tid].reset();
		__syncthreads();
		if(tid == 0)
		{
			offset = (offset + 1) % max_delay;
		}
		__syncthreads();
	}
};

#endif

