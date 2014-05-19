#ifndef _SPIKEQUEUE_H_
#define _SPIKEQUEUE_H_

#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include "CudaVector.h"
#include <thrust/device_vector.h>

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
	unsigned int *delays;
	int max_delay;
	int num_parallel;
	int source_start;
	int source_end;
	int* targets;
	CudaVector<int>* synapses;

	__device__ void init(int num_mps, int _source_start, int _source_end)
	{
		source_start = _source_start;
		source_end = _source_end;
		offset = 0;
		dt = 0.0;
		delays = NULL;
		max_delay = 0;
		num_parallel = num_mps;
	};

	__device__ void destroy()
	{
		for(int i = 0; i < max_delay; i++)
		{
			delete [] synapses_queue[i];
			delete [] pre_neuron_queue[i];
			delete [] post_neuron_queue[i];
		}
		delete [] synapses_queue;
		delete [] pre_neuron_queue;
		delete [] post_neuron_queue;
		delete [] delays;
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

	__device__ void prepare(scalar *real_delays, int *sources, int* target, unsigned int n_synapses,
		 double _dt)
	{
		unsigned int newsize = 0;
		targets = target;

		if (delays)
		    delete [] delays;

		delays = new unsigned int[n_synapses];
		synapses = new CudaVector<int>[source_start - source_end];

		//syn connect aufbauen
		for(unsigned int i = 0; i < n_synapses + 1; i++)
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

	__device__ void push(int mpid, int* par_spikes, int nspikes)
	{

		for(int idx_spike = mpid*(nspikes/num_parallel); idx_spike < (mpid + 1)*(nspikes/num_parallel) - 1; idx_spike++)
		{
			if(idx_spike < source_start - source_end)
			{
				const int idx_neuron = par_spikes[idx_spike] - source_start;
				if(idx_neuron != -1)
				{
					for(unsigned int idx_indices = 0; idx_indices < synapses[idx_neuron].size(); idx_indices++)
					{
						const int synaptic_index = synapses[idx_neuron].get(idx_indices);
						const unsigned int delay = delays[synaptic_index];
						// insert the index into the correct queue
						synapses_queue[mpid][(offset+delay)%max_delay].push(synaptic_index);
						pre_neuron_queue[mpid][(offset+delay)%max_delay].push(idx_neuron);
						post_neuron_queue[mpid][(offset+delay)%max_delay].push(targets[synaptic_index]);

					}
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
		synapses_queue[tid][offset].reset();
		pre_neuron_queue[tid][offset].reset();
		post_neuron_queue[tid][offset].reset();
		__syncthreads();
		if(tid == 0)
		{
			offset = (offset + 1) % max_delay;
		}
		__syncthreads();
	}
};

#endif

