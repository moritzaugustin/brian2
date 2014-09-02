#ifndef _SPIKEQUEUE_H_
#define _SPIKEQUEUE_H_

#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include <stdio.h>

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
	int* delays;
	unsigned int max_delay;
	int num_neurons;
	int num_parallel;
	int source_start;
	int source_end;
	int32_t* targets;
	int32_t* starting_positions;

	__device__ void init(int num_mps, int _source_start, int _source_end)
	{
		source_start = _source_start;
		source_end = _source_end;
		offset = 0;
		dt = 0.0;
		delays = NULL;
		max_delay = 0;
		num_neurons = _source_end - _source_start;
		targets = NULL;
		num_parallel = num_mps;
		synapses_queue = NULL;
		pre_neuron_queue = NULL;
		post_neuron_queue = NULL;
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
	}

	__device__ int get_maxdelay(int* delays, int n_delays)
	{
		if(n_delays == 0)
		{
			return 0;
		}
		unsigned int max = delays[0];
		for(int i = 1; i < n_delays; i++)
		{
			if(delays[i] > max)
			{
				max = delays[i];
			}
		}
		return max + 1;
	};

	__device__ void prepare(int tid, scalar *real_delays, int32_t* sources, int32_t* target, int32_t* pos, unsigned int n_synapses,
		 int n_neurons, double _dt)
	{
		unsigned int newsize = 0;
		if(tid == 0)
		{
			targets = target;
			starting_positions = pos;
			if (delays)
			    delete [] delays;
			
			delays = new int[n_synapses];
		}

		__syncthreads();
		for(int i = tid; i < n_synapses; i += 1000) //change 1000
		{
			delays[i] =  (int)(real_delays[i] / _dt + 0.5);
		}
		__syncthreads();

		if(tid == 0)
		{
			max_delay = get_maxdelay(delays, n_synapses);
			//Get maximal delay of new and old spikes
			max_delay = (max_delay > newsize) ? max_delay : newsize;

			synapses_queue = new CudaVector<DTYPE_int>*[max_delay];
			pre_neuron_queue = new CudaVector<DTYPE_int>*[max_delay];
			post_neuron_queue = new CudaVector<DTYPE_int>*[max_delay];
		}

		__syncthreads();
		if(tid < max_delay)
		{
			synapses_queue[tid] = new CudaVector<DTYPE_int>[num_parallel];
			pre_neuron_queue[tid] = new CudaVector<DTYPE_int>[num_parallel];
			post_neuron_queue[tid] = new CudaVector<DTYPE_int>[num_parallel];
		}
	}

	__device__ void push(int32_t pre_id, int mpid, int* data, int* spikespace, int num_connected)
	{
		int32_t syn_id = starting_positions[pre_id] + mpid;
		int32_t post_id = targets[syn_id];
		int delay = delays[syn_id];
		data[mpid*3] = syn_id;
		data[mpid*3 + 1] = post_id;
		data[mpid*3 + 2] = delay;

		if(mpid < num_parallel)
		{
			for(int i = 0; i < num_connected; i++)
			{
				int queue_post_id = data[i*3 + 1];
				if(queue_post_id == mpid)
				{
					int queue_pre_id = pre_id;
					int queue_syn_id = data[i*3];

					int queue_delay = data[i*3 + 2];
					int queue_id = (queue_post_id * num_parallel) / num_neurons;
					synapses_queue[(offset+queue_delay)%max_delay][queue_id].push(queue_syn_id);
					pre_neuron_queue[(offset+queue_delay)%max_delay][queue_id].push(queue_pre_id);
					post_neuron_queue[(offset+queue_delay)%max_delay][queue_id].push(queue_post_id);
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

	__device__ void advance(int bid)
	{
		synapses_queue[offset][bid].reset();
		pre_neuron_queue[offset][bid].reset();
		post_neuron_queue[offset][bid].reset();
		__syncthreads();
		if(bid == 0)
		{
			offset = (offset + 1) % max_delay;
		}
	}
};

#endif

