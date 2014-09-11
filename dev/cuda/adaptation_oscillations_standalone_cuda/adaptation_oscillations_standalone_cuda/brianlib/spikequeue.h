#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include "cudaVector.h"

#include <cstdio>

using namespace std;

//TODO: The data type for indices is currently fixed (int), all floating point
//      variables (delays, dt) are assumed to use the same data type
typedef int32_t DTYPE_int;

template <class scalar>
class CSpikeQueue
{
public:
	cudaVector<DTYPE_int>** pre_neuron_queue;
	cudaVector<DTYPE_int>** post_neuron_queue;
	cudaVector<DTYPE_int>** synapses_queue;

	DTYPE_int* synapses_id_by_pre;
	DTYPE_int* pre_id_by_syn;
	DTYPE_int* post_id_by_syn;
	unsigned int* synapses_delays;

	unsigned int current_offset;
	unsigned int max_delay;
	unsigned int num_blocks_sequential;
	unsigned int neuron_N;
	unsigned int syn_N;

	__device__ unsigned int get_max_delay(
		unsigned int* delays,
		int synapses_N)
	{
		if(synapses_N == 0)
		{
			return 0;
		}
		unsigned int max = delays[0];
		for(int i = 1; i < synapses_N; i++)
		{
			if(delays[i] > max)
			{
				max = delays[i];
			}
		}
		return max + 1; //add +1 because we also need the current step
	}

	__device__ void destroy()
	{
		for(unsigned int i = 0; i < num_blocks_sequential; i++)
		{
			if(pre_neuron_queue[i])
			{
				delete [] pre_neuron_queue[i];
			}
			if(synapses_queue[i])
			{
				delete [] synapses_queue[i];
			}
			if(post_neuron_queue[i])
			{
				delete [] post_neuron_queue[i];
			}
		}
		if(pre_neuron_queue)
		{
			delete [] pre_neuron_queue;
			pre_neuron_queue = 0;
		}
		if(synapses_queue)
		{
			delete [] synapses_queue;
			synapses_queue = 0;
		}
		if(post_neuron_queue)
		{
			delete [] post_neuron_queue;
			post_neuron_queue = 0;
		}
	}
	
	__device__ void prepare(
		int tid,
		int num_threads,
		unsigned int _num_blocks_sequential,
		scalar _dt,
		unsigned int _neuron_N,
		unsigned int _syn_N,
		scalar* real_delays,
		DTYPE_int* _synapses_id_by_pre,
		DTYPE_int* _pre_id_by_syn,
		DTYPE_int* _post_id_by_syn)
	{
		if(tid == 0)
		{
			current_offset = 0;
			num_blocks_sequential = _num_blocks_sequential;
			neuron_N = _neuron_N;
			syn_N = _syn_N;
			synapses_id_by_pre = _synapses_id_by_pre;
			pre_id_by_syn = _pre_id_by_syn;
			post_id_by_syn = _post_id_by_syn;
			synapses_delays = new unsigned int[syn_N];
		}

		__syncthreads();
		//ignore invalid tids (should not happen, but still...)
		if(tid < 0 || tid >= syn_N)
		{
			return;
		}

		for(int i = tid; i < syn_N; i += num_threads)
		{
			synapses_delays[i] =  (int)(real_delays[i] / _dt + 0.5); //round to nearest int
		}

		__syncthreads();
		
		if(tid == 0)
		{
			max_delay = get_max_delay(synapses_delays, syn_N);
			pre_neuron_queue = new cudaVector<DTYPE_int>*[max_delay];
			synapses_queue = new cudaVector<DTYPE_int>*[max_delay];
			post_neuron_queue = new cudaVector<DTYPE_int>*[max_delay];
		}
		__syncthreads();

		//only the first few threads can work now
		if(tid >= max_delay)
		{
			return;
		}

		pre_neuron_queue[tid] = new cudaVector<DTYPE_int>[num_blocks_sequential];
		synapses_queue[tid] = new cudaVector<DTYPE_int>[num_blocks_sequential];
		post_neuron_queue[tid] = new cudaVector<DTYPE_int>[num_blocks_sequential];
	}

//offset into shared memory in push()
#define SYN_ID(tid)  (3*tid + 0)
#define DELAY(tid)   (3*tid + 1)
#define POST_ID(tid) (3*tid + 2)

	__device__ void push(
		unsigned int tid,
		unsigned int num_threads,
		unsigned int _pre_id,
		int32_t* _shared_mem)
	{
		int32_t* shared_mem = _shared_mem;
		unsigned int neuron_pre_id = _pre_id;

		//ignore invalid pre_ids
		if(neuron_pre_id >= neuron_N || tid > num_threads)
		{
			return;
		}

		int32_t syn_id = synapses_id_by_pre[neuron_pre_id] + tid;
		shared_mem[SYN_ID(tid)] = syn_id;
		unsigned int delay = synapses_delays[syn_id];
		shared_mem[DELAY(tid)] = delay;
		int32_t post_id = post_id_by_syn[syn_id];
		shared_mem[POST_ID(tid)] = post_id;

		//only one thread per blocks inserts into queues
		if(tid != 0)
		{
			return;
		}

		for(int i = 0; i < num_threads; i++)
		{
			int32_t queue_pre_id = neuron_pre_id;
			int32_t queue_syn_id = shared_mem[SYN_ID(i)];
			int32_t queue_post_id = shared_mem[POST_ID(i)];
			unsigned int queue_delay = shared_mem[DELAY(i)];
			unsigned int adjusted_delay = (current_offset + queue_delay)%max_delay;
			unsigned int queue_id = (queue_post_id * num_blocks_sequential) / neuron_N;

			pre_neuron_queue[adjusted_delay][queue_id].push(queue_pre_id);
			synapses_queue[adjusted_delay][queue_id].push(queue_syn_id);
			post_neuron_queue[adjusted_delay][queue_id].push(queue_post_id);
		}
	}

	__device__ void advance(
		unsigned int tid)
	{
		if(tid >= num_blocks_sequential || current_offset >= max_delay)
		{
			return;
		}
		pre_neuron_queue[current_offset][tid].reset();
		synapses_queue[current_offset][tid].reset();
		post_neuron_queue[current_offset][tid].reset();
		__syncthreads();
		if(tid == 0)
		{
			current_offset = (current_offset + 1)%max_delay;
		}
	}

	__device__  void peek(
		cudaVector<DTYPE_int>** _synapses_queue,
		cudaVector<DTYPE_int>** _pre_neuron_queue,
		cudaVector<DTYPE_int>** _post_neuron_queue)
	{
		*(_synapses_queue) =  &(synapses_queue[current_offset][0]);
		*(_pre_neuron_queue) =  &(pre_neuron_queue[current_offset][0]);
		*(_post_neuron_queue) =  &(post_neuron_queue[current_offset][0]);
	}
};

