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
	//these vectors should ALWAYS be the same size, since each index refers to a triple of (pre_id, syn_id, post_id)
	cudaVector<DTYPE_int>** pre_neuron_queue;
	cudaVector<DTYPE_int>** post_neuron_queue;
	cudaVector<DTYPE_int>** synapses_queue;

	//our connectivity matrix with dimensions (num_blocks) * neuron_N
	//each element
	unsigned int* size_by_pre;
	DTYPE_int** synapses_id_by_pre;
	DTYPE_int** post_id_by_pre;
	unsigned int** delay_by_pre;

	unsigned int current_offset;
	unsigned int max_delay;
	unsigned int num_blocks;
	unsigned int neuron_N;
	unsigned int syn_N;

	//Since we can't have a destructor, we need to call this function manually
	__device__ void destroy()
	{
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
	
	/* this function also initiliases all variables, allocs arrays, etc.
	 * so we need to call it before using the queue
	 * also call prepare_connect_matrix!
	 */
	__device__ void prepare(
		int tid,
		int num_threads,
		unsigned int _num_blocks,
		scalar _dt,
		unsigned int _neuron_N,
		unsigned int _syn_N,
		unsigned int _max_delay,
		unsigned int* _size_by_pre,
		DTYPE_int** _synapses_by_pre,
		DTYPE_int** _post_id_by_pre,
		unsigned int** _delay_by_pre)
	{
		if(tid == 0)
		{
			current_offset = 0;
			num_blocks = _num_blocks;
			neuron_N = _neuron_N;
			syn_N = _syn_N;
			max_delay = _max_delay;

			size_by_pre = _size_by_pre;
			synapses_id_by_pre = _synapses_by_pre;
			post_id_by_pre = _post_id_by_pre;
			delay_by_pre = _delay_by_pre;

			pre_neuron_queue = new cudaVector<DTYPE_int>*[max_delay];
			if(!pre_neuron_queue)
			{
				printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>*)*max_delay);
			}
			synapses_queue = new cudaVector<DTYPE_int>*[max_delay];
			if(!synapses_queue)
			{
				printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>*)*max_delay);
			}
			post_neuron_queue = new cudaVector<DTYPE_int>*[max_delay];
			if(!post_neuron_queue)
			{
				printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>*)*max_delay);
			}
		};
		__syncthreads();

		//only the first few threads can work now
		if(tid >= max_delay)
		{
			return;
		}

		pre_neuron_queue[tid] = new cudaVector<DTYPE_int>[num_blocks];
		if(!pre_neuron_queue[tid])
		{
			printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>)*num_blocks);
		}
		synapses_queue[tid] = new cudaVector<DTYPE_int>[num_blocks];
		if(!synapses_queue[tid])
		{
			printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>)*num_blocks);
		}
		post_neuron_queue[tid] = new cudaVector<DTYPE_int>[num_blocks];
		if(!post_neuron_queue[tid])
		{
			printf("ERROR while allocating memory with size %ld in spikequeue.h/prepare()\n", sizeof(cudaVector<DTYPE_int>)*num_blocks);
		}
	};

	__device__ void push(
		unsigned int bid,
		unsigned int tid,
		unsigned int num_threads,
		unsigned int _pre_id,
		char* _shared_mem)
	{
		unsigned int neuron_pre_id = _pre_id;
		unsigned int right_offset = neuron_pre_id*num_blocks + bid;
		unsigned int num_connected_synapses = size_by_pre[right_offset];
		//shared_mem is allocated in push_spikes
		int32_t* shared_mem_synapses_id = (int32_t*)_shared_mem;
		unsigned int* shared_mem_synapses_delay = (unsigned int*)((int32_t*)shared_mem_synapses_id + num_connected_synapses);
		int32_t* shared_mem_post_id = (int32_t*)((unsigned int*)shared_mem_synapses_delay + num_connected_synapses);

		//ignore invalid pre_ids
		if(neuron_pre_id >= neuron_N)
		{
			return;
		}

		for(int i = tid; i < num_connected_synapses; i += num_threads)
		{
			int32_t syn_id = synapses_id_by_pre[right_offset][i];
			shared_mem_synapses_id[tid] = syn_id;
			unsigned int delay = delay_by_pre[right_offset][i];
			shared_mem_synapses_delay[tid] = delay;
			unsigned int post_id = post_id_by_pre[right_offset][i];
			shared_mem_post_id[tid] = post_id;

			if(tid < max_delay)
			{
				for(int j = 0; j < num_threads && (i-tid)+j < num_connected_synapses; j++)
				{
					int32_t queue_pre_id = neuron_pre_id;
					int32_t queue_syn_id = shared_mem_synapses_id[j];
					int32_t queue_post_id = shared_mem_post_id[j];
					unsigned int queue_delay = shared_mem_synapses_delay[j];
					unsigned int adjusted_delay = (current_offset + queue_delay)%max_delay;
					unsigned int queue_id = bid;

					if(tid == adjusted_delay)
					{
						pre_neuron_queue[adjusted_delay][queue_id].push(queue_pre_id);
						synapses_queue[adjusted_delay][queue_id].push(queue_syn_id);
						post_neuron_queue[adjusted_delay][queue_id].push(queue_post_id);
					}
				}
			}
			__syncthreads();
		}
	}

	__device__ void advance(
		unsigned int tid)
	{
		if(tid >= num_blocks || current_offset >= max_delay)
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

