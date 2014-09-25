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
	cudaVector<DTYPE_int>** synapses_id_by_pre;
	cudaVector<DTYPE_int>** post_id_by_pre;
	cudaVector<unsigned int>** delay_by_pre;
	DTYPE_int* pre_id_by_syn;
	DTYPE_int* post_id_by_syn;
	unsigned int* synapses_delays;

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
		scalar* real_delays)
	{
		if(tid == 0)
		{
			current_offset = 0;
			num_blocks = _num_blocks;
			neuron_N = _neuron_N;
			syn_N = _syn_N;
			synapses_delays = new unsigned int[syn_N];

			synapses_id_by_pre = new cudaVector<DTYPE_int>*[num_blocks];
			post_id_by_pre = new cudaVector<DTYPE_int>*[num_blocks];
			delay_by_pre = new cudaVector<unsigned int>*[num_blocks];

			for(int i = 0; i < num_blocks; i++)
			{
				synapses_id_by_pre[i] = new cudaVector<DTYPE_int>[neuron_N];
				post_id_by_pre[i] = new cudaVector<DTYPE_int>[neuron_N];
				delay_by_pre[i] = new cudaVector<unsigned int>[neuron_N];
			}
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
		};
		__syncthreads();

		//only the first few threads can work now
		if(tid >= max_delay)
		{
			return;
		}

		pre_neuron_queue[tid] = new cudaVector<DTYPE_int>[num_blocks];
		synapses_queue[tid] = new cudaVector<DTYPE_int>[num_blocks];
		post_neuron_queue[tid] = new cudaVector<DTYPE_int>[num_blocks];
	};

	//second part of init
	//needs to be a different kernel, since the launch grid is different
	__device__ void prepare_connect_matrix(
		unsigned int bid,
		unsigned int syn_N,
		DTYPE_int* _synapses_id_by_pre,
		DTYPE_int* _pre_id_by_syn,
		DTYPE_int* _post_id_by_syn)
	{
		//ignore invalid tids (should not happen, but still...)
		if(bid >= num_blocks)
		{
			return;
		}

		for(int syn_id = 0; syn_id < syn_N; syn_id++)
		{
			int32_t post_neuron_id = _post_id_by_syn[syn_id];
			//push only neurons belonging into our queue
			if((post_neuron_id*num_blocks)/neuron_N == bid)
			{
				int32_t pre_neuron_id = _pre_id_by_syn[syn_id];
				unsigned int delay = synapses_delays[syn_id];
				synapses_id_by_pre[bid][pre_neuron_id].push(syn_id);
				post_id_by_pre[bid][pre_neuron_id].push(post_neuron_id);
				delay_by_pre[bid][pre_neuron_id].push(delay);
			}
		}
	};

//offset into shared memory in push kernel
#define SYN_ID_OFFSET(tid)  (3*tid + 0)
#define DELAY_OFFSET(tid)   (3*tid + 1)
#define POST_ID_OFFSET(tid) (3*tid + 2)

	__device__ void push(
		unsigned int bid,
		unsigned int tid,
		unsigned int _pre_id,
		int32_t* _shared_mem)
	{
		int32_t* shared_mem = _shared_mem;	//allocated in push_spikes_kernel
		unsigned int neuron_pre_id = _pre_id;
		unsigned int num_connected_synapses = synapses_id_by_pre[neuron_pre_id][bid].size();

		//ignore invalid pre_ids
		if(neuron_pre_id >= neuron_N || tid > num_connected_synapses)
		{
			return;
		}

		int32_t syn_id = synapses_id_by_pre[neuron_pre_id][bid].getDataByIndex(tid);
		shared_mem[SYN_ID_OFFSET(tid)] = syn_id;
		unsigned int delay = delay_by_pre[neuron_pre_id][bid].getDataByIndex(tid);
		shared_mem[DELAY_OFFSET(tid)] = delay;
		unsigned int post_id = post_id_by_pre[neuron_pre_id][bid].getDataByIndex(tid);
		shared_mem[POST_ID_OFFSET(tid)] = post_id;

		//only one thread per block inserts into queues
		if(tid != 0)
		{
			return;
		}

		for(int i = 0; i < num_connected_synapses; i++)
		{
			int32_t queue_pre_id = neuron_pre_id;
			int32_t queue_syn_id = shared_mem[SYN_ID_OFFSET(i)];
			int32_t queue_post_id = shared_mem[POST_ID_OFFSET(i)];
			unsigned int queue_delay = shared_mem[DELAY_OFFSET(i)];
			unsigned int adjusted_delay = (current_offset + queue_delay)%max_delay;
			unsigned int queue_id = bid;

			pre_neuron_queue[adjusted_delay][queue_id].push(queue_pre_id);
			synapses_queue[adjusted_delay][queue_id].push(queue_syn_id);
			post_neuron_queue[adjusted_delay][queue_id].push(queue_post_id);
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

private:
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
};

