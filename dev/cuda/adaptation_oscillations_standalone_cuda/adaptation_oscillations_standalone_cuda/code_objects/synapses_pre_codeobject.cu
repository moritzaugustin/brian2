#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

#define MEMORY_PER_THREAD_POST (4*sizeof(double))

//does nothing in this program, here just to provide a skeleton for this kind of kernel
__global__ void _run_synapses_pre_codeobject_pre_kernel(
	unsigned int _neurongroup_N,
	unsigned int _threads_per_block)
{
	using namespace brian;

	int bid = blockIdx.x;

	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;

	synapses_pre.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	if(bid < 0 || bid >= synapses_pre.queue->num_blocks)
	{
		return;
	}

	int neurons_per_thread = (_neurongroup_N + _threads_per_block - 1)/_threads_per_block;	//= ceil(N/num_threads)
	int lower_limit = bid*neurons_per_thread;
	int upper_limit = (bid + 1)*neurons_per_thread;

	int num_queues = synapses_pre.queue->num_blocks;
	for(int i = 0; i < num_queues; i++)
	{
		int size = pre_neuron_queue[i].size();
		for(int j = 0; j < size; j++)
		{
			int32_t pre_neuron_id = pre_neuron_queue[i].getDataByIndex(j);
			if(pre_neuron_id >= lower_limit && pre_neuron_id < upper_limit)
			{
				//DO NOTHING
			}
		}
	}
}

__global__ void _run_synapses_pre_codeobject_syn_kernel(
	double _t,
	double* _array_synapses_lastupdate)
{
	using namespace brian;

	int bid = blockIdx.x;
	int tid = blockIdx.x;
	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;

	double t = _t;
	double* _ptr_array_synapses_lastupdate = _array_synapses_lastupdate;

	if(bid < 0 || bid >= synapses_pre.queue->num_blocks)
	{
		return;
	}

	synapses_pre.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	//each threads works on spikes with id % no. threads == 0
	int size = synapses_queue[bid].size();
	for(int j = tid; j < size; j++)
	{
		int32_t syn_id = synapses_queue[bid].getDataByIndex(j);	
		_ptr_array_synapses_lastupdate[syn_id] = t;		
	}
}

// parallelization over post neuron id groups (due to organization of spike queues)
__global__ void _run_synapses_pre_codeobject_post_kernel(
	unsigned int _num_threads,
	double* _array_synapses_c,
	bool* _array_neurongroup_not_refractory,
	double* _array_neurongroup_v)
{
	using namespace brian;
	extern __shared__ char shared_mem[];
	double* shared_mem_post_ids = (double*)shared_mem;
	bool* shared_mem_neurongroup_not_refractory = (bool*)((double*)shared_mem_post_ids + _num_threads);
	double* shared_mem_synapses_c = (double*)(shared_mem_neurongroup_not_refractory + _num_threads);

	unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;

	//ignore invalid bids and tids
	if(bid >= synapses_pre.queue->num_blocks || tid >= _num_threads)
	{
		return;
	}

	synapses_pre.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	/* each block works only on one queue_block
	 * we have an outer and an inner loop
	 * in each outer loop iteration we read data into shared memory 
	 * and only one thread per block updates the global arrays
	 */
	int size = post_neuron_queue[bid].size();
	//outer loop, since most likely not all spikes fit into our shared memory
	for(int j = tid; j < size; j += _num_threads)
	{
		int32_t post_neuron_id = post_neuron_queue[bid].getDataByIndex(j);
		shared_mem_post_ids[tid] = post_neuron_id;
		int32_t syn_id = synapses_queue[bid].getDataByIndex(j);
		bool not_refractory = _array_neurongroup_not_refractory[post_neuron_id];
		shared_mem_neurongroup_not_refractory[tid] = not_refractory;
		double c = _array_synapses_c[syn_id];
		shared_mem_synapses_c[tid] = c;

		if(tid == 0)
		{
			//iterate over shared_mem
			for(int k = 0; k < _num_threads && j+k < size; k++)
			{
				bool spike_not_refractory = shared_mem_neurongroup_not_refractory[k];
				if(spike_not_refractory)
				{
					int32_t spike_post_id = shared_mem_post_ids[k];
					double spike_neuron_v = _array_neurongroup_v[spike_post_id];
					double spike_syn_c = shared_mem_synapses_c[k];
					spike_neuron_v += spike_syn_c;
					_array_neurongroup_v[spike_post_id] = spike_neuron_v;
				}
			}
		}
	}
}

void _run_synapses_pre_codeobject()
{
	using namespace brian;

	double* dev_array_synapses_c = thrust::raw_pointer_cast(&_dynamic_array_synapses_c[0]);
	double* dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);
	double t = defaultclock.t_();

	/*
	_run_synapses_pre_codeobject_pre_kernel<<<num_blocks, max_threads_per_block>>>(
		neurongroup_N);
	*/

	_run_synapses_pre_codeobject_syn_kernel<<<num_blocks, max_threads_per_block>>>(
		t,
		dev_array_synapses_lastupdate);

	unsigned int num_threads = max_shared_mem_size / MEMORY_PER_THREAD_POST;
	num_threads = num_threads > max_threads_per_block? max_threads_per_block : num_threads;	// get min of both

	//TODO: #threads = number of spikes instead of maximal possible value
	_run_synapses_pre_codeobject_post_kernel<<<num_blocks, num_threads, num_threads*MEMORY_PER_THREAD_POST>>>(
		num_threads,
		dev_array_synapses_c,
		dev_array_neurongroup_not_refractory,
		dev_array_neurongroup_v);
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

