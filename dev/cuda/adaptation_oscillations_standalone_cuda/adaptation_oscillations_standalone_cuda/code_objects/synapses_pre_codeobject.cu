#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

#define THREADS 1024
#define THREADS_POST 256
#define BLOCKS (num_blocks)
#define neurons_N 4000
#define MEMORY_PER_THREAD_POST (4*sizeof(double))

//does nothing in this program, here just to provide a skeleton for this kind of kernel
__global__ void _run_synapses_pre_codeobject_pre_kernel()
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

	int neurons_per_thread = (neurons_N + THREADS - 1)/THREADS;
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

//offset into shared memory in pre_codeobject kernel
#define POST_ID(tid)		(4*tid + 0)
#define NEURON_REFRACTORY(tid)	(4*tid + 1)
#define NEURON_V(tid)		(4*tid + 2)
#define SYN_C(tid)		(4*tid + 3)

// parallelization over post neuron id groups (due to organization of spike queues)
__global__ void _run_synapses_pre_codeobject_post_kernel(
	double* _array_synapses_c,
	bool* _array_neurongroup_not_refractory,
	double* _array_neurongroup_v)
{
	using namespace brian;
	extern __shared__ double shared_mem[];

	unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;
	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;

	double* _ptr_array_synapses_c = _array_synapses_c;
	bool* _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;
	double* _ptr_array_neurongroup_v = _array_neurongroup_v;

	//ignore invalid bids
	if(bid >= synapses_pre.queue->num_blocks)
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
	for(int j = tid; j < size; j += THREADS_POST)
	{
		int32_t post_neuron_id = post_neuron_queue[bid].getDataByIndex(j);
		shared_mem[POST_ID(tid)] = post_neuron_id;
		int32_t syn_id = synapses_queue[bid].getDataByIndex(j);
		bool not_refractory = _ptr_array_neurongroup_not_refractory[post_neuron_id];
		shared_mem[NEURON_REFRACTORY(tid)] = not_refractory;
		double v = _ptr_array_neurongroup_v[post_neuron_id];
		shared_mem[NEURON_V(tid)] = v;
		double c = _ptr_array_synapses_c[syn_id];
		shared_mem[SYN_C(tid)] = c;

		if(tid == 0)
		{
			//iterate over shared_mem
			for(int k = 0; k < THREADS_POST && j+k < size; k++)
			{
				bool spike_not_refractory = shared_mem[NEURON_REFRACTORY(k)];
				if(spike_not_refractory)
				{
					int32_t spike_post_id = shared_mem[POST_ID(k)];
					double spike_neuron_v = shared_mem[NEURON_V(k)];
					double spike_syn_c = shared_mem[SYN_C(k)];
					spike_neuron_v += spike_syn_c;
					_ptr_array_neurongroup_v[spike_post_id] = spike_neuron_v;
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

	//_run_synapses_pre_codeobject_pre_kernel<<<BLOCKS, THREADS>>>();

	_run_synapses_pre_codeobject_syn_kernel<<<BLOCKS, THREADS>>>(
		t,
		dev_array_synapses_lastupdate);

	//TODO: real number of spikes instead of THREADS_POST
	_run_synapses_pre_codeobject_post_kernel<<<BLOCKS, THREADS_POST, THREADS_POST*MEMORY_PER_THREAD_POST>>>(
		dev_array_synapses_c,
		dev_array_neurongroup_not_refractory,
		dev_array_neurongroup_v);
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

