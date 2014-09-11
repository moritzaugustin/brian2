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
#define BLOCKS (num_blocks_sequential)
#define neurons_N 4000

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

	if(bid < 0 || bid >= synapses_pre.queue->num_blocks_sequential)
	{
		return;
	}

	int neurons_per_thread = (neurons_N + THREADS - 1)/THREADS;
	int lower_limit = bid*neurons_per_thread;
	int upper_limit = (bid + 1)*neurons_per_thread;

	int num_queues = synapses_pre.queue->num_blocks_sequential;
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

	if(bid < 0 || bid >= synapses_pre.queue->num_blocks_sequential)
	{
		return;
	}

	synapses_pre.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	int size = synapses_queue[bid].size();
	for(int j = tid; j < size; j++)
	{
		int32_t syn_id = synapses_queue[bid].getDataByIndex(j);	
		_ptr_array_synapses_lastupdate[syn_id] = t;		
	}
}

__global__ void _run_synapses_pre_codeobject_post_kernel(
	double* _array_synapses_c,
	bool* _array_neurongroup_not_refractory,
	double* _array_neurongroup_v)
{
	using namespace brian;

	int bid = blockIdx.x;
	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;

	double* _ptr_array_synapses_c = _array_synapses_c;
	bool* _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;
	double* _ptr_array_neurongroup_v = _array_neurongroup_v;

	if(bid < 0 || bid >= synapses_pre.queue->num_blocks_sequential)
	{
		return;
	}

	synapses_pre.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	int size = post_neuron_queue[bid].size();
	for(int j = 0; j < size; j++)
	{
		int32_t post_neuron_id = post_neuron_queue[bid].getDataByIndex(j);
		int32_t syn_id = synapses_queue[bid].getDataByIndex(j);

		bool not_refractory = _ptr_array_neurongroup_not_refractory[post_neuron_id];
		if(not_refractory)
		{
			double v = _ptr_array_neurongroup_v[post_neuron_id];
			double c = _ptr_array_synapses_c[syn_id];
			v += c;
			_ptr_array_neurongroup_v[post_neuron_id] = v;
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

	_run_synapses_pre_codeobject_post_kernel<<<BLOCKS, 1>>>(
		dev_array_synapses_c,
		dev_array_neurongroup_not_refractory,
		dev_array_neurongroup_v);
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

