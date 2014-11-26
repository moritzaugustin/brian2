#include "objects.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_neurongroup_resetter_codeobject_kernel(
	unsigned int _neurongroup_N,
	unsigned int max_threads_per_block,
	int32_t* _array_neurongroup__spikespace,
	double* _array_neurongroup_w,
	double* _array_neurongroup_v)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int neuron_id = bid*max_threads_per_block + tid;
	if(neuron_id < 0 || neuron_id >= _neurongroup_N)
	{
		return;
	}

	int32_t* _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
	double* _ptr_array_neurongroup_w = _array_neurongroup_w;
	double* _ptr_array_neurongroup_v = _array_neurongroup_v;

	int32_t spiking_neuron = _ptr_array_neurongroup__spikespace[neuron_id];
	if(spiking_neuron != -1)
	{
		double w = _ptr_array_neurongroup_w[spiking_neuron];
		double v;
		v = 0.0;
		w = w + 0.0001;
		_ptr_array_neurongroup_v[spiking_neuron] = v;
		_ptr_array_neurongroup_w[spiking_neuron] = w;
	}

	//also reset spikespace array
	_ptr_array_neurongroup__spikespace[neuron_id] = -1;
}

void _run_neurongroup_resetter_codeobject()
{
	using namespace brian;

	unsigned int blocks = (neurongroup_N + max_threads_per_block - 1)/max_threads_per_block;	// = ceil(N/num_threads)

	_run_neurongroup_resetter_codeobject_kernel<<<blocks, max_threads_per_block>>>(
		neurongroup_N,
		max_threads_per_block,
		dev_array_neurongroup__spikespace,
		dev_array_neurongroup_w,
		dev_array_neurongroup_v);
}

