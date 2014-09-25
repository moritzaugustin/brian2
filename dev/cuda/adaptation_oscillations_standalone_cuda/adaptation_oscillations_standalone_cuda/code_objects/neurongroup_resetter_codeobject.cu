#include "objects.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define neuron_N 4000
#define THREADS 1024
#define BLOCKS (neuron_N + THREADS -1)/THREADS

__global__ void _run_neurongroup_resetter_codeobject_kernel(
	int32_t* par_array_neurongroup__spikespace,
	double* par_array_neurongroup_w,
	double* par_array_neurongroup_v)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int neuron_id = bid*THREADS + tid;
	if(neuron_id < 0 || neuron_id >= neuron_N)
	{
		return;
	}

	int32_t* _ptr_array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double* _ptr_array_neurongroup_w = par_array_neurongroup_w;
	double* _ptr_array_neurongroup_v = par_array_neurongroup_v;

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

	_run_neurongroup_resetter_codeobject_kernel<<<BLOCKS, THREADS>>>(
		dev_array_neurongroup__spikespace,
		dev_array_neurongroup_w,
		dev_array_neurongroup_v);
}

