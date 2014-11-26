#include "objects.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_neurongroup_group_variable_set_conditional_codeobject_1_kernel(
	unsigned int _neurongroup_N,
	unsigned int max_threads_per_block,
	float* par_array_rands,
	double* par_array_neurongroup_w,
	bool* par_array_neurongroup_not_refractory)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	float* _ptr_array_rands = par_array_rands;
	double* _ptr_array_neurongroup_w = par_array_neurongroup_w;
	bool* _ptr_array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;
	
	int neuron_id = bid*max_threads_per_block + tid;
	if(neuron_id < 0 || neuron_id >= _neurongroup_N)
	{
		return;
	}

	bool _cond = true;
	if(_cond)
	{
		const bool not_refractory = _ptr_array_neurongroup_not_refractory[neuron_id];
		double w = _ptr_array_neurongroup_w[neuron_id];
		if(not_refractory)
		{
			float r = _ptr_array_rands[neuron_id];	//get random pregenerated number
			w = r * 10 * 0.0001;
			_ptr_array_neurongroup_w[neuron_id] = w;
		}
	}
}

void _run_neurongroup_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;

	//genenerate an arry of random numbers on the device
	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*neurongroup_N);
	if(!dev_array_rands)
	{
		printf("ERROR while allocating device memory with size %ld in _run_neurongroup_group_variable_set_conditional_codeobject_1()\n", sizeof(float)*neurongroup_N);
	}
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, neurongroup_N);

	unsigned int blocks = (neurongroup_N + max_threads_per_block - 1)/max_threads_per_block;	// = ceil(N/num_threads)

	_run_neurongroup_group_variable_set_conditional_codeobject_1_kernel<<<blocks, max_threads_per_block>>>(
		neurongroup_N,
		max_threads_per_block,
		dev_array_rands,
		dev_array_neurongroup_w,
		dev_array_neurongroup_not_refractory);

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
}

