#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_synapses_group_variable_set_conditional_codeobject_1_kernel(
	int _syn_N,
	unsigned int max_threads_per_block,
	double* _array_synapses_pre_delay,
	float* _array_rands)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = _syn_N;
	double* _ptr_array_synapses_pre_delay = _array_synapses_pre_delay;
	float* _ptr_array_rands = _array_rands;

	int syn_id = bid*max_threads_per_block + tid;
	if(syn_id < 0 || syn_id >= syn_N)
	{
		return;
	}

	const bool _cond = true;
	if(_cond)
	{
		double delay;
		float r = _ptr_array_rands[syn_id];
		delay = 0.002 + r * 0.001 - 0.001 / 2;
		_ptr_array_synapses_pre_delay[syn_id] = delay;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;

	const int syn_N = synapses._N();
	double* const dev_array_synapses_pre_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]);

	//genenerate an arry of random numbers on the device
	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*syn_N);
	if(!dev_array_rands)
	{
		printf("ERROR while allocating device memory with size %ld in _run_synapses_group_variable_set_conditional_codeobject_1()\n", sizeof(float)*syn_N);
	}
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, syn_N);

	unsigned int blocks = (syn_N + max_threads_per_block - 1)/max_threads_per_block;	// = ceil(N/num_threads)

	_run_synapses_group_variable_set_conditional_codeobject_1_kernel<<<blocks, max_threads_per_block>>>(
		syn_N,
		max_threads_per_block,
		dev_array_synapses_pre_delay,
		dev_array_rands);

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
}

