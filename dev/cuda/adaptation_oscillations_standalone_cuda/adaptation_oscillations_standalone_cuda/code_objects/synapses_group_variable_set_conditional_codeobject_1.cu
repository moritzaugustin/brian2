#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS -1)/THREADS

__global__ void _run_synapses_group_variable_set_conditional_codeobject_1_kernel(
	int par_syn_N,
	double* par_array_synapses_pre_delay,
	float* par_array_rands)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = par_syn_N;
	double* _ptr_array_synapses_pre_delay = par_array_synapses_pre_delay;
	float* _ptr_array_rands = par_array_rands;

	int syn_id = bid*THREADS + tid;
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
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, syn_N);

	_run_synapses_group_variable_set_conditional_codeobject_1_kernel<<<BLOCKS(syn_N), THREADS>>>(
		syn_N,
		dev_array_synapses_pre_delay,
		dev_array_rands);

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
}

