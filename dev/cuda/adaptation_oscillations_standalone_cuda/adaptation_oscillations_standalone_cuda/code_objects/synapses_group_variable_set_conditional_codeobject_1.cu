#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel_1(int par_N, double* par_array_synapses_pre_delay, float* par_rands)
{
	using namespace brian;

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * 1024 + tid;

	if(idx >= par_N)
	{
		return;
	}

	double* _ptr_array_synapses_pre_delay = par_array_synapses_pre_delay;
	const bool _cond = true;
	double r = par_rands[idx];

	if(_cond)
	{
		double delay;
		delay = 0.002 + r * 0.001 - 0.001 / 2;
		_ptr_array_synapses_pre_delay[idx] = delay;
	}	
}

void _run_synapses_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;

	const int64_t N = synapses._N();
	double* dev_array_synapses_pre_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]);

	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*N);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, N);

	_run_synapses_group_variable_set_conditional_codeobject_kernel_1<<<ceil(N, 1024),1024>>>(N, dev_array_synapses_pre_delay, dev_array_rands);
}

