#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel(int par_N, double* par_array_synapses_c, float* par_rands)
{
	using namespace brian;
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * 1024 + tid;

	if(idx >= par_N)
	{
		return;
	}

	double* _ptr_array_synapses_c = par_array_synapses_c;
	const bool _cond = true;
	float r = par_rands[idx];

	if(_cond)
	{
		double c;
		c = 5.3e-06 + r * 2.65e-06 - 2.65e-06 / 2;
		_ptr_array_synapses_c[idx] = c;
	}	
}

void _run_synapses_group_variable_set_conditional_codeobject()
{
	using namespace brian;

	const int64_t N = synapses._N();
	double* dev_array_synapses_c = thrust::raw_pointer_cast(&_dynamic_array_synapses_c[0]);

	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*N);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, N);


	_run_synapses_group_variable_set_conditional_codeobject_kernel<<<ceil(N, 1024),1024>>>(N, dev_array_synapses_c, dev_array_rands);
}

