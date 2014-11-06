#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define THREADS 1024
#define BLOCKS(N) ((N + THREADS -1)/THREADS)

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel(
	int par_syn_N,
	double* par_array_synapses_c,
	float* par_array_rands)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = par_syn_N;
	double* _ptr_array_synapses_c = par_array_synapses_c;
	float* _ptr_array_rands = par_array_rands;

	int syn_id = bid*THREADS + tid;
	if(syn_id < 0 || syn_id >= syn_N)
	{
		return;
	}

	const bool _cond = true;
	if(_cond)
	{
		double c;
		float r = _ptr_array_rands[syn_id];
		c = 5.3e-06 + r * 2.65e-06 - 2.65e-06 / 2;
		_ptr_array_synapses_c[syn_id] = c;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject()
{
	using namespace brian;

	const int syn_N = synapses._N();
	double* const dev_array_synapses_c = thrust::raw_pointer_cast(&_dynamic_array_synapses_c[0]);

	//genenerate an arry of random numbers on the device
	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*syn_N);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, syn_N);

	_run_synapses_group_variable_set_conditional_codeobject_kernel<<<BLOCKS(syn_N), THREADS>>>(
		syn_N,
		dev_array_synapses_c,
		dev_array_rands);

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
}

