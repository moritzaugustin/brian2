#include "objects.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define N 4000
#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_neurongroup_group_variable_set_conditional_codeobject_kernel_1(int par_N, double* par_array_neurongroup_w, bool* par_array_neurongroup_not_refractory, float* par_rands)
{
	using namespace brian;

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * 1024 + tid;

	if(idx >= par_N)
	{
		return;
	}

	double * _ptr_array_neurongroup_w= par_array_neurongroup_w;
	bool * _ptr_array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;
	float r = par_rands[idx];

	double w = _ptr_array_neurongroup_w[idx];
	bool not_refractory = _ptr_array_neurongroup_not_refractory[idx];
	bool _cond = true;
	if(_cond)
	{
		if(not_refractory)
		{
			w = r * 10 * 0.0001;
		}
		_ptr_array_neurongroup_w[idx] = w;
	}
}


void _run_neurongroup_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;

	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*N);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, N);

	_run_neurongroup_group_variable_set_conditional_codeobject_kernel_1<<<ceil(N, 1024),1024>>>(N, dev_array_neurongroup_w, dev_array_neurongroup_not_refractory, dev_array_rands);

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
}

