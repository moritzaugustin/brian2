#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <curand.h>


////// SUPPORT CODE ///////
namespace {
}

////// HASH DEFINES ///////

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel(float* par_rands,
	int64_t par_N, int par_numw, double* par_array_synapses_w)
{
	int tid = threadIdx.x;
	//int64_t N = par_N;
	//int numw = par_numw;
	double * _ptr_array_synapses_w = par_array_synapses_w;

	const bool _cond = true;
	if(_cond)
	{
		double w;
		w = par_rands[tid] * 0.01;
		_ptr_array_synapses_w[tid] = w;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject()
{
	using namespace brian;

	const int64_t N = synapses._N();
	double r = rand();
	const int _numw = _dynamic_array_synapses_w.size();

	double* dev_array_synapses_w = thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0]);

	float* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*N);
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, dev_array_rands, N);

	//// MAIN CODE ////////////
	_run_synapses_group_variable_set_conditional_codeobject_kernel<<<1,N>>>(dev_array_rands,
		N, _numw, dev_array_synapses_w);

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
}


