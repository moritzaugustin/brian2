#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	double _rand(int vectorisation_idx)
	{
	    return (double)rand()/RAND_MAX;
	}
}

////// HASH DEFINES ///////

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel(
	double* par_rands, int64_t par_N, int par_numw, double* par_array_synapses_w)
{
	int tid = threadIdx.x;
	//int64_t N = par_N;
	//int numw = par_numw;
	double * _ptr_array_synapses_w = par_array_synapses_w;

	const bool _cond = true;
	if(_cond)
	{
		double w;
		//TODO: real rand
		//w = rand() * 0.01;
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

	double* dev_array_synapses_w;
	cudaMalloc((void**)&dev_array_synapses_w, sizeof(double)*_numw);
	cudaMemcpy(dev_array_synapses_w, &_dynamic_array_synapses_w[0], sizeof(double)*_numw, cudaMemcpyHostToDevice);

	//TODO: real rands on GPU
	double* dev_array_rands;
	cudaMalloc((void**)&dev_array_rands, sizeof(double)*N);
	double* rands = (double*)malloc(sizeof(double)*N);
	for(int i = 0; i < N; i++) rands[i] = _rand(0);
	cudaMemcpy(dev_array_rands, rands, sizeof(double)*N, cudaMemcpyHostToDevice);

	//// MAIN CODE ////////////
	_run_synapses_group_variable_set_conditional_codeobject_kernel<<<1,N>>>(dev_array_rands,
		N, _numw, dev_array_synapses_w);

	cudaMemcpy(&_dynamic_array_synapses_w[0], dev_array_synapses_w, sizeof(double)*_numw, cudaMemcpyDeviceToHost);

	cudaFree(dev_array_synapses_w);
}


