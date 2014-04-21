#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////

__global__ void _run_synapses_group_variable_set_conditional_codeobject_1_kernel(
	int64_t par_N, int par_numlastupdate, double* par_array_synapses_lastupdate)
{
	int tid = threadIdx.x;
	//int64_t N = par_N;
	//int _numlastupdate = par_numlastupdate;
	double * _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;

	const bool _cond = true;
	if(_cond)
	{
		double lastupdate;
		lastupdate = 0.0 * 1.0;
		_ptr_array_synapses_lastupdate[tid] = lastupdate;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;

	const int64_t N = synapses._N();
	const int _numlastupdate = _dynamic_array_synapses_lastupdate.size();

	double* dev_array_synapses_lastupdate;
	cudaMalloc((void**)&dev_array_synapses_lastupdate, sizeof(double)*_numlastupdate);
	cudaMemcpy(dev_array_synapses_lastupdate, &_dynamic_array_synapses_lastupdate[0], sizeof(double)*_numlastupdate, cudaMemcpyHostToDevice);

	//// MAIN CODE ////////////
	_run_synapses_group_variable_set_conditional_codeobject_1_kernel<<<1,N>>>(N,
		_numlastupdate, dev_array_synapses_lastupdate);

	cudaMemcpy(&_dynamic_array_synapses_lastupdate[0], dev_array_synapses_lastupdate, sizeof(double)*_numlastupdate, cudaMemcpyDeviceToHost);

	cudaFree(dev_array_synapses_lastupdate);
}


