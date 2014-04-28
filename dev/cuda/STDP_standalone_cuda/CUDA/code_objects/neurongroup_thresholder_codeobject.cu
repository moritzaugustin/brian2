#include "objects.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

#define N 1

////// HASH DEFINES ///////

__global__ void _run_neurongroup_thresholder_codeobject_kernel(unsigned int par_num_spikespace,
	double par_t, int par_numv, int32_t* par_dev_array_neurongroup__spikespace,
	double* par_dev_array_neurongroup_v)
{
	int tid = threadIdx.x;
	//unsigned int _num_spikespace = par_num_spikespace;
	//const double t = par_t;
	//const int _numv = par_numv;
	int32_t* _ptr_array_neurongroup__spikespace = par_dev_array_neurongroup__spikespace;
	double* _ptr_array_neurongroup_v = par_dev_array_neurongroup_v;

	double v = _ptr_array_neurongroup_v[tid];
	double _cond = v > -0.054;
	if(_cond) {
		_ptr_array_neurongroup__spikespace[tid] = tid;
	}
	else {
		_ptr_array_neurongroup__spikespace[tid] = -1;
	}
	int _num_spikes = __syncthreads_count(_cond);
	if(tid == 0)
	{
		_ptr_array_neurongroup__spikespace[N] = _num_spikes;
	}
}


void _run_neurongroup_thresholder_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const unsigned int _num_spikespace = 2;
	const double t = defaultclock.t_();
	const int _numv = 1;

	///// POINTERS ////////////

	_run_neurongroup_thresholder_codeobject_kernel<<<1,N>>>(_num_spikespace, t,
		_numv, dev_array_neurongroup__spikespace, dev_array_neurongroup_v);
}


