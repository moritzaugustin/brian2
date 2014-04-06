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

__global__ void _run_neurongroup_thresholder_codeobject_kernel(unsigned int par_num_spikespace, double par_t, int par_numv, int32_t* par_dev_array_neurongroup__spikespace, double* par_dev_array_neurongroup_v)
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
	__syncthreads();
	if(tid == 0)
	{
		int i = 0;
		for(int j = 0; j < N; j++)
		{
			if(_ptr_array_neurongroup__spikespace[j] != -1)
			{
				_ptr_array_neurongroup__spikespace[i] = _ptr_array_neurongroup__spikespace[j];
				//_ptr_array_neurongroup__spikespace[j] = -1;
				i++;
			}
		}
	_ptr_array_neurongroup__spikespace[N] = i;
	}
	__syncthreads();
}


void _run_neurongroup_thresholder_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const unsigned int _num_spikespace = 2;
	const double t = defaultclock.t_();
	const int _numv = 1;

	///// POINTERS ////////////

	_run_neurongroup_thresholder_codeobject_kernel<<<1,N>>>(_num_spikespace, defaultclock.t_(), _numv, dev_array_neurongroup__spikespace, dev_array_neurongroup_v);
}


