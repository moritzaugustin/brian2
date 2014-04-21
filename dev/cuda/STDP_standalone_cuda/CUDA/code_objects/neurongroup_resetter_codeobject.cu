#include "objects.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////

#define N 1

__global__ void _run_neurongroup_resetter_codeobject_kernel(int par_num_spikespace,
	int par_numv, int32_t* par_array_neurongroup__spikespace,
	double* par_array_neurongroup_v)
{
	int tid = threadIdx.x;
	//int _num_spikespace = par_num_spikespace;
	//int _numv = par_numv;
	int32_t * _ptr_array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double * _ptr_array_neurongroup_v = par_array_neurongroup_v;
	int32_t *_spikes = _ptr_array_neurongroup__spikespace;
	//const int _num_spikes = _ptr_array_neurongroup__spikespace[1];

	const int _idx = _spikes[tid];
	double v;
	v = -0.06;
	if(_idx != -1)
	{
		_ptr_array_neurongroup_v[_idx] = v;
	}
}

void _run_neurongroup_resetter_codeobject()
{
	using namespace brian;

	const int _num_spikespace = 2;
	const int _numv = 1;

	//// MAIN CODE ////////////
	_run_neurongroup_resetter_codeobject_kernel<<<1, N>>>(_num_spikespace,
		_numv, dev_array_neurongroup__spikespace, dev_array_neurongroup_v);
}


