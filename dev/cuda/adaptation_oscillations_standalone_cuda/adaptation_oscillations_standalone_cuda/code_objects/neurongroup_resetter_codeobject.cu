#include "objects.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define N 4000
#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_neurongroup_resetter_codeobject_kernel(int par_N, int32_t* par_array_neurongroup__spikespace, double* par_array_neurongroup_w, double* par_array_neurongroup_v)
{
	using namespace brian;

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * 1024 + tid;

	if(idx >= par_N)
	{
		return;
	}

	int32_t* _ptr_array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double* _ptr_array_neurongroup_w = par_array_neurongroup_w;
	double* _ptr_array_neurongroup_v = par_array_neurongroup_v;

	int _idx = _ptr_array_neurongroup__spikespace[idx];
	if(_idx != -1)
	{
		double w = _ptr_array_neurongroup_w[_idx];
		double v;
		v = 0.0;
		w = w + 0.0001;
		_ptr_array_neurongroup_w[_idx] = w;
		_ptr_array_neurongroup_v[_idx] = v;
	}
}

void _run_neurongroup_resetter_codeobject()
{
	using namespace brian;

	_run_neurongroup_resetter_codeobject_kernel<<<ceil(N, 1024),1024>>>(N, dev_array_neurongroup__spikespace, dev_array_neurongroup_w, dev_array_neurongroup_v);
	
}

