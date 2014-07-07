#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_2.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel_2(int par_N, double* par_array_synapses_lastupdate)
{
	using namespace brian;

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * 1024 + tid;

	if(idx >= par_N)
	{
		return;
	}

	double* _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;
	const bool _cond = true;

	if(_cond)
	{
		double lastupdate;
		lastupdate = 0.0 * 1.0;
		_ptr_array_synapses_lastupdate[idx] = lastupdate;
	}	
}

void _run_synapses_group_variable_set_conditional_codeobject_2()
{
	using namespace brian;

	const int64_t N = synapses._N();
	double* dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);

	_run_synapses_group_variable_set_conditional_codeobject_kernel_2<<<ceil(N, 1024),1024>>>(N, dev_array_synapses_lastupdate);
}

