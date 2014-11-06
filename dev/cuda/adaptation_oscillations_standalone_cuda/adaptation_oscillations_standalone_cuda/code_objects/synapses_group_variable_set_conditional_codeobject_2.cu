#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_2.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS -1)/THREADS

__global__ void _run_synapses_group_variable_set_conditional_codeobject_2_kernel(
	int par_syn_N,
	double* par_array_synapses_lastupdate)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = par_syn_N;
	double* _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;

	int syn_id = bid*THREADS + tid;
	if(syn_id < 0 || syn_id >= syn_N)
	{
		return;
	}

	const bool _cond = true;
	if(_cond)
	{
		double lastupdate;
		lastupdate = 0.0 * 1.0;
		_ptr_array_synapses_lastupdate[syn_id] = lastupdate;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject_2()
{
	using namespace brian;

	const int syn_N = synapses._N();
	double* const dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);

	_run_synapses_group_variable_set_conditional_codeobject_2_kernel<<<BLOCKS(syn_N), THREADS>>>(
		syn_N,
		dev_array_synapses_lastupdate);
}

