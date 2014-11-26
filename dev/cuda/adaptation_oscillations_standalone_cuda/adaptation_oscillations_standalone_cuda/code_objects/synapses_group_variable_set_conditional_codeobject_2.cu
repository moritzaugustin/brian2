#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_2.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_synapses_group_variable_set_conditional_codeobject_2_kernel(
	int par_syn_N,
	unsigned int max_threads_per_block,
	double* par_array_synapses_lastupdate)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = par_syn_N;
	double* _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;

	int syn_id = bid*max_threads_per_block + tid;
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

	unsigned int blocks = (syn_N + max_threads_per_block - 1)/max_threads_per_block;	// = ceil(N/num_threads)
	_run_synapses_group_variable_set_conditional_codeobject_2_kernel<<<blocks, max_threads_per_block>>>(
		syn_N,
		max_threads_per_block,
		dev_array_synapses_lastupdate);
}

