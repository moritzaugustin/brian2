#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS -1)/THREADS

__global__ void _run_synapses_group_variable_set_conditional_codeobject_1_kernel(
	int par_syn_N,
	double* par_array_synapses_pre_delay)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = par_syn_N;
	double* _ptr_array_synapses_pre_delay = par_array_synapses_pre_delay;

	int syn_id = bid*THREADS + tid;
	if(syn_id < 0 || syn_id >= syn_N)
	{
		return;
	}

	const bool _cond = true;
	if(_cond)
	{
		double delay;
		delay = 0.002;
		_ptr_array_synapses_pre_delay[syn_id] = delay;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;

	const int64_t syn_N = synapses._N();
	double* const dev_array_synapses_pre_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]);

	_run_synapses_group_variable_set_conditional_codeobject_1_kernel<<<BLOCKS(syn_N), THREADS>>>(
		syn_N,
		dev_array_synapses_pre_delay);
}

