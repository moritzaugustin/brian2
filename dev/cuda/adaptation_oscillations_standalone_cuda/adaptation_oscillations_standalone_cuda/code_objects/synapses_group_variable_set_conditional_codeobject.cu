#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define THREADS 1024
#define BLOCKS(N) ((N + THREADS -1)/THREADS)

__global__ void _run_synapses_group_variable_set_conditional_codeobject_kernel(
	int par_syn_N,
	double* par_array_synapses_c)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int syn_N = par_syn_N;
	double* _ptr_array_synapses_c = par_array_synapses_c;

	int syn_id = bid*THREADS + tid;
	if(syn_id < 0 || syn_id >= syn_N)
	{
		return;
	}

	const bool _cond = true;
	if(_cond)
	{
		double c;
		c = 5.15e-06;
		_ptr_array_synapses_c[syn_id] = c;
	}
}

void _run_synapses_group_variable_set_conditional_codeobject()
{
	using namespace brian;

	const int64_t syn_N = synapses._N();
	double* const dev_array_synapses_c = thrust::raw_pointer_cast(&_dynamic_array_synapses_c[0]);

	_run_synapses_group_variable_set_conditional_codeobject_kernel<<<BLOCKS(syn_N), THREADS>>>(
		syn_N,
		dev_array_synapses_c);
}

