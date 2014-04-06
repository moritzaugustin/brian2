#include "objects.h"
#include "code_objects/poissongroup_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////

#define N 1000

__global__ void _run_poissongroup_group_variable_set_conditional_codeobject_kernel(int par_numrates, double* par_array_poissongroup_rates)
{
	int tid = threadIdx.x;
	//int _numrates = par_numrates;
	double * _ptr_array_poissongroup_rates = par_array_poissongroup_rates;

	const bool _cond = true;
	if(_cond)
	{
		double rates;
		rates = 15.0 * 1.0;
		_ptr_array_poissongroup_rates[tid] = rates;
	}
}

void _run_poissongroup_group_variable_set_conditional_codeobject()
{
	using namespace brian;

	const int _numrates = 1000;

	_run_poissongroup_group_variable_set_conditional_codeobject_kernel<<<1,N>>>(_numrates, dev_array_poissongroup_rates);
}


