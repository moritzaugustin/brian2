#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_synapses_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_synapses_pre_delay = &_dynamic_array_synapses_pre_delay[0];
const int _numdelay = _dynamic_array_synapses_pre_delay.size();
const int64_t N = synapses._N();
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_synapses_pre_delay = _array_synapses_pre_delay;

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<N; _idx++)
	{
		const int _vectorisation_idx = _idx;
	const bool _cond = true;
	if(_cond)
	{
		double delay;
		delay = 0.002;
		_ptr_array_synapses_pre_delay[_idx] = delay;
	}
	}
}


