#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_synapses_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_synapses_c = &_dynamic_array_synapses_c[0];
const int _numc = _dynamic_array_synapses_c.size();
const int64_t N = synapses._N();
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_synapses_c = _array_synapses_c;

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<N; _idx++)
	{
		const int _vectorisation_idx = _idx;
	const bool _cond = true;
	if(_cond)
	{
		double c;
		c = 5.15e-06;
		_ptr_array_synapses_c[_idx] = c;
	}
	}
}


