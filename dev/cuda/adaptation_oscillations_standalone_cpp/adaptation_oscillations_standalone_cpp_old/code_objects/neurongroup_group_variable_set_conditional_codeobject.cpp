#include "objects.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	double _rand(int vectorisation_idx)
	{
	    return (double)rand()/RAND_MAX;
	}
}

////// HASH DEFINES ///////


void _run_neurongroup_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numv = 4000;
const int _numnot_refractory = 4000;
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;
	bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<4000; _idx++)
	{
		const int _vectorisation_idx = _idx;
	const bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
	const double v = _ptr_array_neurongroup_v[_idx];
	const bool _cond = true;
	if(_cond)
	{
		const bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
		double v = _ptr_array_neurongroup_v[_idx];
		if(not_refractory)
		    v = _rand(_vectorisation_idx) * 0.001;
		_ptr_array_neurongroup_v[_idx] = v;
	}
	}
}


