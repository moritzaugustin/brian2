#include "objects.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject_1.h"
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


void _run_neurongroup_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numw = 4000;
const int _numnot_refractory = 4000;
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_neurongroup_w = _array_neurongroup_w;
	bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<4000; _idx++)
	{
		const int _vectorisation_idx = _idx;
	const double w = _ptr_array_neurongroup_w[_idx];
	const bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
	const bool _cond = true;
	if(_cond)
	{
		double w = _ptr_array_neurongroup_w[_idx];
		const bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
		if(not_refractory)
		    w = _rand(_vectorisation_idx) * 10 * 0.0001;
		_ptr_array_neurongroup_w[_idx] = w;
	}
	}
}

