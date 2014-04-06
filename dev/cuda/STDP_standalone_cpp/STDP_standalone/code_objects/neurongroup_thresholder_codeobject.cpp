#include "objects.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_neurongroup_thresholder_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_spikespace = 2;
const double t = defaultclock.t_();
const int _numv = 1;
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;

	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<1; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		const double v = _ptr_array_neurongroup_v[_idx];
		const double _cond = v > -0.054;
		if(_cond) {
			_ptr_array_neurongroup__spikespace[_cpp_numspikes++] = _idx;
		}
	}
	_ptr_array_neurongroup__spikespace[1] = _cpp_numspikes;
}


