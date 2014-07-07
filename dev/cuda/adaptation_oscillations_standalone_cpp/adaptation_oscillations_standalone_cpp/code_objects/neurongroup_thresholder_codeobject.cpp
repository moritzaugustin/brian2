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
	const int _num_spikespace = 4001;
const double t = defaultclock.t_();
const int _numv = 4000;
const int _numlastspike = 4000;
const int _numnot_refractory = 4000;
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;
	double * __restrict__ _ptr_array_neurongroup_lastspike = _array_neurongroup_lastspike;
	bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;

	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<4000; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		const double v = _ptr_array_neurongroup_v[_idx];
		const bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
		const double _cond = (v > 0.001) && (not_refractory);
		if(_cond) {
			_ptr_array_neurongroup__spikespace[_cpp_numspikes++] = _idx;
			// We have to use the pointer names directly here: The condition
			// might contain references to not_refractory or lastspike and in
			// that case the names will refer to a single entry.
			_ptr_array_neurongroup_not_refractory[_idx] = false;
			_ptr_array_neurongroup_lastspike[_idx] = t;
		}
	}
	_ptr_array_neurongroup__spikespace[4000] = _cpp_numspikes;
}


