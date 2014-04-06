#include "objects.h"
#include "code_objects/poissongroup_thresholder_codeobject.h"
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


void _run_poissongroup_thresholder_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numrates = 1000;
const double t = defaultclock.t_();
const int _num_spikespace = 1001;
const double dt = defaultclock.dt_();
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_poissongroup_rates = _array_poissongroup_rates;
	int32_t * __restrict__ _ptr_array_poissongroup__spikespace = _array_poissongroup__spikespace;

	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	long _cpp_numspikes = 0;
	for(int _idx=0; _idx<1000; _idx++)
	{
	    const int _vectorisation_idx = _idx;
		const double rates = _ptr_array_poissongroup_rates[_idx];
		const double _cond = _rand(_vectorisation_idx) < rates * dt;
		if(_cond) {
			_ptr_array_poissongroup__spikespace[_cpp_numspikes++] = _idx;
		}
	}
	_ptr_array_poissongroup__spikespace[1000] = _cpp_numspikes;
}


