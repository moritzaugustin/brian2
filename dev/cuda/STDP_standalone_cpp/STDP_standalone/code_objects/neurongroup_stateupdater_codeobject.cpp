#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_neurongroup_stateupdater_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numge = 1;
const int _numv = 1;
const double dt = defaultclock.dt_();
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_neurongroup_ge = _array_neurongroup_ge;
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<1; _idx++)
	{
		const int _vectorisation_idx = _idx;
			double ge = _ptr_array_neurongroup_ge[_idx];
			double v = _ptr_array_neurongroup_v[_idx];
			const double _ge = ge * exp(-(dt) / 0.005);
			const double _BA_v = -(0.0) * ge - -0.074 + ge * -0.06;
			const double _v = -(_BA_v) + (_BA_v + v) * exp(-(dt) / 0.01);
			ge = _ge;
			v = _v;
			_ptr_array_neurongroup_ge[_idx] = ge;
			_ptr_array_neurongroup_v[_idx] = v;
	}
}


