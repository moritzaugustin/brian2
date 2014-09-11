#include "objects.h"
#include "code_objects/statemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_statemonitor_codeobject()
{
/*
	using namespace brian;
	///// CONSTANTS ///////////
	const double _clock_t = defaultclock.t_();
	const int _num_indices = 1;
	double* const _array_statemonitor_t = &_dynamic_array_statemonitor_t[0];
	const int _numt = _dynamic_array_statemonitor_t.size();
	const int _numw = 4000;
	const int _numv = 4000;
	const int _numnot_refractory = 4000;
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_statemonitor__indices = _array_statemonitor__indices;
	double * __restrict__ _ptr_array_statemonitor_t = _array_statemonitor_t;
	double * __restrict__ _ptr_array_neurongroup_w = _array_neurongroup_w;
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;
	bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;
	_dynamic_array_statemonitor_t.push_back(_clock_t);
	const int _new_size = _dynamic_array_statemonitor_t.size();
	// Resize the dynamic arrays
	_dynamic_array_statemonitor__recorded_v.resize(_new_size, _num_indices);
	_dynamic_array_statemonitor__recorded_w.resize(_new_size, _num_indices);

	for (int _i = 0; _i < _num_indices; _i++)
	{
		const int _idx = _ptr_array_statemonitor__indices[_i];
		const int _vectorisation_idx = _idx;
		const bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
		const double w = _ptr_array_neurongroup_w[_idx];
		const double v = _ptr_array_neurongroup_v[_idx];
		const double _to_record_v = v;
		const double _to_record_w = w;
		_dynamic_array_statemonitor__recorded_v(_new_size-1, _i) = _to_record_v;
		_dynamic_array_statemonitor__recorded_w(_new_size-1, _i) = _to_record_w;
	}
*/
}


