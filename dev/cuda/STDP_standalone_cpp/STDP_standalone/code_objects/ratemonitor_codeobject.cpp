#include "objects.h"
#include "code_objects/ratemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_ratemonitor_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const double _clock_t = defaultclock.t_();
const double _clock_dt = defaultclock.dt_();
const int _num_spikespace = 1001;
double* const _array_ratemonitor_t = &_dynamic_array_ratemonitor_t[0];
const int _numt = _dynamic_array_ratemonitor_t.size();
double* const _array_ratemonitor_rate = &_dynamic_array_ratemonitor_rate[0];
const int _numrate = _dynamic_array_ratemonitor_rate.size();
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_poissongroup__spikespace = _array_poissongroup__spikespace;
	double * __restrict__ _ptr_array_ratemonitor_t = _array_ratemonitor_t;
	double * __restrict__ _ptr_array_ratemonitor_rate = _array_ratemonitor_rate;


	int _num_spikes = _ptr_array_poissongroup__spikespace[_num_spikespace-1];
	_dynamic_array_ratemonitor_rate.push_back(1.0*_num_spikes/_clock_dt/1000);
	_dynamic_array_ratemonitor_t.push_back(_clock_t);
}


