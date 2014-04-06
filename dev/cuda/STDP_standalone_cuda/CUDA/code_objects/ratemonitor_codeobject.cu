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

__global__ void _run_ratemonitor_codeobject_kernel(double par_t, double par_dt, int par_num_spikespace, int par_numt, int par_numrate, int32_t* par_array_poissongroup__spikespace, double*  par_array_ratemonitor_t, double* par_array_ratemonitor_rate)
{
	double _clock_t = par_t;
	double _clock_dt = par_dt;
	int _num_spikespace = par_num_spikespace;
	int _numt = par_numt;
	int _numrate = par_numrate;
	int32_t * _ptr_array_poissongroup__spikespace = par_array_poissongroup__spikespace;
	double * _ptr_array_ratemonitor_t = par_array_ratemonitor_t;
	double * _ptr_array_ratemonitor_rate = par_array_ratemonitor_rate;
	
	int _num_spikes = _ptr_array_poissongroup__spikespace[_num_spikespace-1];
	_ptr_array_ratemonitor_rate[_numrate] = 1.0*_num_spikes/_clock_dt/1000;
	_ptr_array_ratemonitor_t[_numt] = _clock_t;
}


void _run_ratemonitor_codeobject()
{
	using namespace brian;
	const double _clock_t = defaultclock.t_();
	const double _clock_dt = defaultclock.dt_();
	const int _num_spikespace = 1001;
	const int _numt = _dynamic_array_ratemonitor_t.size();
	const int _numrate = _dynamic_array_ratemonitor_rate.size();

	_dynamic_array_ratemonitor_t.resize(_numt + 1);
	_dynamic_array_ratemonitor_rate.resize(_numrate + 1);

	double* dev_array_ratemonitor_t = thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_t[0]);
	double* dev_array_ratemonitor_rate = thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_rate[0]);

	_run_ratemonitor_codeobject_kernel<<<1,1>>>(_clock_t, _clock_dt, _num_spikespace, _numt, _numrate, dev_array_poissongroup__spikespace, dev_array_ratemonitor_t, dev_array_ratemonitor_rate);
}


