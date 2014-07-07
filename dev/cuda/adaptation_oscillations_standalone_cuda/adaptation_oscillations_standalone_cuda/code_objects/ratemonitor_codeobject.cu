#include "objects.h"
#include "code_objects/ratemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define N 4000

__global__ void _run_ratemonitor_codeobject_kernel(double par_t, double par_dt, int32_t* par_array_neurongroup__spikespace, double* par_array_ratemonitor_rate, double* par_array_ratemonitor_t, int par_numt, int par_numrate)
{
	using namespace brian;

	double _clock_t = par_t;
	double _clock_dt = par_dt;
	int32_t* array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double* array_ratemonitor_rate = par_array_ratemonitor_rate;
	double* array_ratemonitor_t = par_array_ratemonitor_t;
	int numt = par_numt;
	int numrate = par_numrate;

	int _num_spikes = array_neurongroup__spikespace[N];
	array_ratemonitor_rate[numrate] = 1.0*_num_spikes/_clock_dt/N;
	array_ratemonitor_t[numt] = _clock_t;
}


void _run_ratemonitor_codeobject()
{
	using namespace brian;
	const double _clock_t = defaultclock.t_();
	const double _clock_dt = defaultclock.dt_();
	const int _numt = _dynamic_array_ratemonitor_t.size();
	const int _numrate = _dynamic_array_ratemonitor_rate.size();

	_dynamic_array_ratemonitor_t.resize(_numt + 1);
	_dynamic_array_ratemonitor_rate.resize(_numrate + 1);

	double* dev_array_ratemonitor_t = thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_t[0]);
	double* dev_array_ratemonitor_rate = thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_rate[0]);

	_run_ratemonitor_codeobject_kernel<<<1,1>>>(_clock_t, _clock_dt, dev_array_neurongroup__spikespace, dev_array_ratemonitor_rate, dev_array_ratemonitor_t, _numt, _numrate);
}

