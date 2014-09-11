#include "objects.h"
#include "code_objects/ratemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define neuron_N 4000

__global__ void _run_ratemonitor_codeobject_kernel(
	double par_dt,
	int32_t* par_array_neurongroup__spikespace,
	double* par_array_ratemonitor_rate)
{
	double dt = par_dt;
	int32_t* _ptr_array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double* _ptr_array_ratemonitor_rate = par_array_ratemonitor_rate;

	int num_spikes = _ptr_array_neurongroup__spikespace[neuron_N];
	*_ptr_array_ratemonitor_rate = 1.0*num_spikes/dt/neuron_N;
}

void _run_ratemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();
	double dt = defaultclock.dt_();

	_dynamic_array_ratemonitor_t.push_back(t);
	_dynamic_array_ratemonitor_rate.push_back(0.0);
	int num_rate = _dynamic_array_ratemonitor_rate.size();

	double* dev_array_ratemonitor_rate = thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_rate[num_rate - 1]);

	_run_ratemonitor_codeobject_kernel<<<1, 1>>>(
		dt,
		dev_array_neurongroup__spikespace,
		dev_array_ratemonitor_rate);
}

