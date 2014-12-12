#include "objects.h"
#include "code_objects/ratemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_ratemonitor_codeobject_kernel(
		unsigned int _neurongroup_N,
		double _dt,
		int32_t index,
		int32_t* _array_neurongroup__spikespace,
		double* _dynamic_array_ratemonitor_rate
	)
{
	unsigned int num_spikes = _array_neurongroup__spikespace[_neurongroup_N];
	_dynamic_array_ratemonitor_rate[index] = 1.0*num_spikes/_dt/_neurongroup_N;
}

void _run_ratemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();
	double dt = defaultclock.dt_();

	_dynamic_array_ratemonitor_t.push_back(t);
	_dynamic_array_ratemonitor_rate.push_back(0.0);	//push dummy value

	double* dev_dynamic_array_ratemonitor_rate = thrust::raw_pointer_cast(&(_dynamic_array_ratemonitor_rate[0]));
	int index_last_element = _dynamic_array_ratemonitor_rate.size() - 1;	

	_run_ratemonitor_codeobject_kernel<<<1,1>>>(
		neurongroup_N,
		dt,
		index_last_element,
		dev_array_neurongroup__spikespace,
		dev_dynamic_array_ratemonitor_rate);
}

