#include "objects.h"
#include "code_objects/ratemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

void _run_ratemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();
	double dt = defaultclock.dt_();

	//spikespace is already on CPU-side, so we can just read the last element (=number of spikes)
	int32_t num_spikes = _array_neurongroup__spikespace[neurongroup_N];
	_dynamic_array_ratemonitor_t.push_back(t);
	_dynamic_array_ratemonitor_rate.push_back(1.0*num_spikes/dt/neurongroup_N);
}

