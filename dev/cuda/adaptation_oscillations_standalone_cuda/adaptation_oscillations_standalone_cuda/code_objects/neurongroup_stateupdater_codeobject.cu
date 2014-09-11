#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define neuron_N 4000
#define THREADS 1024
#define BLOCKS (neuron_N + THREADS -1)/THREADS

namespace {
	__device__ int int_(const bool value)
	{
	    return value ? 1 : 0;
	}
}

__global__ void _run_neurongroup_stateupdater_codeobject_kernel(
	double par_t,
	double par_dt,
	float* par_array_random_floats,
	double* par_array_neurongroup_v,
	double* par_array_neurongroup_w,
	double* par_array_neurongroup_lastspike,
	bool* par_array_neurongroup_not_refractory)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int neuron_id = bid * THREADS + tid;

	if(neuron_id < 0 || neuron_id >= neuron_N)
	{
		return;
	}

	double t = par_t;
	double dt = par_dt;
	float* _ptr_array_random_floats = par_array_random_floats;
	double* _ptr_array_neurongroup_v = par_array_neurongroup_v;
	double* _ptr_array_neurongroup_w = par_array_neurongroup_w;
	double* _ptr_array_neurongroup_lastspike = par_array_neurongroup_lastspike;
	bool* _ptr_array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;

	double w = _ptr_array_neurongroup_w[neuron_id];
	double v = _ptr_array_neurongroup_v[neuron_id];
	double lastspike = _ptr_array_neurongroup_lastspike[neuron_id];
	bool not_refractory = _ptr_array_neurongroup_not_refractory[neuron_id];

	not_refractory = t - lastspike > 0.0025;
	float r = _ptr_array_random_floats[neuron_id]; //get random pregenerated number
	const double xi = pow(dt, 0.5) * r;
	const double _w = -(dt) * w * int_(not_refractory) / 0.2 + w;
	const double _v = dt * (0.14 * int_(not_refractory) - v * int_(not_refractory) / 0.01 - w * int_(not_refractory) / 0.01) + v + 0.002213594362117866 * xi * int_(not_refractory);
	if(not_refractory)
	{
		w = _w;
	}
	if(not_refractory)
	{
		v = _v;
	}

	_ptr_array_neurongroup_not_refractory[neuron_id] = not_refractory;
	_ptr_array_neurongroup_w[neuron_id] = w;
	_ptr_array_neurongroup_v[neuron_id] = v;
}

void _run_neurongroup_stateupdater_codeobject()
{
	using namespace brian;

	const double t = defaultclock.t_();
	const double dt = defaultclock.dt_();

	_run_neurongroup_stateupdater_codeobject_kernel<<<BLOCKS, THREADS>>>(
		t,
		dt,
		dev_array_random_floats,
		dev_array_neurongroup_v,
		dev_array_neurongroup_w,
		dev_array_neurongroup_lastspike,
		dev_array_neurongroup_not_refractory
		);
}

