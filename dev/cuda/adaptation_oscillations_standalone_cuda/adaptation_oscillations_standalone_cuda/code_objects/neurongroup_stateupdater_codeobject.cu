#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

namespace {
	__device__ int int_(const bool value)
	{
	    return value ? 1 : 0;
	}
}

__global__ void _run_neurongroup_stateupdater_codeobject_kernel(
	unsigned int _neurongroup_N,
	unsigned int max_threads_per_block,
	double _t,
	double _dt,
	float* _array_random_floats,
	double* _array_neurongroup_v,
	double* _array_neurongroup_w,
	double* _array_neurongroup_lastspike,
	bool* _array_neurongroup_not_refractory)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int neuron_id = bid * max_threads_per_block + tid;

	if(neuron_id < 0 || neuron_id >= _neurongroup_N)
	{
		return;
	}

	double w = _array_neurongroup_w[neuron_id];
	double v = _array_neurongroup_v[neuron_id];
	double lastspike = _array_neurongroup_lastspike[neuron_id];
	bool not_refractory = _array_neurongroup_not_refractory[neuron_id];

	not_refractory = _t - lastspike > 0.0025;
	float r = _array_random_floats[neuron_id]; //get random pregenerated number
	const double xi = pow(_dt, 0.5) * r;
	const double _w = -(_dt) * w * int_(not_refractory) / 0.2 + w;
	const double _v = _dt * (0.14 * int_(not_refractory) - v * int_(not_refractory) / 0.01 - w * int_(not_refractory) / 0.01) + v + 0.002213594362117866 * xi * int_(not_refractory);
	if(not_refractory)
	{
		w = _w;
		v = _v;
	}

	_array_neurongroup_not_refractory[neuron_id] = not_refractory;
	_array_neurongroup_w[neuron_id] = w;
	_array_neurongroup_v[neuron_id] = v;
}

void _run_neurongroup_stateupdater_codeobject()
{
	using namespace brian;

	const double t = defaultclock.t_();
	const double dt = defaultclock.dt_();

	unsigned int blocks = (neurongroup_N + max_threads_per_block - 1)/max_threads_per_block;	// = ceil(N/num_threads)

	_run_neurongroup_stateupdater_codeobject_kernel<<<blocks, max_threads_per_block>>>(
		neurongroup_N,
		max_threads_per_block,
		t,
		dt,
		dev_array_random_floats,
		dev_array_neurongroup_v,
		dev_array_neurongroup_w,
		dev_array_neurongroup_lastspike,
		dev_array_neurongroup_not_refractory
		);
}

