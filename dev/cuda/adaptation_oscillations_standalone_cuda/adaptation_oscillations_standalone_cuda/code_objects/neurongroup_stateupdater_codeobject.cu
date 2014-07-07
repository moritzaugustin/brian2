#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define N 4000
#define ceil(N, num) ((N + num-1)/num)

namespace {
	__device__ int int_(const bool value)
	{
	    return value ? 1 : 0;
	}
}


__global__ void _run_neurongroup_stateupdater_codeobject_kernel(int par_N, float* par_array_random, double par_t, double par_dt, double* par_array_neurongroup_w, double* par_array_neurongroup_v, double* par_array_neurongroup_lastspike, bool* par_array_neurongroup_not_refractory)
{
	using namespace brian;

	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int idx = bid * 1024 + tid;

	if(idx >= par_N)
	{
		return;
	}

	double dt = par_dt;
	double t = par_t;
	float* _ptr_array_random = par_array_random;
	double* _ptr_array_neurongroup_w = par_array_neurongroup_w;
	double* _ptr_array_neurongroup_v = par_array_neurongroup_v;
	double* _ptr_array_neurongroup_lastspike = par_array_neurongroup_lastspike;
	bool* _ptr_array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;

	double w = _ptr_array_neurongroup_w[idx];
	double v = _ptr_array_neurongroup_v[idx];
	const double lastspike = _ptr_array_neurongroup_lastspike[idx];
	bool not_refractory = _ptr_array_neurongroup_not_refractory[idx];
	not_refractory = t - lastspike > 0.0025;
	float r = _ptr_array_random[idx];
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
	_ptr_array_neurongroup_not_refractory[idx] = not_refractory;
	_ptr_array_neurongroup_w[idx] = w;
	_ptr_array_neurongroup_v[idx] = v;
}

void _run_neurongroup_stateupdater_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();
	double dt = defaultclock.dt_();

	_run_neurongroup_stateupdater_codeobject_kernel<<<ceil(N, 1024),1024>>>(N, dev_array_random, t, dt, dev_array_neurongroup_w, dev_array_neurongroup_v, dev_array_neurongroup_lastspike, dev_array_neurongroup_not_refractory);
}

