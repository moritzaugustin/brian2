#include "objects.h"
#include "code_objects/statemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_statemonitor_codeobject_kernel(int par_index, double* par_statemonitor__recorded_v, double* par_statemonitor__recorded_w, double* par_array_neurongroup_v, double* par_array_neurongroup_w)
{
	int idx = par_index;

	double* statemonitor__recorded_v = par_statemonitor__recorded_v;
	double* statemonitor__recorded_w = par_statemonitor__recorded_w;
	double* array_neurongroup_v = par_array_neurongroup_v;
	double* array_neurongroup_w = par_array_neurongroup_w;

	const double v = array_neurongroup_v[idx];
	const double w = array_neurongroup_w[idx];

	*statemonitor__recorded_v = v;
	*statemonitor__recorded_w = w;
}

void _run_statemonitor_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_indices = 1;
	const double _clock_t = defaultclock.t_();
	int32_t * _ptr_array_statemonitor__indices = _array_statemonitor__indices;

	_dynamic_array_statemonitor_t.push_back(_clock_t);

	for(int i = 0; i < _num_indices; i++)
	{
		int size = _dynamic_array_statemonitor__recorded_v[i].size();
		_dynamic_array_statemonitor__recorded_v[i].resize(size + 1);
		_dynamic_array_statemonitor__recorded_w[i].resize(size + 1);
		int index = _ptr_array_statemonitor__indices[i];
		double* dev_statemonitor__recorded_v = thrust::raw_pointer_cast(&_dynamic_array_statemonitor__recorded_v[i][size]);
		double* dev_statemonitor__recorded_w = thrust::raw_pointer_cast(&_dynamic_array_statemonitor__recorded_w[i][size]);
		_run_statemonitor_codeobject_kernel<<<1, 1>>>(index, dev_statemonitor__recorded_v, dev_statemonitor__recorded_w, dev_array_neurongroup_v, dev_array_neurongroup_w);
	}
}

