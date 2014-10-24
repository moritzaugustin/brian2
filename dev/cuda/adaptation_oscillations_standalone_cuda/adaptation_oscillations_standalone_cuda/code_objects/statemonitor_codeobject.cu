#include "objects.h"
#include "code_objects/statemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


__global__ void _run_statemonitor_codeobject_kernel(
	int par_neuron_id,
	int par_index_last_element,
	double* par_statemonitor_v,
	double* par_statemonitor_w,
	double* par_array_neurongroup_v,
	double* par_array_neurongroup_w
	)
{
	int neuron_id = par_neuron_id;
	int index_last_element = par_index_last_element;
	double* statemonitor_v = par_statemonitor_v;
	double* statemonitor_w = par_statemonitor_w;
	double* array_neurongroup_v = par_array_neurongroup_v;
	double* array_neurongroup_w = par_array_neurongroup_w;

	statemonitor_v[index_last_element] = array_neurongroup_v[neuron_id];
	statemonitor_w[index_last_element] = array_neurongroup_w[neuron_id];
}

void _run_statemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();
	int num_indices = _num__array_statemonitor__indices;
	_dynamic_array_statemonitor_t.push_back(t);

	for(int i = 0; i < num_indices; i++)
	{
		unsigned int neuron_id = _static_array__array_statemonitor__indices[i];
		_dynamic_array_statemonitor__recorded_v[i].push_back(0.0);	//push dummy value
		_dynamic_array_statemonitor__recorded_w[i].push_back(0.0);	//push dummy value
		double* dev_dynamic_array_statemonitor__recorded_v = thrust::raw_pointer_cast(&(_dynamic_array_statemonitor__recorded_v[i][0]));
		double* dev_dynamic_array_statemonitor__recorded_w = thrust::raw_pointer_cast(&(_dynamic_array_statemonitor__recorded_w[i][0]));
		int index_last_element = _dynamic_array_statemonitor__recorded_v[i].size() - 1;
		_run_statemonitor_codeobject_kernel<<<1, 1>>>(
			neuron_id,
			index_last_element,
			dev_dynamic_array_statemonitor__recorded_v,
			dev_dynamic_array_statemonitor__recorded_w,
			dev_array_neurongroup_v,
			dev_array_neurongroup_w);
	}
}

