#include "objects.h"
#include "code_objects/statemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////

__global__ void _run_statemonitor_codeobject_kernel(double* par_array_synapses_w, int par_index, double* par_statemonitor__recorded_w)
{
	double* _ptr_array_synapses_w = par_array_synapses_w;
	int _idx = par_index;
	double* statemonitor__recorded_w = par_statemonitor__recorded_w;

	const double w = _ptr_array_synapses_w[_idx];
	const double _to_record_w = w;

	*statemonitor__recorded_w = _to_record_w;
}

void _run_statemonitor_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_indices = 2;
	const double _clock_t = defaultclock.t_();
	double* dev_array_synapses_w = thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0]);
	int32_t * _ptr_array_statemonitor__indices = _array_statemonitor__indices;

	_dynamic_array_statemonitor_t.push_back(_clock_t);

	for(int i = 0; i < _num_indices; i++)
	{
		int size = _dynamic_array_statemonitor__recorded_w[i].size();
		_dynamic_array_statemonitor__recorded_w[i].resize(size + 1);
		int index = _ptr_array_statemonitor__indices[i];
		double* dev_statemonitor__recorded_w = thrust::raw_pointer_cast(&_dynamic_array_statemonitor__recorded_w[i][size]);
		_run_statemonitor_codeobject_kernel<<<1, 1>>>(dev_array_synapses_w, index, dev_statemonitor__recorded_w);
	}
}


