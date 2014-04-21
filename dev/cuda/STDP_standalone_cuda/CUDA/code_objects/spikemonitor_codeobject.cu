#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////

#define N 1000

struct is_in_range
{
	__host__ __device__ bool operator()(const int32_t x) const
	{
		//return true;
		return x >= 0 && x < 1000;
	}
};

void _run_spikemonitor_codeobject()
{
	using namespace brian;

	const double _clock_t = defaultclock.t_();
	const int _numt = _dynamic_array_spikemonitor_t.size();
	const int _numi = _dynamic_array_spikemonitor_i.size();

	int num_spikes = thrust::count_if(thrust::device,
		dev_array_poissongroup__spikespace,
		dev_array_poissongroup__spikespace + _num__array_poissongroup__spikespace - 1,
		is_in_range());

	_dynamic_array_spikemonitor_t.resize(_numt + num_spikes);
	_dynamic_array_spikemonitor_i.resize(_numi + num_spikes);

	double* dev_array_spikemonitor_t = thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t[_numt - 1]);
	int32_t* dev_array_spikemonitor_i = thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i[_numi - 1]);

	thrust::copy_if(thrust::device, dev_array_poissongroup__spikespace,
		dev_array_poissongroup__spikespace + _num__array_poissongroup__spikespace - 1,
		dev_array_spikemonitor_i, is_in_range());
	thrust::fill_n(thrust::device, dev_array_spikemonitor_t, num_spikes, _clock_t);
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;
	std::cout << "Number of spikes: " << _dynamic_array_spikemonitor_i.size() << endl;
}
