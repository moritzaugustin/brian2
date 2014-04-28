#include "objects.h"
#include "code_objects/poissongroup_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
}

////// HASH DEFINES ///////

#define N 1000

__device__ int cpp_numspikes;

__global__ void _run_poissongroup_thresholder_codeobject_kernel(float* par_rands,
	int par_numrates, double par_t, int par_numspikespace, double par_dt,
	double* par_array_poissongroup_rates, int32_t*  par_array_poissongroup__spikespace)
{
	int tid = threadIdx.x;

	if(tid == 0)
	{
		cpp_numspikes = 0;
	}

	double r = par_rands[tid];
	//const int _numrates = par_numrates;
	//const double t = par_t;
	//const int _num_spikespace = par_numspikespace;
	const double dt = par_dt;
	double * _ptr_array_poissongroup_rates = par_array_poissongroup_rates;
	int32_t * _ptr_array_poissongroup__spikespace = par_array_poissongroup__spikespace;

	const double rates = _ptr_array_poissongroup_rates[tid];
	//TODO: const double _cond = _rand(_vectorisation_idx) < rates * dt;
	const double _cond = r < rates * dt;
	if(_cond) {
		_ptr_array_poissongroup__spikespace[tid] = tid;
	}
	else {
		_ptr_array_poissongroup__spikespace[tid] = -1;
	}
	int _num_spikes = __syncthreads_count(_cond);
	if(tid == 0)
	{
		_ptr_array_poissongroup__spikespace[N] = _num_spikes;
	}
}

void _run_poissongroup_thresholder_codeobject()
{
	using namespace brian;
	const int _numrates = 1000;
	const double t = defaultclock.t_();
	const int _num_spikespace = 1001;
	const double dt = defaultclock.dt_();

	//// MAIN CODE ////////////
	_run_poissongroup_thresholder_codeobject_kernel<<<1, N>>>(dev_array_rands, _numrates, t,
		_num_spikespace, dt, dev_array_poissongroup_rates, dev_array_poissongroup__spikespace);
}
