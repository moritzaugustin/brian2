#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
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

__global__ void _run_spikemonitor_codeobject_kernel(int par_num_spikespace, int par_numt, int par_numi, double par_clock_t, int32_t* par_array_poissongroup__spikespace, double* par_array_spikemonitor_t, int32_t* par_array_spikemonitor_i, int* result)
{
	int tid = threadIdx.x;
	int _num_spikespace = par_num_spikespace;
	int _numt = par_numt;
	int _numi = par_numi;
	double _clock_t = par_clock_t;
	int32_t * _ptr_array_poissongroup__spikespace = par_array_poissongroup__spikespace;
	double * _ptr_array_spikemonitor_t = par_array_spikemonitor_t;
	int32_t * _ptr_array_spikemonitor_i = par_array_spikemonitor_i;
	if(tid == 0)
	{
		*result = 0;
	}

	if(_ptr_array_poissongroup__spikespace[_num_spikespace - 1] == 0)
		return;
	int data = _ptr_array_poissongroup__spikespace[tid];
	int start_idx = __syncthreads_count(data >= 0 && data < 0);
	int _num_spikes = __syncthreads_count(data >= 0 && data < 1000);
	if(tid >= _num_spikes)
		return;
	_ptr_array_spikemonitor_i[_numi + tid] = _ptr_array_poissongroup__spikespace[start_idx + tid] - 0;
	_ptr_array_spikemonitor_t[_numt + tid] = _clock_t;
	if(tid == 0)
	{
		*result = _num_spikes;
	}
	__syncthreads();
}

void _run_spikemonitor_codeobject()
{
	using namespace brian;
	const int _num_spikespace = 1001;
	const int _numt = _dynamic_array_spikemonitor_t.size();
	const int _numi = _dynamic_array_spikemonitor_i.size();
	const double _clock_t = defaultclock.t_();
	int* dev_result;
	int result;
	cudaMalloc((void**)&dev_result, sizeof(int));

	_dynamic_array_spikemonitor_t.resize(_numt + N);
	_dynamic_array_spikemonitor_i.resize(_numi + N);

	double* const dev_array_spikemonitor_t = thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t[0]);
	int32_t* const dev_array_spikemonitor_i = thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i[0]);


	_run_spikemonitor_codeobject_kernel<<<1, N>>>(_num_spikespace, _numt, _numi, _clock_t, dev_array_poissongroup__spikespace, dev_array_spikemonitor_t, dev_array_spikemonitor_i, dev_result);

	cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
	
	_dynamic_array_spikemonitor_t.resize(_numt + result);
	_dynamic_array_spikemonitor_i.resize(_numi + result);
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;
	std::cout << "Number of spikes: " << _dynamic_array_spikemonitor_i.size() << endl;
}

