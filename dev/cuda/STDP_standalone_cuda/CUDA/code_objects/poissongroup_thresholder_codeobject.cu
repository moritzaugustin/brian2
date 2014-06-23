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

__global__ void _run_poissongroup_thresholder_codeobject_kernel(int stride, float* par_rands,
	int par_numrates, double par_t, int par_numspikespace, double par_dt,
	double* par_array_poissongroup_rates, int32_t*  par_array_poissongroup__spikespace)
{
	extern __shared__ double data[];
	int rates_offset = stride;
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	double * _ptr_array_poissongroup_rates = par_array_poissongroup_rates;	

	par_array_poissongroup__spikespace[bid * stride + tid] = -1;

	if(bid*stride + tid >= N)
		return;

	//befüllen des shared memory
	data[tid] = par_rands[bid * stride + tid];
	data[rates_offset + tid] = _ptr_array_poissongroup_rates[bid * stride + tid];

	//const int _numrates = par_numrates;
	//const double t = par_t;
	//const int _num_spikespace = par_numspikespace;
	
	if(tid == 0)
	{
		const double dt = par_dt;
		int32_t * _ptr_array_poissongroup__spikespace = par_array_poissongroup__spikespace;
		int num_spikes = 0;

		for(int i = bid * stride; i < (bid + 1)*stride; i++)
		{
			int index = i % stride;
			const double _cond = data[index] < data[rates_offset + index] * dt;
			if(_cond && i < N)
			{
				_ptr_array_poissongroup__spikespace[bid * stride + num_spikes] = i;
				num_spikes++;
			}
		}
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
	_run_poissongroup_thresholder_codeobject_kernel<<<num_blocks_sequential, 1024/num_blocks_sequential, 2*sizeof(double)*1024/num_blocks_sequential>>>(1024/num_blocks_sequential, dev_array_rands, _numrates, t,
		_num_spikespace, dt, dev_array_poissongroup_rates, dev_array_poissongroup__spikespace);
}
