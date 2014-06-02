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

__global__ void _run_spikemonitor_codeobject_kernel(int par_num_threads, double par_clock_t, int32_t* par_array_poissongroup__spikespace)
{
	using namespace brian;
	int bid = blockIdx.x;

	int _num__array_poissongroup__spikespace = 1000;
	int32_t* _array_poissongroup__spikespace = par_array_poissongroup__spikespace;
	double clock_t = par_clock_t;

	int num_threads = par_num_threads;
	int num_elements = _num__array_poissongroup__spikespace;
	float num_per_thread = (float)num_elements/(float)num_threads;
	int lower = bid*(num_per_thread);
	int upper = (bid + 1)*(num_per_thread);


	for(int i = lower; i < upper; i++)
	{
		if(_array_poissongroup__spikespace[i] != -1)
		{
			(_dynamic_array_spikemonitor_i[bid])->push(i);
			(_dynamic_array_spikemonitor_t[bid])->push(clock_t);
		}
	}
}

void _run_spikemonitor_codeobject()
{
	using namespace brian;
	const double _clock_t = defaultclock.t_();

	_run_spikemonitor_codeobject_kernel<<<num_blocks_sequential,1>>>(num_blocks_sequential, _clock_t, dev_array_poissongroup__spikespace);
}

__global__ void _debugmsg_spikemonitor_codeobject_kernel(int par_num_threads)
{
	using namespace brian;

	int num_threads = par_num_threads;
	int num_spikes = 0;	

	for(int i = 0; i < num_threads; i++)
	{
		num_spikes += (_dynamic_array_spikemonitor_i[i])->size();
	}
	printf("Number of spikes: %d\n", num_spikes);
}


void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;
	_debugmsg_spikemonitor_codeobject_kernel<<<1,1>>>(num_blocks_sequential);
}

