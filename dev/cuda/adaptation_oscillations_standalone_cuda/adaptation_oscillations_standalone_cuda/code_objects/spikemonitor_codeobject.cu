#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define N 4000
#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_spikemonitor_codeobject_kernel(int stride, double par_clock_t, int32_t* par_array_neurongroup__spikespace)
{
	using namespace brian;

	int bid = blockIdx.x;

	int32_t* _array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double clock_t = par_clock_t;

	for(int i = bid * stride; i < (bid + 1)*stride && i < N; i++)
	{
		if(_array_neurongroup__spikespace[i] != -1)
		{
			(_dynamic_array_spikemonitor_i[bid])->push(i);
			(_dynamic_array_spikemonitor_t[bid])->push(clock_t);
		}
		else
		{
			return;
		}
	}
}

void _run_spikemonitor_codeobject()
{
	using namespace brian;
	const double _clock_t = defaultclock.t_();

	_run_spikemonitor_codeobject_kernel<<<num_blocks_sequential,1>>>(ceil(N, num_blocks_sequential), _clock_t, dev_array_neurongroup__spikespace);
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

