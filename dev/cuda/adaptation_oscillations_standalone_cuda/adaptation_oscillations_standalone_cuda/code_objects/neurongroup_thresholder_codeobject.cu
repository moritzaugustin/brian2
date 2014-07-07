#include "objects.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define N 4000
#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_neurongroup_thresholder_codeobject_kernel(int stride, double par_t, int32_t* par_array_neurongroup__spikespace, double* par_array_neurongroup_v, double* par_array_neurongroup_lastspike, bool* par_array_neurongroup_not_refractory)
{
	using namespace brian;

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	if(bid*stride + tid >= N)
		return;

	double t = par_t;
	int32_t* array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double* array_neurongroup_v = par_array_neurongroup_v;
	double* array_neurongroup_lastspike = par_array_neurongroup_lastspike;
	bool* array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;


	array_neurongroup__spikespace[bid * stride + tid] = -1;
	if(tid == bid && bid == 0)
	{
		array_neurongroup__spikespace[N] = 0;
	}

	if(tid == 0)
	{
		int num_spikes = 0;

		for(int i = bid * stride; i < (bid + 1)*stride; i++)
		{
			const double v = array_neurongroup_v[i];
			const bool not_refractory = array_neurongroup_not_refractory[i];
			const double _cond = (v > 0.001) && (not_refractory);
			if(_cond && i < N)
			{
				array_neurongroup__spikespace[bid*stride + num_spikes] = i;
				array_neurongroup_not_refractory[i] = false;
				array_neurongroup_lastspike[i] = t;
				num_spikes++;
			}
		}
		atomicAdd(&array_neurongroup__spikespace[N], num_spikes);
	}
}

void _run_neurongroup_thresholder_codeobject()
{
	using namespace brian;

	const double t = defaultclock.t_();

	//// MAIN CODE ////////////
	_run_neurongroup_thresholder_codeobject_kernel<<<num_blocks_sequential, ceil(N, num_blocks_sequential)>>>(ceil(N, num_blocks_sequential), t, dev_array_neurongroup__spikespace, dev_array_neurongroup_v, dev_array_neurongroup_lastspike, dev_array_neurongroup_not_refractory);
}


