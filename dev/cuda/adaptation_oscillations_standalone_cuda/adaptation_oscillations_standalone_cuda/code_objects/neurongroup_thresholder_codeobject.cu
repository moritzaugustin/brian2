#include "objects.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define neuron_N 4000
#define MEM_PER_THREAD (sizeof(bool))
#define THREADS (4000 + BLOCKS - 1)/(BLOCKS)
#define BLOCKS (num_blocks_sequential)

__global__ void _run_neurongroup_thresholder_codeobject_kernel(
	int par_num_thread_per_block,
	double par_t,
	int32_t* par_array_neurongroup__spikespace,
	double* par_array_neurongroup_v,
	double* par_array_neurongroup_lastspike,
	bool* par_array_neurongroup_not_refractory)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int num_threads_per_block = par_num_thread_per_block;

	int neuron_id = bid*num_threads_per_block + tid;
	if(neuron_id < 0 || neuron_id >= neuron_N)
	{
		return;
	}

	extern __shared__ bool spike_cache[];
	double t = par_t;
	int32_t* _ptr_array_neurongroup__spikespace = par_array_neurongroup__spikespace;
	double* _ptr_array_neurongroup_v = par_array_neurongroup_v;
	double* _ptr_array_neurongroup_lastspike = par_array_neurongroup_lastspike;
	bool* _ptr_array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;

	spike_cache[tid] = false;
	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		_ptr_array_neurongroup__spikespace[neuron_N] = 0;
	}

	double v = _ptr_array_neurongroup_v[neuron_id];
	bool not_refractory = _ptr_array_neurongroup_not_refractory[neuron_id];
	spike_cache[tid] = (v > 0.001) && (not_refractory);

	//only one thread per block iterates over the cache
	if(tid != 0)
	{
		return;
	}

	int num_spikes_in_block = 0;
	for(int i = 0; (i < num_threads_per_block) && (neuron_id + i < neuron_N); i++)
	{
		if(spike_cache[i])
		{
			int spiking_neuron = neuron_id + i;
			_ptr_array_neurongroup__spikespace[neuron_id + num_spikes_in_block] = spiking_neuron;
			_ptr_array_neurongroup_not_refractory[spiking_neuron] = false;
			_ptr_array_neurongroup_lastspike[spiking_neuron] = t;
			num_spikes_in_block++;
		}
		//add all blocks together
		atomicAdd(&_ptr_array_neurongroup__spikespace[neuron_N], num_spikes_in_block);
	}
}

void _run_neurongroup_thresholder_codeobject()
{
	using namespace brian;

	const double t = defaultclock.t_();

	_run_neurongroup_thresholder_codeobject_kernel<<<BLOCKS, THREADS, THREADS*MEM_PER_THREAD>>>(
		THREADS,
		t,
		dev_array_neurongroup__spikespace,
		dev_array_neurongroup_v,
		dev_array_neurongroup_lastspike,
		dev_array_neurongroup_not_refractory);
}

