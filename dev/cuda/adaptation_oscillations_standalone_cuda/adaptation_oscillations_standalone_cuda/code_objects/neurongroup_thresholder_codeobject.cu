#include "objects.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define MEM_PER_THREAD (sizeof(bool))

__global__ void _run_neurongroup_thresholder_codeobject_kernel(
	unsigned int _neurongroup_N,
	unsigned int _num_thread_per_block,
	double _t,
	int32_t* _array_neurongroup__spikespace,
	double* _array_neurongroup_v,
	double* _array_neurongroup_lastspike,
	bool* _array_neurongroup_not_refractory)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	extern __shared__ bool spike_cache[];	//format: id[x] = true <=> neuron[x] has spiked

	int num_threads_per_block = _num_thread_per_block;

	int neuron_id = bid*num_threads_per_block + tid;
	if(neuron_id < 0 || neuron_id >= _neurongroup_N)
	{
		return;
	}

	spike_cache[tid] = false;
	_array_neurongroup__spikespace[neuron_id] = -1;
	
	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		_array_neurongroup__spikespace[_neurongroup_N] = 0;
	}
	__syncthreads();

	double v = _array_neurongroup_v[neuron_id];
	bool not_refractory = _array_neurongroup_not_refractory[neuron_id];
	spike_cache[tid] = (v > 0.001) && (not_refractory);

	//only one thread per block iterates over the cache
	if(tid != 0)
	{
		return;
	}

	int first_neuron_in_block = neuron_id;	//tid = 0, so neuron_id = bid*num_threads_per_block = start of block no. bid
	int num_spikes_in_block = 0;
	for(int i = 0; (i < num_threads_per_block) && (first_neuron_in_block + i < _neurongroup_N); i++)
	{
		if(spike_cache[i])
		{
			//spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
			int spiking_neuron = first_neuron_in_block + i;
			_array_neurongroup__spikespace[first_neuron_in_block + num_spikes_in_block] = spiking_neuron;
			_array_neurongroup_not_refractory[spiking_neuron] = false;
			_array_neurongroup_lastspike[spiking_neuron] = _t;
			num_spikes_in_block++;
		}
	}
	//add number of spikes of all blocks together
	//last element of spikespace holds total number of spikes
	atomicAdd(&_array_neurongroup__spikespace[_neurongroup_N], num_spikes_in_block);
}

void _run_neurongroup_thresholder_codeobject()
{
	using namespace brian;

	const double t = defaultclock.t_();

	unsigned int threads = (neurongroup_N + num_blocks - 1)/num_blocks;	// = ceil(N/num_threads)

	_run_neurongroup_thresholder_codeobject_kernel<<<num_blocks, threads, threads*MEM_PER_THREAD>>>(
		neurongroup_N,
		threads,
		t,
		dev_array_neurongroup__spikespace,
		dev_array_neurongroup_v,
		dev_array_neurongroup_lastspike,
		dev_array_neurongroup_not_refractory);
}

