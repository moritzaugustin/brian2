#include "objects.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

#define MEMORY_PER_THREAD (2*sizeof(int32_t) + sizeof(unsigned int))

__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	
	synapses_pre.queue->advance(
		tid);
}

__global__ void _run_synapses_pre_push_spikes_push_kernel(
	unsigned int _neurongroup_N,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	int32_t* _array_neurongroup_spikespace)
{
	using namespace brian;

	extern __shared__ char shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = 0; i < _neurongroup_N;)
	{
		int32_t spiking_neuron = _array_neurongroup_spikespace[i];
		if(spiking_neuron != -1)
		{
			__syncthreads();
			i++;
			synapses_pre.queue->push(
				bid,
				tid,
				_num_threads,
				spiking_neuron,
				shared_mem);
		}
		else
		{
			//round to nearest multiple of N/num_blocks = start of next block
			i += _neurongroup_N/_num_blocks - i % (_neurongroup_N/_num_blocks);
		}
	}
}

void _run_synapses_pre_push_spikes()
{
	using namespace brian;
	
	_run_synapses_pre_push_spikes_advance_kernel<<<1, num_blocks>>>();

	unsigned int num_threads = max_shared_mem_size / MEMORY_PER_THREAD;
	num_threads = num_threads > max_threads_per_block? max_threads_per_block : num_threads;	// get min of both
	_run_synapses_pre_push_spikes_push_kernel<<<num_blocks, num_threads, num_threads*MEMORY_PER_THREAD>>>(
		neurongroup_N,
		num_blocks,
		num_threads,
		dev_array_neurongroup__spikespace);

}
