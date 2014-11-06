#include "objects.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

#define neuron_N 4000
#define MEMORY_PER_THREAD (3*sizeof(int32_t))
#define BLOCKS (num_blocks)

__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	
	synapses_pre.queue->advance(
		tid);
}

__global__ void _run_synapses_pre_push_spikes_push_kernel(
	unsigned int _pre_id)
{
	using namespace brian;

	extern __shared__ int shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	
	synapses_pre.queue->push(
		bid,
		tid,
		_pre_id,
		shared_mem);
}

void _run_synapses_pre_push_spikes()
{
	using namespace brian;
	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);

	_run_synapses_pre_push_spikes_advance_kernel<<<1, BLOCKS>>>();

	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = 0; i < neuron_N;)
	{
		int32_t spiking_neuron = _array_neurongroup__spikespace[i];
		if(spiking_neuron != -1)
		{
			unsigned int num_connected_synapses = _array_synapses_N_outgoing[spiking_neuron];
			//TODO: case num_connected_synapses > THREADS_PER_BLOCK = 1024
			_run_synapses_pre_push_spikes_push_kernel<<<BLOCKS, num_connected_synapses, num_connected_synapses*MEMORY_PER_THREAD>>>(
				spiking_neuron);
			i++;
		}
		else
		{
			//round to nearest multiple of N/num_blocks = start of next block
			i += neuron_N/BLOCKS - i % (neuron_N/BLOCKS);
		}
	}
}
