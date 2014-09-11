#include "objects.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

#define neuron_N 4000
#define MEMORY_PER_THREAD (3*sizeof(int32_t))

__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	
	synapses_pre.queue->advance(
		tid);
}

__global__ void _run_synapses_pre_push_spikes_push_kernel(unsigned int _pre_id, unsigned int _num_threads)
{
	using namespace brian;

	extern __shared__ int shared_mem[];
	int tid = threadIdx.x;
	
	synapses_pre.queue->push(
		tid,
		_num_threads,
		_pre_id,
		shared_mem);
}

void _run_synapses_pre_push_spikes()
{
	using namespace brian;

	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);

	_run_synapses_pre_push_spikes_advance_kernel<<<1, num_blocks_sequential>>>();

    // please briefly comment the logic (and underlying data structure)
	for(int i = 0; i < neuron_N;)
	{
		int32_t spiking_neuron = _array_neurongroup__spikespace[i];
		if(spiking_neuron != -1)
		{
			unsigned int num_connected = _array_synapses_N_outgoing[spiking_neuron];
			_run_synapses_pre_push_spikes_push_kernel<<<1, num_connected, num_connected*MEMORY_PER_THREAD>>>(
				spiking_neuron,
				num_connected);
			i++;
		}
		else
		{
			//round to nearest multiple of N/num_blocks
			i += neuron_N/num_blocks_sequential - i % (neuron_N/num_blocks_sequential);
		}
	}
}
