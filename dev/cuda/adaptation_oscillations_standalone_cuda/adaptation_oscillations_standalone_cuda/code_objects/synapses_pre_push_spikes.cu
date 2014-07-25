#include "objects.h"


#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

#define N 4000

#define ceil(N, num) ((N + num-1)/num)

__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
	using namespace brian;

	int bid = blockIdx.x;

	synapses_pre.queue->advance(bid);
}

__global__ void _run_synapses_pre_push_spikes_push_kernel(int pre_id, int num_connected, int32_t* spikespace)
{
	using namespace brian;

	extern __shared__ int data[];

	int bid = threadIdx.x;

	synapses_pre.queue->push(pre_id, bid, data, spikespace, num_connected);
}

void _run_synapses_pre_push_spikes()
{
	using namespace brian;

	_run_synapses_pre_push_spikes_advance_kernel<<<num_blocks_sequential, 1>>>();

	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
	{
		int pre_id = _array_neurongroup__spikespace[i];
		if(pre_id != -1)
		{
			int num_connected_synapses = _array_synapses_N_outgoing[pre_id];
			_run_synapses_pre_push_spikes_push_kernel<<<1, num_connected_synapses, 3*sizeof(int32_t)*num_connected_synapses>>>(pre_id, num_connected_synapses, dev_array_neurongroup__spikespace);
		}
		else
		{
			i += N/num_blocks_sequential - i % (N/num_blocks_sequential);
		}
	}
}

