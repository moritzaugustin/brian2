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

__global__ void _run_synapses_pre_push_spikes_push_kernel(int stride, int32_t* spikespace)
{
	using namespace brian;

	int bid = blockIdx.x;

	synapses_pre.queue->push(bid, spikespace, stride, N);
}

void _run_synapses_pre_push_spikes()
{
	using namespace brian;

	int32_t * _ptr_array_neurongroup__spikespace = dev_array_neurongroup__spikespace;

	_run_synapses_pre_push_spikes_advance_kernel<<<num_blocks_sequential, 1>>>();
	_run_synapses_pre_push_spikes_push_kernel<<<num_blocks_sequential, 1>>>(ceil(N, num_blocks_sequential), _ptr_array_neurongroup__spikespace);
}

