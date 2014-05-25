#include "objects.h"


#include "code_objects/synapses_post_push_spikes.h"
#include <math.h>
#include <stdint.h>
#include "brianlib/common_math.h"

__global__ void _run_synapses_post_push_spikes_advance_kernel()
{
	int tid = blockIdx.x;
	brian::synapses_post.queue->advance(tid);
}

__global__ void _run_synapses_post_push_spikes_push_kernel(int* spikespace)
{
	int tid = blockIdx.x;
	brian::synapses_post.queue->push(tid, spikespace, 1);
}

void _run_synapses_post_push_spikes()
{
	using namespace brian;
	///// CONSTANTS ///////////
	//const int _num_spikespace = 2;
	///// POINTERS ////////////

	int32_t * __restrict__ _ptr_array_neurongroup__spikespace = dev_array_neurongroup__spikespace;

	//// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	_run_synapses_post_push_spikes_advance_kernel<<<num_blocks_sequential, 1>>>();
	_run_synapses_post_push_spikes_push_kernel<<<num_blocks_sequential, 1>>>(_ptr_array_neurongroup__spikespace);
}
