#include "objects.h"


#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

__global__ void _run_synapses_pre_push_spikes_advance_kernel()
{
	int tid = threadIdx.x;
	brian::synapses_pre.queue->advance(tid);
}

__global__ void _run_synapses_pre_push_spikes_push_kernel(int* spikespace)
{
	int tid = threadIdx.x;
	brian::synapses_pre.queue->push(tid, spikespace, spikespace[1000]);
}

void _run_synapses_pre_push_spikes()
{
	using namespace brian;
	///// CONSTANTS ///////////
	//const int _num_spikespace = 2;
	///// POINTERS ////////////

	int32_t * __restrict__ _ptr_array_poissongroup__spikespace = _array_poissongroup__spikespace;

	//// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	//TODO <<<num, 1>>> auch bei allen anderen
	_run_synapses_pre_push_spikes_advance_kernel<<<num_multiprocessors, 1>>>();
	_run_synapses_pre_push_spikes_push_kernel<<<num_multiprocessors, 1>>>(_ptr_array_poissongroup__spikespace);
}
