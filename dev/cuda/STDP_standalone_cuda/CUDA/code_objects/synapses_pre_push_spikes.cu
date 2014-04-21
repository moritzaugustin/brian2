#include "objects.h"


#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

void _run_synapses_pre_push_spikes()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	//const int _num_spikespace = 1001;
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_poissongroup__spikespace = _array_poissongroup__spikespace;

    //// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses

	cudaMemcpy(_array_poissongroup__spikespace, dev_array_poissongroup__spikespace, sizeof(int32_t)*_num__array_poissongroup__spikespace, cudaMemcpyDeviceToHost);

	synapses_pre.queue->advance();
	synapses_pre.queue->push(_ptr_array_poissongroup__spikespace, _ptr_array_poissongroup__spikespace[1000]);
	synapses_pre.queue->peek();
}
