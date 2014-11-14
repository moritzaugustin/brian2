#include "objects.h"


#include "code_objects/synapses_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

void _run_synapses_pre_push_spikes()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	const int _num_spikespace = 4001;
	///// POINTERS ////////////
        
    int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;


    //// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	synapses_pre.queue->advance();
	synapses_pre.queue->push(_ptr_array_neurongroup__spikespace, _ptr_array_neurongroup__spikespace[_num_spikespace-1]);
	synapses_pre.queue->peek();
}
