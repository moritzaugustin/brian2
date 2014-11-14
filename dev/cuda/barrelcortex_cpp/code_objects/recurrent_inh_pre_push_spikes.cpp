#include "objects.h"


#include "code_objects/recurrent_inh_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

void _run_recurrent_inh_pre_push_spikes()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	const int _num_spikespace = 3077;
	///// POINTERS ////////////
        
    int32_t * __restrict _ptr_array_layer23__spikespace = _array_layer23__spikespace;


    //// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	recurrent_inh_pre.advance();
	recurrent_inh_pre.push(_ptr_array_layer23__spikespace, _ptr_array_layer23__spikespace[_num_spikespace-1]);
	//recurrent_inh_pre.queue->peek();
}
