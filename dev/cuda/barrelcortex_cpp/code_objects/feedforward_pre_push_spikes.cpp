#include "objects.h"


#include "code_objects/feedforward_pre_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

void _run_feedforward_pre_push_spikes()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	const int _num_spikespace = 1937;
	///// POINTERS ////////////
        
    int32_t * __restrict _ptr_array_layer4__spikespace = _array_layer4__spikespace;


    //// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	feedforward_pre.advance();
	feedforward_pre.push(_ptr_array_layer4__spikespace, _ptr_array_layer4__spikespace[_num_spikespace-1]);
	//feedforward_pre.queue->peek();
}
