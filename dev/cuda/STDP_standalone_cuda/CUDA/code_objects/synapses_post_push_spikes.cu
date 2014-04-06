#include "objects.h"


#include "code_objects/synapses_post_push_spikes.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

void _run_synapses_post_push_spikes()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	const int _num_spikespace = 2;
	///// POINTERS ////////////

	int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;

	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);

    //// MAIN CODE ////////////
	// we do advance at the beginning rather than at the end because it saves us making
	// a copy of the current spiking synapses
	synapses_post.queue->advance();
	synapses_post.queue->push(_ptr_array_neurongroup__spikespace, _ptr_array_neurongroup__spikespace[1]);
	synapses_post.queue->peek();

	cudaMemcpy(dev_array_neurongroup__spikespace, _array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);
}
