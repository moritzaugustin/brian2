#include "objects.h"
#include "code_objects/synapses_post_initialise_queue.h"
void _run_synapses_post_initialise_queue() {
	using namespace brian;
    double* real_delays = &(synapses_post.delay[0]);
    int* sources = &(synapses_post.sources[0]);
    const unsigned int n_synapses = synapses_post.sources.size();
    synapses_post.queue->prepare(real_delays, sources, n_synapses,
                               synapses_post.dt);
}
