#include "objects.h"
#include "code_objects/feedforward_post_initialise_queue.h"
void _run_feedforward_post_initialise_queue() {
	using namespace brian;
    double* real_delays = &(feedforward_post.delay[0]);
    int32_t* sources = &(feedforward_post.sources[0]);
    const unsigned int n_delays = feedforward_post.delay.size();
    const unsigned int n_synapses = feedforward_post.sources.size();
    feedforward_post.prepare(real_delays, n_delays, sources, n_synapses,
                        feedforward_post.dt);
}
