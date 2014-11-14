#include "objects.h"
#include "code_objects/feedforward_pre_initialise_queue.h"
void _run_feedforward_pre_initialise_queue() {
	using namespace brian;
    double* real_delays = &(feedforward_pre.delay[0]);
    int32_t* sources = &(feedforward_pre.sources[0]);
    const unsigned int n_delays = feedforward_pre.delay.size();
    const unsigned int n_synapses = feedforward_pre.sources.size();
    feedforward_pre.prepare(real_delays, n_delays, sources, n_synapses,
                        feedforward_pre.dt);
}