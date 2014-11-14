#include "objects.h"
#include "code_objects/recurrent_exc_pre_initialise_queue.h"
void _run_recurrent_exc_pre_initialise_queue() {
	using namespace brian;
    double* real_delays = &(recurrent_exc_pre.delay[0]);
    int32_t* sources = &(recurrent_exc_pre.sources[0]);
    const unsigned int n_delays = recurrent_exc_pre.delay.size();
    const unsigned int n_synapses = recurrent_exc_pre.sources.size();
    recurrent_exc_pre.prepare(real_delays, n_delays, sources, n_synapses,
                        recurrent_exc_pre.dt);
}
