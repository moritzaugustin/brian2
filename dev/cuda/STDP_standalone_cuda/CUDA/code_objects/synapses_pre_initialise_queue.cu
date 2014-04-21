#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"
void _run_synapses_pre_initialise_queue() {
	using namespace brian;
	double* real_delays = &(synapses_pre.delay[0]);
	int* sources = &(synapses_pre.sources[0]);
	const unsigned int n_synapses = synapses_pre.sources.size();
	synapses_pre.queue->prepare(real_delays, sources, n_synapses, synapses_pre.dt);
}
