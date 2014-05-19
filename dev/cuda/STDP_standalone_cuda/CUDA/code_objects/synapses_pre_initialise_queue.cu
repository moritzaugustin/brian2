#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"

__global__ void _run_synapses_pre_initialise_queue_kernel(double* par_real_delays, int* par_sources, int* par_targets, int n_synapses)
{	
	brian::synapses_pre.queue->prepare(par_real_delays, par_sources, par_targets, n_synapses, brian::synapses_pre.dt);
}

void _run_synapses_pre_initialise_queue()
{
	using namespace brian;
	_run_synapses_pre_initialise_queue_kernel<<<1, 1>>>(
		thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]),
		_dynamic_array_synapses__synaptic_pre.size());
}
