#include "objects.h"
#include "code_objects/synapses_post_initialise_queue.h"

__global__ void _run_synapses_post_initialise_queue_kernel(double* par_real_delays, int32_t* par_sources, int32_t* par_targets, int n_synapses)
{	
	brian::synapses_post.queue->prepare(par_real_delays, par_sources, par_targets, n_synapses, brian::synapses_post.dt);
}

void _run_synapses_post_initialise_queue()
{
	using namespace brian;
	_run_synapses_post_initialise_queue_kernel<<<1, 1>>>(
		thrust::raw_pointer_cast(&_dynamic_array_synapses_post_delay[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]),
		_dynamic_array_synapses__synaptic_post.size());
}