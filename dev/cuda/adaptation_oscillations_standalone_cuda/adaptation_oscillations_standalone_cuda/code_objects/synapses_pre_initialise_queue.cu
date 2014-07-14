#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"

#define N 4000
#define ceil(M, num) ((M + num-1)/num)

__global__ void _run_synapses_pre_initialise_queue_kernel(double* par_real_delays, int32_t* par_sources, int32_t* par_targets, int32_t* par_pos, int n_synapses)
{
	using namespace brian;

	synapses_pre.queue->prepare(par_real_delays, par_sources, par_targets, par_pos, n_synapses, synapses_pre.dt);
}

void _run_synapses_pre_initialise_queue()
{
	using namespace brian;

	int syn_N = _dynamic_array_synapses__synaptic_pre.size();

	_run_synapses_pre_initialise_queue_kernel<<<1, 1>>>(
		thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]),
		thrust::raw_pointer_cast(&_dynamic_array_synapses__pos[0]),
		syn_N);
}
