#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"

#define neuron_N 4000
#define THREADS 1024

__global__ void _run_synapses_pre_initialise_queue_kernel(
	unsigned int _num_blocks_sequential,
	double _dt,
	unsigned int _syn_N,
	double* _synapses_delay,
	int32_t* _synapses_by_pre_neuron,
	int32_t* _dynamic_array_synapses__synaptic_pre,
	int32_t* _dynamic_array_synapses__synaptic_post)
{
	using namespace brian;

	int tid = threadIdx.x;

	unsigned int num_blocks_sequential = _num_blocks_sequential;
	double dt = _dt;
	unsigned int syn_N = _syn_N;
	double* synapses_delay = _synapses_delay;
	int32_t* synapses_by_pre_neuron = _synapses_by_pre_neuron;
	int32_t* dynamic_array_synapses__synaptic_pre = _dynamic_array_synapses__synaptic_pre;
	int32_t* dynamic_array_synapses__synaptic_post = _dynamic_array_synapses__synaptic_post;

	synapses_pre.queue->prepare(
		tid,
		THREADS,
		num_blocks_sequential,
		dt,
		neuron_N,
		syn_N,
		synapses_delay,
		synapses_by_pre_neuron,
		dynamic_array_synapses__synaptic_pre,
		dynamic_array_synapses__synaptic_post);

}

void _run_synapses_pre_initialise_queue()
{
	using namespace brian;

	double dt = defaultclock.dt_();
	unsigned syn_N = synapses._N();
	double* dev_array_synapses_pre_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]);
	int32_t* dev_synapses_by_pre_neuron = thrust::raw_pointer_cast(&synapses_by_pre_neuron[0]);
	int32_t* dev_dynamic_array_synapses__synaptic_pre = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]);
	int32_t* dev_dynamic_array_synapses__synaptic_post = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]);

	_run_synapses_pre_initialise_queue_kernel<<<1, THREADS>>>(
		num_blocks_sequential,
		dt,
		syn_N,
		dev_array_synapses_pre_delay,
		dev_synapses_by_pre_neuron,
		dev_dynamic_array_synapses__synaptic_pre,
		dev_dynamic_array_synapses__synaptic_post);
}

