#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"

#define neuron_N 4000
#define THREADS 1024
#define BLOCKS (num_blocks)

__global__ void _run_synapses_pre_initialise_queue_kernel(
	unsigned int _num_blocks,
	double _dt,
	unsigned int _syn_N,
	double* _synapses_delay)
{
	using namespace brian;

	int tid = threadIdx.x;

	unsigned int num_blocks = _num_blocks;
	double dt = _dt;
	unsigned int syn_N = _syn_N;
	double* synapses_delay = _synapses_delay;

	synapses_pre.queue->prepare(
		tid,
		THREADS,
		num_blocks,
		dt,
		neuron_N,
		syn_N,
		synapses_delay);
}

__global__ void _run_synapses_pre_initialise_queue_kernel_1(
	unsigned int _syn_N,
	int32_t* _synapses_by_pre_neuron,
	int32_t* _dynamic_array_synapses__synaptic_pre,
	int32_t* _dynamic_array_synapses__synaptic_post)
{
	using namespace brian;

	int bid = blockIdx.x;

	unsigned int syn_N = _syn_N;
	int32_t* synapses_by_pre_neuron = _synapses_by_pre_neuron;
	int32_t* dynamic_array_synapses__synaptic_pre = _dynamic_array_synapses__synaptic_pre;
	int32_t* dynamic_array_synapses__synaptic_post = _dynamic_array_synapses__synaptic_post;

	synapses_pre.queue->prepare_connect_matrix(
		bid,
		syn_N,
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

	//we have two init functions, because their launch grid is different
	_run_synapses_pre_initialise_queue_kernel<<<1, THREADS>>>(
		num_blocks,
		dt,
		syn_N,
		dev_array_synapses_pre_delay);

	_run_synapses_pre_initialise_queue_kernel_1<<<BLOCKS, 1>>>(
		syn_N,
		dev_synapses_by_pre_neuron,
		dev_dynamic_array_synapses__synaptic_pre,
		dev_dynamic_array_synapses__synaptic_post);
}

