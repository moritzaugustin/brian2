#include "objects.h"
#include "code_objects/synapses_pre_initialise_queue.h"

#define neuron_N 4000
#define THREADS 1024
#define BLOCKS (num_blocks)

__global__ void _run_synapses_pre_initialise_queue_kernel(
	unsigned int _num_blocks,
	double _dt,
	unsigned int _syn_N,
	unsigned int max_delay,
	unsigned int* size_by_pre_id,
	int32_t** synapses_by_pre_id,
	int32_t** post_neuron_by_pre_id,
	unsigned int** delay_by_pre_id)
{
	using namespace brian;

	int tid = threadIdx.x;

	unsigned int num_blocks = _num_blocks;
	double dt = _dt;
	unsigned int syn_N = _syn_N;

	synapses_pre.queue->prepare(
		tid,
		THREADS,
		num_blocks,
		dt,
		neuron_N,
		syn_N,
		max_delay,
		size_by_pre_id,
		synapses_by_pre_id,
		post_neuron_by_pre_id,
		delay_by_pre_id);
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_synapses_pre_initialise_queue()
{
	using namespace brian;

	double dt = defaultclock.dt_();
	unsigned syn_N = synapses._N();
	double* dev_array_synapses_pre_delay = thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay[0]);
	int32_t* dev_synapses_by_pre_neuron = thrust::raw_pointer_cast(&synapses_by_pre_neuron[0]);
	int32_t* dev_dynamic_array_synapses__synaptic_pre = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]);
	int32_t* dev_dynamic_array_synapses__synaptic_post = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]);

	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[BLOCKS*neuron_N];
	thrust::host_vector<int32_t>* h_post_neuron_by_pre_id = new thrust::host_vector<int32_t>[BLOCKS*neuron_N];
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[BLOCKS*neuron_N];

	unsigned int max_delay = 0;
	for(int syn_id = 0; syn_id < syn_N; syn_id++)
	{
		int32_t pre_neuron_id = _dynamic_array_synapses__synaptic_pre[syn_id];
		int32_t post_neuron_id = _dynamic_array_synapses__synaptic_post[syn_id];
		unsigned int delay = (int)(_dynamic_array_synapses_pre_delay[syn_id] / dt + 0.5);
		if(delay > max_delay)
			max_delay = delay;
		unsigned int right_queue = (post_neuron_id*BLOCKS)/neuron_N;
		unsigned int right_offset = OFFSET(right_queue, pre_neuron_id, neuron_N);
		h_synapses_by_pre_id[right_offset].push_back(syn_id);
		h_post_neuron_by_pre_id[right_offset].push_back(post_neuron_id);
		h_delay_by_pre_id[right_offset].push_back(delay);
	}

	unsigned int* temp_size_by_pre_id = new unsigned int[BLOCKS*neuron_N];
	int32_t** temp_synapses_by_pre_id = new int32_t*[BLOCKS*neuron_N];
	int32_t** temp_post_neuron_by_pre_id = new int32_t*[BLOCKS*neuron_N];
	unsigned int** temp_delay_by_pre_id = new unsigned int*[BLOCKS*neuron_N];

	for(int i = 0; i < BLOCKS*neuron_N; i++)
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		temp_size_by_pre_id[i] = num_elements;
		cudaMalloc((void**)&temp_synapses_by_pre_id[i], sizeof(int32_t)*num_elements);
		cudaMalloc((void**)&temp_post_neuron_by_pre_id[i], sizeof(int32_t)*num_elements);
		cudaMalloc((void**)&temp_delay_by_pre_id[i], sizeof(unsigned int)*num_elements);
		cudaMemcpy(temp_synapses_by_pre_id[i],
			thrust::raw_pointer_cast(&(h_synapses_by_pre_id[i][0])),
			sizeof(int32_t)*num_elements,
			cudaMemcpyHostToDevice);
		cudaMemcpy(temp_post_neuron_by_pre_id[i],
			thrust::raw_pointer_cast(&(h_post_neuron_by_pre_id[i][0])),
			sizeof(int32_t)*num_elements,
			cudaMemcpyHostToDevice);
		cudaMemcpy(temp_delay_by_pre_id[i],
			thrust::raw_pointer_cast(&(h_delay_by_pre_id[i][0])),
			sizeof(int32_t)*num_elements,
			cudaMemcpyHostToDevice);
	}

	//copy temp to device
	cudaMemcpy(size_by_pre, temp_size_by_pre_id, sizeof(unsigned int)*neuron_N*BLOCKS, cudaMemcpyHostToDevice);
	cudaMemcpy(synapses_id_by_pre, temp_synapses_by_pre_id, sizeof(int32_t)*neuron_N*BLOCKS, cudaMemcpyHostToDevice);
	cudaMemcpy(post_neuron_by_pre, temp_post_neuron_by_pre_id, sizeof(int32_t)*neuron_N*BLOCKS, cudaMemcpyHostToDevice);
	cudaMemcpy(delay_by_pre, temp_delay_by_pre_id, sizeof(unsigned int)*neuron_N*BLOCKS, cudaMemcpyHostToDevice);

	_run_synapses_pre_initialise_queue_kernel<<<1, max_delay>>>(
		BLOCKS,
		dt,
		syn_N,
		max_delay,
		size_by_pre,
		synapses_id_by_pre,
		post_neuron_by_pre,
		delay_by_pre);
}

