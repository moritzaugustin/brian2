{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
{% set pathobj = owner.name %}

namespace {
	int num_blocks(int objects){
		return ceil(objects / (double)brian::max_threads_per_block);
	}
	int num_threads(int objects){
		return brian::max_threads_per_block;
	}
}

__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int _target_N,
	unsigned int _num_blocks,
	unsigned int _num_threads_per_block,
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

	{{pathobj}}.queue->prepare(
		tid,
		_num_threads_per_block,
		_num_blocks,
		_dt,
		_target_N,
		_syn_N,
		max_delay,
		size_by_pre_id,
		synapses_by_pre_id,
		post_neuron_by_pre_id,
		delay_by_pre_id);
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_{{pathobj}}_initialise_queue()
{
	using namespace brian;

	double dt = defaultclock.dt_();
	unsigned syn_N = {{pathobj}}._N();
	unsigned source_N = {{pathobj}}.Nsource;
	double* dev_array_synapses_pre_delay = {{pathobj}}.dev_delay;
	int32_t* dev_synapses_by_pre_neuron = thrust::raw_pointer_cast(&synapses_by_pre_neuron[0]);
	int32_t* dev_dynamic_array_synapses__synaptic_pre = {{pathobj}}.dev_sources;
	int32_t* dev_dynamic_array_synapses__synaptic_post = {{pathobj}}.dev_targets;

	//Create temporary host vectors
	thrust::host_vector<int32_t> h_synapses__synaptic_pre = {{pathobj}}.dev_sources;
	thrust::host_vector<int32_t> h_synapses__synaptic_post = {{pathobj}}.dev_targets;
	thrust::host_vector<double> h_synapses_pre_delay = {{pathobj}}.dev_delays;
	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[num_cuda_processors*source_N];
	thrust::host_vector<int32_t>* h_post_neuron_by_pre_id = new thrust::host_vector<int32_t>[num_cuda_processors*source_N];
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_cuda_processors*source_N];

	//fill vectors with pre_neuron, post_neuron, delay data
	unsigned int max_delay = 0;
	for(int syn_id = 0; syn_id < syn_N; syn_id++)
	{
		int32_t pre_neuron_id = h_synapses__synaptic_pre[syn_id];
		int32_t post_neuron_id = h_synapses__synaptic_post[syn_id];
		unsigned int delay = (int)(h_synapses_pre_delay[syn_id] / dt + 0.5);
		if(delay > max_delay)
		{
			max_delay = delay;
		}
		unsigned int right_queue = (post_neuron_id*num_cuda_processors)/source_N;
		unsigned int right_offset = OFFSET(right_queue, pre_neuron_id, source_N);
		h_synapses_by_pre_id[right_offset].push_back(syn_id);
		h_post_neuron_by_pre_id[right_offset].push_back(post_neuron_id);
		h_delay_by_pre_id[right_offset].push_back(delay);
	}

	//create array for device pointers
	unsigned int* temp_size_by_pre_id = new unsigned int[num_cuda_processors*source_N];
	int32_t** temp_synapses_by_pre_id = new int32_t*[num_cuda_processors*source_N];
	int32_t** temp_post_neuron_by_pre_id = new int32_t*[num_cuda_processors*source_N];
	unsigned int** temp_delay_by_pre_id = new unsigned int*[num_cuda_processors*source_N];
	//fill temp arrays with device pointers
	for(int i = 0; i < num_cuda_processors*source_N; i++)
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		temp_size_by_pre_id[i] = num_elements;
		if(num_elements > 0)
		{
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
				sizeof(unsigned int)*num_elements,
				cudaMemcpyHostToDevice);
		}
	}

	//copy temp arrays to device
	cudaMemcpy({{pathobj}}_size_by_pre, temp_size_by_pre_id, sizeof(unsigned int)*num_cuda_processors*source_N, cudaMemcpyHostToDevice);
	cudaMemcpy({{pathobj}}_synapses_id_by_pre, temp_synapses_by_pre_id, sizeof(int32_t*)*num_cuda_processors*source_N, cudaMemcpyHostToDevice);
	cudaMemcpy({{pathobj}}_post_neuron_by_pre, temp_post_neuron_by_pre_id, sizeof(int32_t*)*num_cuda_processors*source_N, cudaMemcpyHostToDevice);
	cudaMemcpy({{pathobj}}_delay_by_pre, temp_delay_by_pre_id, sizeof(unsigned int*)*num_cuda_processors*source_N, cudaMemcpyHostToDevice);

	_run_{{codeobj_name}}_kernel<<<1, max_delay>>>(
		source_N,
		num_cuda_processors,
		max_threads_per_block,
		dt,
		syn_N,
		max_delay,
		dev_size_by_pre,
		dev_synapses_id_by_pre,
		dev_post_neuron_by_pre,
		dev_delay_by_pre);

	//delete temp arrays
	delete [] h_synapses_by_pre_id;
	delete [] h_post_neuron_by_pre_id;
	delete [] h_delay_by_pre_id;
	delete [] temp_size_by_pre_id;
	delete [] temp_synapses_by_pre_id;
	delete [] temp_post_neuron_by_pre_id;
	delete [] temp_delay_by_pre_id;
}

{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
