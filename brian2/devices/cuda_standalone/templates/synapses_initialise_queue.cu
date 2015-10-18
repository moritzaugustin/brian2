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
	bool new_mode)
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
		{{pathobj}}_size_by_pre,
		{{pathobj}}_synapses_id_by_pre,
		{{pathobj}}_delay_by_pre);
	{{pathobj}}.no_or_const_delay_mode = new_mode;
}

//POS(queue_id, neuron_id, neurons_N)
#define OFFSET(a, b, c)	(a*c + b)

void _run_{{pathobj}}_initialise_queue()
{
	using namespace brian;

	{% if no_or_const_delay_mode %}
	unsigned int save_num_blocks = num_parallel_blocks;
	num_parallel_blocks = 1;
	{% endif %}
	
	//{{owner.source}}
	//{{owner.source.name}}

	double dt = {{owner._clock._name}}.dt_();
	unsigned int syn_N = dev_dynamic_array_{{pathobj}}_delay.size();
	unsigned int source_N = {{owner.source.N}};
	unsigned int target_N = {{owner.target.N}};

	//Create temporary host vectors
	int32_t* h_synapses_synaptic_sources = new int32_t[syn_N];
	int32_t* h_synapses_synaptic_targets = new int32_t[syn_N];
	double* h_synapses_delay = new double[syn_N];

	cudaMemcpy(h_synapses_synaptic_sources, thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_sources.name}}[0]), sizeof(int32_t) * syn_N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_synapses_synaptic_targets, thrust::raw_pointer_cast(&dev_dynamic_array_{{owner.synapses.name}}_{{owner.synapse_targets.name}}[0]), sizeof(int32_t) * syn_N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_synapses_delay, thrust::raw_pointer_cast(&dev_dynamic_array_{{pathobj}}_delay[0]), sizeof(double) * syn_N, cudaMemcpyDeviceToHost);
	thrust::host_vector<int32_t>* h_synapses_by_pre_id = new thrust::host_vector<int32_t>[num_parallel_blocks*source_N];
	thrust::host_vector<unsigned int>* h_delay_by_pre_id = new thrust::host_vector<unsigned int>[num_parallel_blocks*source_N];

	//fill vectors with pre_neuron, post_neuron, delay data
	unsigned int max_delay = 0;
	for(int syn_id = 0; syn_id < syn_N; syn_id++)
	{
		int32_t pre_neuron_id = h_synapses_synaptic_sources[syn_id] - {{owner.source.start}};
		int32_t post_neuron_id = h_synapses_synaptic_targets[syn_id]  - {{owner.target.start}};
		unsigned int delay = (int)(h_synapses_delay[syn_id] / dt + 0.5);
		if(delay > max_delay)
		{
			max_delay = delay;
		}
		unsigned int right_queue = (post_neuron_id*num_parallel_blocks)/target_N;
		unsigned int right_offset = pre_neuron_id * num_parallel_blocks + right_queue;
		h_synapses_by_pre_id[right_offset].push_back(syn_id);
		h_delay_by_pre_id[right_offset].push_back(delay);
	}
	max_delay++;	//we also need a current step

	{% if no_or_const_delay_mode %}
	free(_array_{{owner.name}}__spikespace);
	_array_{{owner.name}}__spikespace = malloc(sizeof(int32_t)*_num_array_{{owner.name}}__spikespace * max_delay); 
	cudaFree(dev_array_{{owner.name}}__spikespace);
	cudaMalloc((void**)&dev_array_{{owner.name}}__spikespace, sizeof(int32_t)*_num__array_{{owner.name}}__spikespace);
	{% endif %}

	//create array for device pointers
	unsigned int* temp_size_by_pre_id = new unsigned int[num_parallel_blocks*source_N];
	int32_t** temp_synapses_by_pre_id = new int32_t*[num_parallel_blocks*source_N];
	unsigned int** temp_delay_by_pre_id = new unsigned int*[num_parallel_blocks*source_N];
	//fill temp arrays with device pointers
	for(int i = 0; i < num_parallel_blocks*source_N; i++)
	{
		int num_elements = h_synapses_by_pre_id[i].size();
		temp_size_by_pre_id[i] = num_elements;
		if(num_elements > 0)
		{
			cudaMalloc((void**)&temp_synapses_by_pre_id[i], sizeof(int32_t)*num_elements);
			cudaMalloc((void**)&temp_delay_by_pre_id[i], sizeof(unsigned int)*num_elements);
			cudaMemcpy(temp_synapses_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_synapses_by_pre_id[i][0])),
				sizeof(int32_t)*num_elements,
				cudaMemcpyHostToDevice);
			cudaMemcpy(temp_delay_by_pre_id[i],
				thrust::raw_pointer_cast(&(h_delay_by_pre_id[i][0])),
				sizeof(unsigned int)*num_elements,
				cudaMemcpyHostToDevice);
		}
	}

	//copy temp arrays to device
	unsigned int* temp;
	cudaMalloc((void**)&temp, sizeof(unsigned int)*num_parallel_blocks*source_N);
	cudaMemcpy(temp, temp_size_by_pre_id, sizeof(unsigned int)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_size_by_pre, &temp, sizeof(unsigned int*));
	int32_t* temp2;
	cudaMalloc((void**)&temp2, sizeof(int32_t*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp2, temp_synapses_by_pre_id, sizeof(int32_t*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_synapses_id_by_pre, &temp2, sizeof(int32_t**));
	unsigned int* temp3;
	cudaMalloc((void**)&temp3, sizeof(unsigned int*)*num_parallel_blocks*source_N);
	cudaMemcpy(temp3, temp_delay_by_pre_id, sizeof(int32_t*)*num_parallel_blocks*source_N, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol({{pathobj}}_delay_by_pre, &temp3, sizeof(unsigned int**));
	
	unsigned int num_threads = max_delay;
	if(num_threads >= max_threads_per_block)
	{
		num_threads = max_threads_per_block;
	}
	_run_{{codeobj_name}}_kernel<<<1, num_threads>>>(
		source_N,
		num_parallel_blocks,
		max_threads_per_block,
		dt,
		syn_N,
		max_delay,
{% if no_or_const_delay_mode %}
		true
{% else %}
		false
{% endif %}
	);

	//delete temp arrays
	delete [] h_synapses_synaptic_sources;
	delete [] h_synapses_synaptic_targets;
	delete [] h_synapses_delay;
	delete [] h_synapses_by_pre_id;
	delete [] h_delay_by_pre_id;
	delete [] temp_size_by_pre_id;
	delete [] temp_synapses_by_pre_id;
	delete [] temp_delay_by_pre_id;

	{% if no_or_const_delay_mode %}
	num_parallel_blocks = save_num_blocks;
	{% endif %}
}

{% endmacro %}

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
