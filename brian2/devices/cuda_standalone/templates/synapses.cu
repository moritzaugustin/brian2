{% extends 'common_synapses.cu' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block kernel %}
{% if no_delay_mode == False %}
__global__ void kernel_{{codeobj_name}}(
	unsigned int bid_offset,
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
{% else %}
__global__ void kernel_{{codeobj_name}}(
	unsigned int* size_by_post,
	int32_t** syn_by_post,
	int32_t* spikespace,
	unsigned int THREADS_PER_BLOCK,
	unsigned int NUM_BLOCKS,
	%DEVICE_PARAMETERS%
	)
{
{% endif %}
	{# USES_VARIABLES { N, _synaptic_pre } #}
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x + bid_offset;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%
	{% block additional_variables %}
	{% endblock %}

	cudaVector<int32_t>* synapses_queue;
	
	{{pathway.name}}.queue->peek(
		&synapses_queue);

	int size = synapses_queue[bid].size();
	
	{{scalar_code|autoindent}}
	
{% if no_delay_mode == False %}
	for(int j = tid; j < size; j+=THREADS_PER_BLOCK)
	{
		int32_t _idx = synapses_queue[bid].at(j);

		{{vector_code|autoindent}}
	}
{% else %}
	for(int j = bid; j < N; j += NUM_BLOCKS)
	{
		int32_t syn_idx = synapses_queue[bid].at(j);
		int32_t spiking_neuron = synapses_queue[bid].at(j);
		unsigned int size = size_by_post[spiking_neuron];
		for(int i = tid; i < size; i += THREADS_PER_BLOCK)
		{
			int32_t _idx = syn_by_post[j][i];
			_vectorisation_idx = _idx;
			{{vector_code|autoindent}}
		}
	}
{% endif %}
}

{% endblock %}

{% block kernel_call %}
{% if no_delay_mode == False %}
	{% if serializing_mode == "syn" %}
	kernel_{{codeobj_name}}<<<num_parallel_blocks,max_threads_per_block>>>(
		0,
		max_threads_per_block,
		%HOST_PARAMETERS%
	);
	{% endif %}
	{% if serializing_mode == "post" %}
	kernel_{{codeobj_name}}<<<num_parallel_blocks,1>>>(
		0,
		1,
		%HOST_PARAMETERS%
	);
	{% endif %}
	{% if serializing_mode == "pre" %}
	for(int i = 0; i < num_parallel_blocks; i++)
	{
		kernel_{{codeobj_name}}<<<1,1>>>(
			i,
			1,
			%HOST_PARAMETERS%
		);
	}
	{% endif %}
{% else %}
	//NO DELAY MODE, process synaptic events immediately
	{% set _spikespace = get_array_name(owner.variables['_spikespace'], access_data=False) %}
	{% if serializing_mode == "syn" %}
	kernel_{{codeobj_name}}<<<num_parallel_blocks, max_threads_per_block>>>(
		{{owner.name}}_size_by_pre,
		{{owner.name}}_synapses_id_by_pre,
		dev{{_spikespace}},
		max_threads_per_block,
		num_parallel_blocks,
		%HOST_PARAMETERS%
	);
	{% endif %}
	{% if serializing_mode == "post" %}
	kernel_{{codeobj_name}}<<<1, max_threads_per_block>>>(
		{{owner.name}}_size_by_pre,
		{{owner.name}}_synapses_id_by_pre,
		dev{{_spikespace}},
		1,
		num_parallel_blocks,
		%HOST_PARAMETERS%
	);
	{% endif %}
	{% if serializing_mode == "pre" %}
	kernel_{{codeobj_name}}<<<1,1>>>(
		{{owner.name}}_size_by_pre,
		{{owner.name}}_synapses_id_by_pre,
		dev{{_spikespace}},
		1,
		1,
		%HOST_PARAMETERS%
	);
	{% endif %}
{% endif %}
{% endblock %}

{% block extra_maincode %}
{% endblock %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	std::cout << "Number of synapses: " << dev{{_dynamic__synaptic_pre}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
