{% extends 'common_synapses.cu' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block num_thread_check %}
{% endblock %}

{% block maincode %}
	// This is only needed for the _debugmsg function below	
	{# USES_VARIABLES { _synaptic_pre } #}	
	
	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;
	
	{{pathway.name}}.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	int size = post_neuron_queue[bid].size();
	for(int j = tid; j < size; j++)
	{
		int32_t _pre_idx = pre_neuron_queue[bid].at(j);
		int32_t _post_idx = post_neuron_queue[bid].at(j);
		int32_t _idx = synapses_queue[bid].at(j);

		{{vector_code|autoindent}}
	}
{% endblock %}

{% block kernel_call %}
kernel_{{codeobj_name}}<<<num_parallel_blocks,1>>>(
		num_threads(N),
		%HOST_PARAMETERS%
	);
{% endblock %}

{% block extra_maincode %}
unsigned int N = {{owner.name}}._N();
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
