{% extends 'common_synapses.cu' %}

{% set _non_synaptic = [] %}
{% for var in variables %}
    {% if variable_indices[var] != '_idx' %}
        {# This is a trick to get around the scoping problem #}
        {% if _non_synaptic.append(1) %}{% endif %}
    {% endif %}
{% endfor %}

{% block maincode %}
	// This is only needed for the _debugmsg function below	
	{# USES_VARIABLES { _synaptic_pre } #}	
	
	cudaVector<int32_t>* pre_neuron_queue;
	cudaVector<int32_t>* synapses_queue;
	cudaVector<int32_t>* post_neuron_queue;

	// scalar code
	const int _vectorisation_idx = -1;
	
	{{pathway.name}}.queue->peek(
		&synapses_queue,
		&pre_neuron_queue,
		&post_neuron_queue);

	int size = post_neuron_queue[bid].size();
	//outer loop, since most likely not all spikes fit into our shared memory
	for(int j = tid; j < size; j += THREADS_PER_BLOCK)
	{
		int32_t _idx = post_neuron_queue[bid].getDataByIndex(j);
		int32_t _spiking_synapse_idx = synapses_queue[bid].getDataByIndex(j);

		{{vector_code|autoindent}}
	}
{% endblock %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	std::cout << "Number of synapses: " << {{_dynamic__synaptic_pre}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
