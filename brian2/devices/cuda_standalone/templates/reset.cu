{% extends 'common_group.cu' %}
{% block maincode %}
	{# USES_VARIABLES { _spikespace, N } #}

	const int32_t *_spikes = {{_spikespace}};
	const int32_t _num_spikes = {{_spikespace}}[N];

	//// MAIN CODE ////////////	
	// scalar code
	const int _vectorisation_idx = -1;
	{{scalar_code|autoindent}}
    
	const int _idx = _spikes[_idx];
	const int _vectorisation_idx = _idx;
	if(_idx != -1)
	{
		{{vector_code|autoindent}}
	}
{% endblock %}
