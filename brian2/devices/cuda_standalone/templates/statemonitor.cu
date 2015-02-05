{% extends 'common_group.cu' %}

{% block kernel_call %}
{# USES_VARIABLES { t, _clock_t, _indices } #}
	for(int i = 0; i < _num_indices; i++)
	{
		unsigned int _id = {{_indices}}[i];
		int index_last_element;
		{{vector_code|autoindent}}
		{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		{{c_data_type(var.dtype)}}* dev_{{_recorded}} = thrust::raw_pointer_cast(&({{_recorded}}[i][0]));
		{{_recorded}}[i].push_back(0.0);	//push dummy value
		index_last_element = {{_recorded}}[i].size() - 1;
		// {{varname}}
		{% endfor %}

		_run_{{codeobj_name}}_kernel<<<1, 1>>>(
			_id,
			{% for varname, var in _recorded_variables | dictsort %}
			{% set _recorded =  get_array_name(var, access_data=False) %}
			dev_{{_recorded}},
			_to_record_{{varname}},
			{% endfor %}
			index_last_element);
	}
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_kernel(
	int _id,
	{% for varname, var in _recorded_variables | dictsort %}
	{% set _recorded =  get_array_name(var, access_data=False) %}
	{{c_data_type(var.dtype)}}* monitor_{{varname}},
	{{c_data_type(var.dtype)}}* data_{{varname}},
	{% endfor %}
	int _index_last_element
	)
{
	{% for varname, var in _recorded_variables | dictsort %}
	{% set _recorded =  get_array_name(var, access_data=False) %}
	monitor_{{varname}}[_index_last_element] = data_{{varname}}[_id];
	{% endfor %}
}
{% endblock %}
