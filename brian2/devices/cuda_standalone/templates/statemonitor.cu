{% extends 'common_group.cu' %}

{% block extra_maincode %}
int num_iterations = {{owner.clock.name}}.i_end;
int current_iteration = {{owner.clock.name}}.i;

dev_dynamic_array_{{owner.name}}_t.push_back({{owner.clock.name}}.t_());
for(int i = 0; i < _num__array_{{owner.name}}__indices; i++)
{
	{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		if({{_recorded}}[i].size() != num_iterations)
		{
			{{_recorded}}[i].resize(num_iterations);
			addresses_monitor_{{_recorded}}.push_back(thrust::raw_pointer_cast(&{{_recorded}}[i][0]));
		}
	{% endfor %}
}
{% endblock %}

{% block kernel_call %}
_run_{{codeobj_name}}_kernel<<<1, _num__array_{{owner.name}}__indices>>>(
	_num__array_{{owner.name}}__indices,
	dev_array_{{owner.name}}__indices,
	{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		thrust::raw_pointer_cast(&addresses_monitor_{{_recorded}}[0]),
		%DATA_{{varname}}%,
	{% endfor %}
	current_iteration);
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_kernel(
	int _num_indices,
	int32_t* indices,
	{% for varname, var in _recorded_variables | dictsort %}
		{{c_data_type(var.dtype)}}** monitor_{{varname}},
		{{c_data_type(var.dtype)}}* data_{{varname}},
	{% endfor %}
	int current_iteration
	)
{
	unsigned int tid = threadIdx.x;
	if(tid > _num_indices)
	{
		return;
	}

	int32_t neuron_id = indices[tid];
	{% for varname, var in _recorded_variables | dictsort %}
		{% set _recorded =  get_array_name(var, access_data=False) %}
		monitor_{{varname}}[tid][current_iteration] = data_{{varname}}[neuron_id];
	{% endfor %}
}
{% endblock %}
