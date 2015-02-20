{% extends 'common_group.cu' %}
{# USES_VARIABLES { rate, t, _spikespace, _clock_t, _clock_dt,
                    _num_source_neurons, _source_start, _source_stop } #}

{% block extra_maincode %}
{{_dynamic_t}}.push_back(_clock_t);
{{_dynamic_rate}}.push_back(0.0);	//push dummy value

double* dev{{_dynamic_rate}} = thrust::raw_pointer_cast(&({{_dynamic_rate}}[0]));
int index_last_element = {{_dynamic_rate}}.size() - 1;
{% endblock %}

{% block kernel_call %}
_run_{{codeobj_name}}_kernel<<<1,1>>>(
	{{owner.source.N}},
	_clock_dt,
	index_last_element,
	dev_array_{{owner.source.name}}__spikespace,
	dev{{_dynamic_rate}});
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int N,
	double _clock_dt,
	int32_t index_last_element,
	int32_t* spikespace,
	double* ratemonitor_rate
	)
{
	using namespace brian;

	unsigned int num_spikes = spikespace[N];
	ratemonitor_rate[index_last_element] = 1.0*num_spikes/_clock_dt/N;
}
{% endblock %}
