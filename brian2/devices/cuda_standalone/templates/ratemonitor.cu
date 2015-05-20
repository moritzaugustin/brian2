{% extends 'common_group.cu' %}
{# USES_VARIABLES { rate, t, _spikespace, _clock_t, _clock_dt,
                    _num_source_neurons, _source_start, _source_stop } #}

{% block extra_maincode %}
int current_iteration = {{owner.clock.name}}.i;
static unsigned int start_offset = current_iteration;
static bool first_run = true;
if(first_run)
{
	int num_iterations = {{owner.clock.name}}.i_end;
	unsigned int size_till_now = dev{{_dynamic_t}}.size();
	dev{{_dynamic_t}}.resize(num_iterations + size_till_now - start_offset);
	dev{{_dynamic_rate}}.resize(num_iterations + size_till_now - start_offset);
	first_run = false;
}
{% endblock %}

{% block kernel_call %}
_run_{{codeobj_name}}_kernel<<<1,1>>>(
	{{owner.source.N}},
	_clock_t,
	_clock_dt,
	current_iteration - start_offset,
	dev_array_{{owner.source.name}}__spikespace,
	thrust::raw_pointer_cast(&(dev{{_dynamic_rate}}[0])),
	thrust::raw_pointer_cast(&(dev{{_dynamic_t}}[0])));
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int N,
	double _clock_t,
	double _clock_dt,
	int32_t current_iteration,
	int32_t* spikespace,
	double* ratemonitor_rate,
	double* ratemonitor_t
	)
{
	using namespace brian;

	unsigned int num_spikes = spikespace[N];
	ratemonitor_rate[current_iteration] = 1.0*num_spikes/_clock_dt/N;
	ratemonitor_t[current_iteration] = _clock_t;
}
{% endblock %}