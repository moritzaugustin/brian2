{% extends 'common_group.cu' %}
{# USES_VARIABLES { t, i, _clock_t, _spikespace, _count,
                    _source_start, _source_stop} #}

{% block extra_maincode %}
unsigned int start_spikes = {{_dynamic_i}}.size();
int32_t num_spikes;
cudaMemcpy(&num_spikes, &dev_array_{{owner.source.name}}__spikespace[{{owner.source.N}}], sizeof(int32_t), cudaMemcpyDeviceToHost);

for(int i = 0; i < num_spikes; i++)
{
	{{_dynamic_i}}.push_back(0);	//push dummy value
	{{_dynamic_t}}.push_back(t);
}
{% endblock %}

{% block kernel_call %}
_run_{{codeobj_name}}_kernel<<<1, 1>>>(
		{{owner.source.N}},
		num_blocks({{owner.source.N}}),
		start_spikes,
		thrust::raw_pointer_cast(&({{_dynamic_i}}[0])),
		dev_array_{{owner.source.name}}__spikespace);
{% endblock %}

{% block kernel %}
_global__ void _run_{{codeobj_name}}_kernel(
	unsigned int neurongroup_N,
	unsigned int num_blocks,
	int32_t index_last_element,
	int32_t* spikemonitor_i,
	int32_t* spikespace
	)
{
	using namespace brian;

	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = 0; i < neurongroup_N;)
	{
		int32_t spiking_neuron = spikespace[i];
		if(spiking_neuron != -1)
		{
			spikemonitor_i[index++] = spiking_neuron;
			i++;
		}
		else
		{
			//round to nearest multiple of N/num_blocks = start of next block
			i += neurongroup_N/num_blocks - i % (neurongroup_N/num_blocks);
		}
	}
}
{% endblock %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	std::cout << "Number of spikes: " << {{_dynamic_i}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
