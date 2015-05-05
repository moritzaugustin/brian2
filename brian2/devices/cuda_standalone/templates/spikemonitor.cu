{% extends 'common_group.cu' %}
{# USES_VARIABLES { t, i, _clock_t, _spikespace, _count,
                    _source_start, _source_stop} #}
                    
{% block extra_device_helper %}
__device__ int32_t dev_num_spikes;	//only needed for subgroups

__global__ void get_num_spikes_subgroup(
	unsigned int neurongroup_N,
	unsigned int block_size,
	int32_t* spikespace)
{	
 	dev_num_spikes = 0;

	for(int i = 0; i < _source_stop;)
	{
		int32_t spiking_neuron = spikespace[i];
		if(spiking_neuron != -1)
		{
			if(spiking_neuron >= _source_start && spiking_neuron < _source_stop)
			{
				dev_num_spikes++;
			}
			i++;
		}
		else
		{
			//round to nearest multiple of block_size (= start of next block)
			i += block_size - i % block_size;
		}
	}
}
{% endblock %}

{% block extra_maincode %}
unsigned int start_spikes = dev{{_dynamic_i}}.size();
int32_t num_spikes;
//check if subgroup of a neurongroup
if(_num_spikespace-1 == (_source_stop - _source_start))
{
	//if not a subgroup, just copy last value of spikespace (= num_spikes)
	{% set spikespace_name =  get_array_name(variables['_spikespace'], access_data=False) %}
	cudaMemcpy(&num_spikes, &dev{{spikespace_name}}[_num_spikespace-1], sizeof(int32_t), cudaMemcpyDeviceToHost);
}
else
{
	//if subgroup, launch kernel to find number of spikes
	get_num_spikes_subgroup<<<1,1>>>(
		_num_spikespace-1,
		num_threads(_num_spikespace-1),
		dev{{spikespace_name}});
	cudaMemcpyFromSymbol(&num_spikes, dev_num_spikes, sizeof(num_spikes), 0, cudaMemcpyDeviceToHost);
}

dev{{_dynamic_i}}.resize(dev{{_dynamic_i}}.size() + num_spikes, 0);	//push dummy value
dev{{_dynamic_t}}.resize(dev{{_dynamic_t}}.size() + num_spikes, _clock_t);
{% endblock %}

{% block kernel_call %}
{% set spikespace_name =  get_array_name(variables['_spikespace'], access_data=False) %}
_run_{{codeobj_name}}_kernel<<<1, 1>>>(
		_num_spikespace-1,
		num_blocks(_num_spikespace-1),
		num_threads(_num_spikespace-1),
		start_spikes,
		thrust::raw_pointer_cast(&(dev{{_dynamic_i}}[0])),
		dev{{spikespace_name}});
{% endblock %}

{% block kernel %}
__global__ void _run_{{codeobj_name}}_kernel(
	unsigned int neurongroup_N,
	unsigned int num_blocks,
	unsigned int block_size,
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
			if(spiking_neuron >= _source_start && spiking_neuron < _source_stop)
			{
				spikemonitor_i[index_last_element++] = spiking_neuron;
			}
			i++;
		}
		else
		{
			//round to nearest multiple of block_size (= start of next block)
			i += block_size - i % block_size;
		}
	}
}
{% endblock %}

{% block extra_functions_cu %}
void _debugmsg_{{codeobj_name}}()
{
	using namespace brian;
	std::cout << "Number of spikes: " << dev{{_dynamic_i}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_debugmsg_{{codeobj_name}}();
{% endmacro %}
