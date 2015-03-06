{% extends 'common_synapses.cu' %}

{% block extra_headers %}
{{ super() }}
#include<iostream>
{% endblock %}

{% block kernel %}
{% endblock %}

{% block kernel_call %}
{% endblock %}

{% block extra_maincode %}
	{# USES_VARIABLES { _synaptic_pre, _synaptic_post, rand,
	                    N_incoming, N_outgoing } #}

	{{scalar_code|autoindent}}
	unsigned int MAX_SYN_N = _num_all_pre*_num_all_post;

	//generate MAX_SYN_N random numbers
	float* _array_random_float_numbers;
	_array_random_float_numbers = (float*)malloc(sizeof(float)*MAX_SYN_N);
	if(!_array_random_float_numbers)
	{
		printf("ERROR while allocating memory with size %ld()\n", sizeof(float)*MAX_SYN_N);
	}
	curandGenerator_t gen;
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, _array_random_float_numbers, MAX_SYN_N);

	//these two vectors just cache everything on the CPU-side
	//data is copied to GPU at the end
	thrust::host_vector<int32_t> temp_synaptic_post;
	thrust::host_vector<int32_t> temp_synaptic_pre;

	{{pointers_lines|autoindent}}

	int syn_id = {{_dynamic__synaptic_pre}}.size();
	for(int _i = 0; _i < _num_all_pre; _i++)
	{
		synapses_by_pre_neuron.push_back(syn_id);
		for(int _j = 0; _j < _num_all_post; _j++)
		{
			{% block maincode_inner %}
		    const int _vectorisation_idx = _j;
			{{vector_code|autoindent}}
			// Add to buffer
			if(_cond)
			{
				if (_p != 1.0)
				{
					float r = _array_random_float_numbers[_i*_numN_outgoing + _j];
					if (r >= _p)
					{
						continue;
					}
				}
				for (int _repetition = 0; _repetition < _n; _repetition++)
				{
					{{N_outgoing}}[_pre_idx] += 1;
					{{N_incoming}}[_post_idx] += 1;
					temp_synaptic_pre.push_back(_pre_idx);
					temp_synaptic_post.push_back(_post_idx);
					syn_id++;
				}
			}
			{% endblock %}
		}
	}
	synapses_by_pre_neuron.push_back(syn_id);

	dev{{_dynamic__synaptic_pre}} = temp_synaptic_post;
	dev{{_dynamic__synaptic_post}} = temp_synaptic_pre;
    
	// now we need to resize all registered variables
	const int32_t newsize = dev{{_dynamic__synaptic_pre}}.size();
	{% for variable in owner._registered_variables | sort(attribute='name') %}
	{% set varname = get_array_name(variable, access_data=False) %}
	dev{{varname}}.resize(newsize);
	{% endfor %}
	// Also update the total number of synapses
	{{owner.name}}._N_value = newsize;

	free(_array_random_float_numbers);
{% endblock %}
