{% extends 'common_group.cu' %}
{% block extra_headers %}
#include "cuda_functions.h"
{% endblock %}

{% block kernel %}
{# USES_VARIABLES { _synaptic_post, _synaptic_pre, N_post, N_pre } #}
__global__ void kernel_{{codeobj_name}}(
	int num_blocks,
	%DEVICE_PARAMETERS%
	)
{
    {% set _target_var_array = get_array_name(_target_var) %}
    using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	%KERNEL_VARIABLES%

	//// MAIN CODE ////////////
	double _local_sum = 0.0;
	
	for(int _idx=tid; _idx<N_pre; _idx += num_blocks)
	{
		// vector code
	    int _vectorisation_idx = bid;
	    int post_id = {{_synaptic_post}}[_idx];
	    
		if(post_id == bid)
        {
            {{vector_code|autoindent}}
			_local_sum += _synaptic_var;
		}
		
	}
	atomicAdd(&{{_target_var_array}}[bid], _local_sum);
}	
{% endblock %}

{% block kernel_call %}
	int num_blocks = 16;
	kernel_{{codeobj_name}}<<<N_post, num_blocks>>>(
			num_blocks,
			%HOST_PARAMETERS%
		);
{% endblock %}
