{% extends 'common_group.cu' %}

{% block kernel %}
{# USES_VARIABLES { _synaptic_post, _synaptic_pre, N_post } #}
__global__ void kernel_{{codeobj_name}}(
	%DEVICE_PARAMETERS%
	)
{
    {% set _target_var_array = get_array_name(_target_var) %}
    using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%

	if(_idx >= N)
	{
		return;
	}
	
	//// MAIN CODE ////////////
	double _local_sum = 0.0;

	for(int _idx=0; _idx<_num_synaptic_post; _idx++)
	{
		// vector code
	    int _vectorisation_idx = tid;
	    int post_id = {{_synaptic_post}}[_idx];
	    
        {{vector_code|autoindent}}
        
		_local_sum += _synaptic_var;
		
	}
	
    {{_target_var_array}}[tid] += _local_sum;
}	
{% endblock %}

{% block kernel_call %}
	kernel_{{codeobj_name}}<<<N_post,1>>>(
			%HOST_PARAMETERS%
		);
{% endblock %}
