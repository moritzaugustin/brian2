{% extends 'common_group.cu' %}

{% block extra_device_helper %}
	//atomic add is currently not supported natively for double values
__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
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
	if(tid == 0)
	{
	    {{_target_var_array}}[bid] = 0.0;
	}
	
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
	atomicAddDouble(&{{_target_var_array}}[bid], _local_sum);
}	
{% endblock %}

{% block kernel_call %}
	int num_blocks = 16;
	kernel_{{codeobj_name}}<<<N_post, num_blocks>>>(
			num_blocks,
			%HOST_PARAMETERS%
		);
{% endblock %}
