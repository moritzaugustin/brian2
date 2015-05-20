{% extends 'common_group.cu' %}

{% block extra_device_helper %}

#define MEM_PER_THREAD (sizeof(double))

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
	int num_threads,
	%DEVICE_PARAMETERS%
	)
{
    {% set _target_var_array = get_array_name(_target_var) %}
    using namespace brian;

	extern __shared__ char shared_mem[];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int neuron_id = bid/N_post;
	unsigned int num_block_for_neuron = bid % N_post;
	double* shared_double_mem = (double*) shared_mem;
	%KERNEL_VARIABLES%

	//// MAIN CODE ////////////
	double _local_sum = 0.0;
	shared_double_mem[tid] = 0.0;
	if(tid == 0)
	{
	    {{_target_var_array}}[bid] = 0.0;
	}
	
	for(int _idx=tid; _idx<N_pre; _idx += num_threads)
	{
		// vector code
	    int _vectorisation_idx = bid;
	    int post_id = {{_synaptic_post}}[_idx];
	    
		if(post_id == neuron_id)
        {
            {{vector_code|autoindent}}
			shared_double_mem[tid] += _synaptic_var;
		}
	}
	if(tid != 0)
	{
		return;
	}
	for(int _idx = 0; _idx < num_threads; _idx++)
	{
		_local_sum += shared_double_mem[_idx];
	}
	atomicAddDouble(&{{_target_var_array}}[neuron_id], _local_sum);
}	
{% endblock %}

{% block kernel_call %}
	int num_blocks = num_parallel_blocks * N_post;
	unsigned int num_threads = max_shared_mem_size / MEM_PER_THREAD;
	num_threads = num_threads < max_threads_per_block? num_threads : max_threads_per_block;	// get min of both
	kernel_{{codeobj_name}}<<<num_blocks, num_threads, num_threads*MEM_PER_THREAD>>>(
			num_threads,
			%HOST_PARAMETERS%
		);
{% endblock %}
