{% extends 'common_group.cu' %}
{% block extra_device_helper %}
int mem_per_thread(){
	return sizeof(int32_t);
}
{% endblock %}


{% block maincode %}
    {# USES_VARIABLES {_spikespace, N, t, dt, neuron_index, spike_time, period, _lastindex } #}

    float padding_before = fmod(t, period);
    float padding_after  = fmod(t+dt, period);
    double epsilon       = 1e-3*dt;

    // We need some precomputed values that will be used during looping
    bool not_first_spike = ({{_lastindex}}[0] > 0);
    bool not_end_period  = (fabs(padding_after) > epsilon);
    bool test;
	
	//// MAIN CODE ////////////
	// scalar code
	
	{{_spikespace}}[_idx] = -1;

	if(tid == 0)
	{
		//init number of spikes with 0
		{{_spikespace}}[N] = 0;
	}
	__syncthreads();

	for(int spike_idx = {{_lastindex}}[0] + tid; spike_idx < _numspike_time; spike_idx += THREADS_PER_BLOCK)
	{
		if (not_end_period)
		{
	        test = ({{spike_time}}[spike_idx] > padding_after) || (fabs({{spike_time}}[spike_idx] - padding_after) < epsilon);
	    }
	    else
	    {
	        // If we are in the last timestep before the end of the period, we remove the first part of the
	        // test, because padding will be 0
	        test = (fabs({{spike_time}}[spike_idx] - padding_after) < epsilon);
	    }
	    if (test)
	    {
	        break;
	    }
	    int32_t neuron_id = {{neuron_index}}[spike_idx];
    	int32_t spikespace_index = atomicAdd(&{{_spikespace}}[N], 1);
		atomicAdd(&{{_lastindex}}[0], 1);
    	{{_spikespace}}[spikespace_index] = neuron_id;
		__syncthreads();
	}
{% endblock %}

{% block kernel_call %}
kernel_{{codeobj_name}}<<<1,max_threads_per_block>>>(
		max_threads_per_block,
		%HOST_PARAMETERS%
	);
{% endblock %}