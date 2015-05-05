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
	extern __shared__ int32_t spike_cache[];
	
	spike_cache[tid] = -1;
	{{_spikespace}}[_idx] = -1;

	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		{{_spikespace}}[N] = 0;
	}
	__syncthreads();

	for(int spike_idx={{_lastindex}}[0] + _idx; spike_idx < _numspike_time; spike_idx+=THREADS_PER_BLOCK)
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
	    if(neuron_id >= bid*THREADS_PER_BLOCK && neuron_id < (bid+1)*THREADS_PER_BLOCK)
	    {
			spike_cache[tid] = neuron_id;
		}
	}

	if(tid != 0)
	{
		return;
	}

	int first_neuron_in_block = _idx;	//tid = 0, so neuron_id = bid*num_threads_per_block = start of block no. bid
	int num_spikes_in_block = 0;
	for(int i = 0; (i < THREADS_PER_BLOCK) && (first_neuron_in_block + i < N); i++)
	{
		if(spike_cache[i] != -1)
		{
			//spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
			int spiking_neuron = spike_cache[i];
			{{_spikespace}}[first_neuron_in_block + num_spikes_in_block] = spiking_neuron;
			num_spikes_in_block++;
		}
	}
	//add number of spikes of all blocks together
	//last element of spikespace holds total number of spikes
	atomicAdd(&{{_spikespace}}[N], num_spikes_in_block);
	atomicAdd(&{{_lastindex}}[0], num_spikes_in_block);
{% endblock %}

{% block kernel_call %}
kernel_{{codeobj_name}}<<<num_blocks(N),num_threads(N), num_threads(N)*mem_per_thread()>>>(
		num_threads(N),
		%HOST_PARAMETERS%
	);
{% endblock %}