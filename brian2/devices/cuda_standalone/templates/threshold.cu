{% extends 'common_group.cu' %}

{% block extra_device_helper %}
int mem_per_thread(){
	return sizeof(bool);
}
{% endblock %}


{% block maincode %}
	{# USES_VARIABLES { t, _spikespace, N } #}
	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
	extern __shared__ bool spike_cache[];
	{{scalar_code|autoindent}}

	spike_cache[_idx] = false;
	{{_spikespace}}[_idx] = -1;

	if(tid == 0 && bid == 0)
	{
		//init number of spikes with 0
		{{_spikespace}}[N] = 0;
	}
	__syncthreads();

	{{vector_code|autoindent}}
	if(_cond) {
		spike_cache[_idx] = true;
		{% if _uses_refractory %}
		// We have to use the pointer names directly here: The condition
		// might contain references to not_refractory or lastspike and in
		// that case the names will refer to a single entry.
		{{not_refractory}}[_idx] = false;
		{{lastspike}}[_idx] = t;
		{% endif %}
	}

	if(tid != 0)
	{
		return;
	}

	int first_neuron_in_block = _idx;	//tid = 0, so neuron_id = bid*num_threads_per_block = start of block no. bid
	int num_spikes_in_block = 0;
	for(int i = 0; (i < THREADS_PER_BLOCK) && (first_neuron_in_block + i < N); i++)
	{
		if(spike_cache[i])
		{
			//spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
			int spiking_neuron = first_neuron_in_block + i;
			{{_spikespace}}[first_neuron_in_block + num_spikes_in_block] = spiking_neuron;
			num_spikes_in_block++;
		}
	}
	//add number of spikes of all blocks together
	//last element of spikespace holds total number of spikes
	atomicAdd(&{{_spikespace}}[N], num_spikes_in_block);
{% endblock %}

{% block kernel_call %}
kernel_{{codeobj_name}}<<<num_blocks(N),num_threads(N), num_threads(N)*mem_per_thread()>>>(
		num_threads(N),
		%HOST_PARAMETERS%
	);
{% endblock %}
