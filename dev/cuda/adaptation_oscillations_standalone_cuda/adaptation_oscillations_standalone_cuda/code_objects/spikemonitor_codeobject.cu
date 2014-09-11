#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define neuron_N 4000
#define BLOCKS (num_blocks_sequential)
#define THREADS 1

void _run_spikemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();

	for(int i = 0; i < neuron_N;)
	{
		int32_t spiking_neuron = _array_neurongroup__spikespace[i];
		if(spiking_neuron != -1)
		{
			_dynamic_array_spikemonitor_i.push_back(spiking_neuron);
			_dynamic_array_spikemonitor_t.push_back(t);
			i++;
		}
		else
		{
			//round to nearest multiple of N/num_blocks
			i += neuron_N/num_blocks_sequential - i % (neuron_N/num_blocks_sequential);
		}
	}
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;

	printf("Number of spikes: %lu\n", _dynamic_array_spikemonitor_i.size());
}

