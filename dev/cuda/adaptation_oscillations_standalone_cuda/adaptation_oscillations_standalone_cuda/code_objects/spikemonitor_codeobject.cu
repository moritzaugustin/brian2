#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

void _run_spikemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();

	//spikespace is already on CPU-side, so we just iterate over it
	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = 0; i < neurongroup_N;)
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
			//round to nearest multiple of N/num_blocks = start of next block
			i += neurongroup_N/num_blocks - i % (neurongroup_N/num_blocks);
		}
	}
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;

	printf("Number of spikes: %lu\n", _dynamic_array_spikemonitor_i.size());
}

