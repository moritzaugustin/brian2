#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

__global__ void _run_spikemonitor_codeobject_kernel(
	unsigned int _neurongroup_N,
	unsigned int _num_blocks,
	unsigned int index,
	int32_t* _array_spikemonitor_i,
	int32_t* _array_neurongroup__spikespace)
{
	using namespace brian;

	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = 0; i < _neurongroup_N;)
	{
		int32_t spiking_neuron = _array_neurongroup__spikespace[i];
		if(spiking_neuron != -1)
		{
			_array_spikemonitor_i[index++] = spiking_neuron;
			i++;
		}
		else
		{
			//round to nearest multiple of N/num_blocks = start of next block
			i += _neurongroup_N/_num_blocks - i % (_neurongroup_N/_num_blocks);
		}
	}
}

void _run_spikemonitor_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();

	unsigned int start_spikes = _dynamic_array_spikemonitor_t.size();
	int32_t num_spikes;
	cudaMemcpy(&num_spikes, &dev_array_neurongroup__spikespace[neurongroup_N], sizeof(int32_t), cudaMemcpyDeviceToHost);

	for(int i = 0; i < num_spikes; i++)
	{
		_dynamic_array_spikemonitor_i.push_back(0);	//push dummy value
		_dynamic_array_spikemonitor_t.push_back(t);
	}

	_run_spikemonitor_codeobject_kernel<<<1, 1>>>(
		neurongroup_N,
		num_blocks,
		start_spikes,
		thrust::raw_pointer_cast(&(_dynamic_array_spikemonitor_i[0])),
		dev_array_neurongroup__spikespace);
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;

	printf("Number of spikes: %lu\n", _dynamic_array_spikemonitor_t.size());
}

