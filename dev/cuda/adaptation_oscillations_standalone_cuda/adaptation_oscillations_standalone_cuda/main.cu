#include<stdlib.h>
#include "objects.h"
#include "random.h"
#include<ctime>
#include "run.h"

#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/synapses_synapses_create_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"

#include<iostream>

/* 
 * original cpu version
 * sim_time = 1s, sparsity = 0.05
 * 150k spikes, 800k synapses
 * 3.2s run_time
 */

int main(int argc, char **argv)
{
	size_t limit = 1*1024*1024*1024;	//500MB should be enough for now
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
	cudaDeviceSynchronize();

	std::clock_t start = std::clock();
	brian_start();

	{
		using namespace brian;
		for(int i=0; i<_num__static_array__array_neurongroup_lastspike; i++)
		{
                	_array_neurongroup_lastspike[i] = _static_array__array_neurongroup_lastspike[i];
		}
		cudaMemcpy(dev_array_neurongroup_lastspike, _array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice);
                for(int i=0; i<_num__static_array__array_neurongroup_not_refractory; i++)
		{
                	_array_neurongroup_not_refractory[i] = _static_array__array_neurongroup_not_refractory[i];
		}
		cudaMemcpy(dev_array_neurongroup_not_refractory, _array_neurongroup_not_refractory, sizeof(double)*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice);

		_run_neurongroup_group_variable_set_conditional_codeobject();
		_run_neurongroup_group_variable_set_conditional_codeobject_1();
		_run_synapses_synapses_create_codeobject();
		_run_synapses_group_variable_set_conditional_codeobject();
		_run_synapses_group_variable_set_conditional_codeobject_1();
		for(int i=0; i<_num__static_array__array_statemonitor__indices; i++)
		{
			_array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
		}
		_run_synapses_group_variable_set_conditional_codeobject_2();
		_run_synapses_pre_initialise_queue();

		/*
		 * IMPORTANT change of order:
		 * now: resetter is always last
		 * since now the resetter also overwrites all data in our spikespace
		 */
		magicnetwork.clear();
		magicnetwork.add(&defaultclock, _random_number_generation);
		magicnetwork.add(&defaultclock, _run_synapses_stateupdater_codeobject);
		magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
		magicnetwork.add(&defaultclock, _run_neurongroup_thresholder_codeobject);
		magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
		magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
		magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
		magicnetwork.add(&defaultclock, _run_ratemonitor_codeobject);
		magicnetwork.add(&defaultclock, _run_neurongroup_resetter_codeobject);
		magicnetwork.add(&defaultclock, _run_statemonitor_codeobject);
		magicnetwork.run(1.0);
		_debugmsg_spikemonitor_codeobject();
		_debugmsg_synapses_pre_codeobject();
	}

	double _run_duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
	std::cout << "Simulation time: " << _run_duration << endl;

	brian_end();
	cudaDeviceReset();
	return 0;
}
