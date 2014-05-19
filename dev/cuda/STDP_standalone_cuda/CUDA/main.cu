#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include "run.h"

#include "random.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/synapses_post_initialise_queue.h"
#include "code_objects/poissongroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_post_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/poissongroup_thresholder_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/synapses_post_push_spikes.h"
#include "code_objects/synapses_synapses_create_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"


#include<iostream>

int main(int argc, char **argv)
{
	std::clock_t start = std::clock();

	brian_start();

	{
		using namespace brian;
		_run_poissongroup_group_variable_set_conditional_codeobject();	
		_run_synapses_synapses_create_codeobject();					//not parallel
		_run_synapses_group_variable_set_conditional_codeobject();
		for(int i=0; i<_num__static_array__array_statemonitor__indices; i++)
		{
		    _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
		}
		_run_synapses_group_variable_set_conditional_codeobject_1();
		_run_synapses_pre_initialise_queue();
		_run_synapses_post_initialise_queue();

		magicnetwork.clear();
		magicnetwork.add(&defaultclock, _get_rands);
		magicnetwork.add(&defaultclock, _run_synapses_stateupdater_codeobject);
		magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
		magicnetwork.add(&defaultclock, _run_neurongroup_thresholder_codeobject);
		magicnetwork.add(&defaultclock, _run_poissongroup_thresholder_codeobject);
		magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
		magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
		magicnetwork.add(&defaultclock, _run_synapses_post_push_spikes);
		magicnetwork.add(&defaultclock, _run_synapses_post_codeobject);
		magicnetwork.add(&defaultclock, _run_neurongroup_resetter_codeobject);
		magicnetwork.add(&defaultclock, _run_spikemonitor_codeobject);
		magicnetwork.add(&defaultclock, _run_ratemonitor_codeobject);
		magicnetwork.add(&defaultclock, _run_statemonitor_codeobject);			//cudaMemcpy, not on GPU
		magicnetwork.run(1.0);

		_debugmsg_spikemonitor_codeobject();
		_debugmsg_synapses_post_codeobject();
		_debugmsg_synapses_pre_codeobject();
	}
	double _run_duration = (std::clock()-start)/(double)CLOCKS_PER_SEC;
	std::cout << "Simulation time: " << _run_duration << endl;

	brian_end();
	return 0;
}
