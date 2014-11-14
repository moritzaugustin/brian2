#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"

#include "code_objects/feedforward_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/feedforward_group_variable_set_conditional_codeobject.h"
#include "code_objects/layer4_thresholder_codeobject.h"
#include "code_objects/layer23_group_variable_set_conditional_codeobject.h"
#include "code_objects/recurrent_inh_synapses_create_codeobject.h"
#include "code_objects/recurrent_exc_pre_push_spikes.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/recurrent_exc_synapses_create_codeobject.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject.h"
#include "code_objects/feedforward_post_codeobject.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/feedforward_pre_push_spikes.h"
#include "code_objects/feedforward_pre_codeobject.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/recurrent_inh_pre_codeobject.h"
#include "code_objects/recurrent_exc_pre_initialise_queue.h"
#include "code_objects/feedforward_post_initialise_queue.h"
#include "code_objects/layer4_stateupdater_codeobject.h"
#include "code_objects/layer4_custom_operation_codeobject.h"
#include "code_objects/recurrent_inh_group_variable_set_conditional_codeobject.h"
#include "code_objects/feedforward_synapses_create_codeobject.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject.h"
#include "code_objects/recurrent_inh_pre_push_spikes.h"
#include "code_objects/layer23_stateupdater_codeobject.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/layer23_thresholder_codeobject.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/recurrent_exc_pre_codeobject.h"
#include "code_objects/layer23_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/recurrent_inh_pre_initialise_queue.h"
#include "code_objects/feedforward_post_push_spikes.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/feedforward_pre_initialise_queue.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/recurrent_exc_synapses_create_codeobject_1.h"
#include "code_objects/layer23_resetter_codeobject.h"


#include <iostream>
#include <fstream>


        void report_progress(const double elapsed, const double completed, const double duration)
        {
            if (completed == 0.0)
            {
                std::cout << "Starting simulation for duration " << duration << " s";
            } else
            {
                std::cout << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << elapsed << " s";
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    std::cout << ", estimated " << remaining << " s remaining.";
                }
            }

            std::cout << std::endl << std::flush;
        }
        


int main(int argc, char **argv)
{

	brian_start();

	{
		using namespace brian;

		
                
        _run_layer4_group_variable_set_conditional_codeobject();
        _run_layer4_group_variable_set_conditional_codeobject_1();
        _run_layer4_group_variable_set_conditional_codeobject_2();
        
                        
                        for(int i=0; i<_num__static_array__array_layer23_lastspike; i++)
                        {
                            _array_layer23_lastspike[i] = _static_array__array_layer23_lastspike[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__static_array__array_layer23_not_refractory; i++)
                        {
                            _array_layer23_not_refractory[i] = _static_array__array_layer23_not_refractory[i];
                        }
                        
        _run_layer23_group_variable_set_conditional_codeobject();
        _run_layer23_group_variable_set_conditional_codeobject_1();
        _run_layer23_subgroup_group_variable_set_conditional_codeobject();
        _run_layer23_subgroup_group_variable_set_conditional_codeobject_1();
        _run_layer23_subgroup_group_variable_set_conditional_codeobject_2();
        _run_layer23_subgroup_1_group_variable_set_conditional_codeobject();
        _run_layer23_subgroup_1_group_variable_set_conditional_codeobject_1();
        _run_layer23_subgroup_1_group_variable_set_conditional_codeobject_2();
        _run_feedforward_synapses_create_codeobject();
        _run_feedforward_group_variable_set_conditional_codeobject();
        _run_recurrent_exc_synapses_create_codeobject();
        _run_recurrent_exc_group_variable_set_conditional_codeobject();
        _run_recurrent_exc_synapses_create_codeobject_1();
        _run_recurrent_exc_group_variable_set_conditional_codeobject_1();
        _run_recurrent_inh_synapses_create_codeobject();
        _run_feedforward_group_variable_set_conditional_codeobject_1();
        _run_recurrent_exc_group_variable_set_conditional_codeobject_2();
        _run_recurrent_inh_group_variable_set_conditional_codeobject();
        _run_feedforward_post_initialise_queue();
        _run_feedforward_pre_initialise_queue();
        _run_recurrent_exc_pre_initialise_queue();
        _run_recurrent_inh_pre_initialise_queue();
        magicnetwork.clear();
        magicnetwork.add(&layer4_custom_operation_clock, _run_layer4_custom_operation_codeobject);
        magicnetwork.add(&defaultclock, _run_layer23_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_layer4_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_layer23_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_layer4_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_feedforward_post_push_spikes);
        magicnetwork.add(&defaultclock, _run_feedforward_post_codeobject);
        magicnetwork.add(&defaultclock, _run_feedforward_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_feedforward_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_recurrent_exc_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_recurrent_exc_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_recurrent_inh_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_recurrent_inh_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_layer23_resetter_codeobject);
        magicnetwork.run(5.0, report_progress, 10.0);
        _debugmsg_feedforward_post_codeobject();
        
        _debugmsg_feedforward_pre_codeobject();
        
        _debugmsg_recurrent_inh_pre_codeobject();
        
        _debugmsg_recurrent_exc_pre_codeobject();

	}

	brian_end();

	return 0;
}