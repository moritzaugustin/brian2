#include<stdlib.h>
#include "objects.h"
#include<ctime>

#include "code_objects/neurongroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/neurongroup_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/neurongroup_thresholder_codeobject.h"
#include "code_objects/ratemonitor_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_initialise_queue.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include "code_objects/synapses_synapses_create_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	srand((unsigned int)time(NULL));
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


