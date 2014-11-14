#include<stdlib.h>
#include "objects.h"
#include<ctime>

#include "code_objects/feedforward_group_variable_set_conditional_codeobject.h"
#include "code_objects/feedforward_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/feedforward_post_codeobject.h"
#include "code_objects/feedforward_post_initialise_queue.h"
#include "code_objects/feedforward_post_push_spikes.h"
#include "code_objects/feedforward_pre_codeobject.h"
#include "code_objects/feedforward_pre_initialise_queue.h"
#include "code_objects/feedforward_pre_push_spikes.h"
#include "code_objects/feedforward_synapses_create_codeobject.h"
#include "code_objects/layer23_group_variable_set_conditional_codeobject.h"
#include "code_objects/layer23_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/layer23_resetter_codeobject.h"
#include "code_objects/layer23_stateupdater_codeobject.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/layer23_thresholder_codeobject.h"
#include "code_objects/layer4_custom_operation_codeobject.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/layer4_stateupdater_codeobject.h"
#include "code_objects/layer4_thresholder_codeobject.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject_2.h"
#include "code_objects/recurrent_exc_pre_codeobject.h"
#include "code_objects/recurrent_exc_pre_initialise_queue.h"
#include "code_objects/recurrent_exc_pre_push_spikes.h"
#include "code_objects/recurrent_exc_synapses_create_codeobject.h"
#include "code_objects/recurrent_exc_synapses_create_codeobject_1.h"
#include "code_objects/recurrent_inh_group_variable_set_conditional_codeobject.h"
#include "code_objects/recurrent_inh_pre_codeobject.h"
#include "code_objects/recurrent_inh_pre_initialise_queue.h"
#include "code_objects/recurrent_inh_pre_push_spikes.h"
#include "code_objects/recurrent_inh_synapses_create_codeobject.h"


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


