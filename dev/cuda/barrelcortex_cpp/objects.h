
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "network.h"


namespace brian {

//////////////// clocks ///////////////////
extern Clock layer4_custom_operation_clock;
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_feedforward__synaptic_post;
extern std::vector<int32_t> _dynamic_array_feedforward__synaptic_pre;
extern std::vector<double> _dynamic_array_feedforward_A_source;
extern std::vector<double> _dynamic_array_feedforward_A_target;
extern std::vector<double> _dynamic_array_feedforward_lastupdate;
extern std::vector<double> _dynamic_array_feedforward_post_delay;
extern std::vector<double> _dynamic_array_feedforward_pre_delay;
extern std::vector<double> _dynamic_array_feedforward_w;
extern std::vector<int32_t> _dynamic_array_recurrent_exc__synaptic_post;
extern std::vector<int32_t> _dynamic_array_recurrent_exc__synaptic_pre;
extern std::vector<double> _dynamic_array_recurrent_exc_lastupdate;
extern std::vector<double> _dynamic_array_recurrent_exc_pre_delay;
extern std::vector<double> _dynamic_array_recurrent_exc_w;
extern std::vector<int32_t> _dynamic_array_recurrent_inh__synaptic_post;
extern std::vector<int32_t> _dynamic_array_recurrent_inh__synaptic_pre;
extern std::vector<double> _dynamic_array_recurrent_inh_lastupdate;
extern std::vector<double> _dynamic_array_recurrent_inh_pre_delay;

//////////////// arrays ///////////////////
extern int32_t *_array_feedforward_N_incoming;
extern const int _num__array_feedforward_N_incoming;
extern int32_t *_array_feedforward_N_outgoing;
extern const int _num__array_feedforward_N_outgoing;
extern int32_t *_array_layer23__spikespace;
extern const int _num__array_layer23__spikespace;
extern int32_t *_array_layer23_barrel_idx;
extern const int _num__array_layer23_barrel_idx;
extern double *_array_layer23_ge;
extern const int _num__array_layer23_ge;
extern double *_array_layer23_gi;
extern const int _num__array_layer23_gi;
extern int32_t *_array_layer23_i;
extern const int _num__array_layer23_i;
extern double *_array_layer23_lastspike;
extern const int _num__array_layer23_lastspike;
extern char *_array_layer23_not_refractory;
extern const int _num__array_layer23_not_refractory;
extern int32_t *_array_layer23_subgroup_1__sub_idx;
extern const int _num__array_layer23_subgroup_1__sub_idx;
extern int32_t *_array_layer23_subgroup__sub_idx;
extern const int _num__array_layer23_subgroup__sub_idx;
extern double *_array_layer23_v;
extern const int _num__array_layer23_v;
extern double *_array_layer23_vt;
extern const int _num__array_layer23_vt;
extern double *_array_layer23_x;
extern const int _num__array_layer23_x;
extern double *_array_layer23_y;
extern const int _num__array_layer23_y;
extern int32_t *_array_layer4__spikespace;
extern const int _num__array_layer4__spikespace;
extern int32_t *_array_layer4_barrel_x;
extern const int _num__array_layer4_barrel_x;
extern int32_t *_array_layer4_barrel_y;
extern const int _num__array_layer4_barrel_y;
extern double *_array_layer4_direction;
extern const int _num__array_layer4_direction;
extern int32_t *_array_layer4_i;
extern const int _num__array_layer4_i;
extern double *_array_layer4_selectivity;
extern const int _num__array_layer4_selectivity;
extern double *_array_layer4_stim_start_time;
extern const int _num__array_layer4_stim_start_time;
extern double *_array_layer4_stim_start_x;
extern const int _num__array_layer4_stim_start_x;
extern double *_array_layer4_stim_start_y;
extern const int _num__array_layer4_stim_start_y;
extern int32_t *_array_recurrent_exc_N_incoming;
extern const int _num__array_recurrent_exc_N_incoming;
extern int32_t *_array_recurrent_exc_N_outgoing;
extern const int _num__array_recurrent_exc_N_outgoing;
extern int32_t *_array_recurrent_inh_N_incoming;
extern const int _num__array_recurrent_inh_N_incoming;
extern int32_t *_array_recurrent_inh_N_outgoing;
extern const int _num__array_recurrent_inh_N_outgoing;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
extern double *_static_array__array_layer23_lastspike;
extern const int _num__static_array__array_layer23_lastspike;
extern char *_static_array__array_layer23_not_refractory;
extern const int _num__static_array__array_layer23_not_refractory;

//////////////// synapses /////////////////
// feedforward
extern Synapses<double> feedforward;
extern SynapticPathway<double> feedforward_post;
extern SynapticPathway<double> feedforward_pre;
// recurrent_exc
extern Synapses<double> recurrent_exc;
extern SynapticPathway<double> recurrent_exc_pre;
// recurrent_inh
extern Synapses<double> recurrent_inh;
extern SynapticPathway<double> recurrent_inh_pre;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


