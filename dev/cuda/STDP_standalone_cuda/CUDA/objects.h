
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include<thrust/device_vector.h>
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"

namespace brian {

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern thrust::device_vector<double> _dynamic_array_ratemonitor_rate;
extern thrust::device_vector<double> _dynamic_array_ratemonitor_t;
extern thrust::device_vector<int32_t> _dynamic_array_spikemonitor_i;
extern thrust::device_vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_Apost;
extern std::vector<double> _dynamic_array_synapses_Apre;
extern std::vector<double> _dynamic_array_synapses_lastupdate;
extern std::vector<double> _dynamic_array_synapses_post_delay;
extern std::vector<double> _dynamic_array_synapses_pre_delay;
extern std::vector<double> _dynamic_array_synapses_w;

//////////////// arrays ///////////////////
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern int32_t * dev_array_neurongroup__spikespace;

extern double *_array_neurongroup_ge;
extern const int _num__array_neurongroup_ge;
extern double * dev_array_neurongroup_ge;

extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;

extern double *_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern double * dev_array_neurongroup_v;

extern int32_t *_array_poissongroup__spikespace;
extern const int _num__array_poissongroup__spikespace;
extern int32_t * dev_array_poissongroup__spikespace;

extern int32_t *_array_poissongroup_i;
extern const int _num__array_poissongroup_i;

extern double *_array_poissongroup_rates;
extern const int _num__array_poissongroup_rates;
extern double * dev_array_poissongroup_rates;

extern int32_t *_array_spikemonitor__count;
extern const int _num__array_spikemonitor__count;

extern int32_t *_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;

extern double *_array_statemonitor__recorded_w;
extern const int _num__array_statemonitor__recorded_w;

extern int32_t *_array_synapses_N_incoming;
extern const int _num__array_synapses_N_incoming;

extern int32_t *_array_synapses_N_outgoing;
extern const int _num__array_synapses_N_outgoing;

//////////////// dynamic arrays 2d /////////
extern DynamicArray2D<double> _dynamic_array_statemonitor__recorded_w;

/////////////// static arrays /////////////
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;

//////////////// synapses /////////////////
// synapses
extern Synapses<double> synapses;
extern SynapticPathway<double> synapses_post;
extern SynapticPathway<double> synapses_pre;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


