
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"
#include "brianlib/cudaVector.h"

#include <curand.h>
#include <thrust/device_vector.h>

namespace brian {

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern std::vector<double> _dynamic_array_ratemonitor_rate;
extern std::vector<double> _dynamic_array_ratemonitor_t;
extern thrust::device_vector<double> _dynamic_array_synapses_c;
extern thrust::device_vector<double> _dynamic_array_synapses_lastupdate;
extern thrust::device_vector<double> _dynamic_array_synapses_pre_delay;
extern thrust::device_vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern thrust::device_vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern thrust::device_vector<int32_t> synapses_by_pre_neuron;	//neuron 0 has syns arr[0] to (arr[1] - 1), etc...
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;
extern std::vector<double> _dynamic_array_statemonitor_t;
extern thrust::device_vector<double>* _dynamic_array_statemonitor__recorded_v;
extern thrust::device_vector<double>* _dynamic_array_statemonitor__recorded_w;

//////////////// arrays ///////////////////
extern int32_t *_array_neurongroup__spikespace;
extern int32_t *dev_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;

extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;

extern double *_array_neurongroup_lastspike;
extern double *dev_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;

extern bool *_array_neurongroup_not_refractory;
extern bool *dev_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;

extern double * _array_neurongroup_v;
extern double * dev_array_neurongroup_v;
extern const int _num__array_neurongroup_v;

extern double *_array_neurongroup_w;
extern double * dev_array_neurongroup_w;
extern const int _num__array_neurongroup_w;

extern int32_t *_array_spikemonitor__count;
extern const int _num__array_spikemonitor__count;

extern int32_t *_array_statemonitor__indices;
extern const int _num__array_statemonitor__indices;

extern int32_t *_array_synapses_N_incoming;
extern const int _num__array_synapses_N_incoming;

extern int32_t *_array_synapses_N_outgoing;
extern const int _num__array_synapses_N_outgoing;

extern unsigned int* dev_size_by_pre;
extern int32_t** dev_synapses_id_by_pre;
extern int32_t** dev_post_neuron_by_pre;
extern unsigned int** dev_delay_by_pre;

/////////////// static arrays /////////////
extern double *_static_array__array_neurongroup_lastspike;
extern const int _num__static_array__array_neurongroup_lastspike;
extern bool *_static_array__array_neurongroup_not_refractory;
extern const int _num__static_array__array_neurongroup_not_refractory;
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;

//////////////// synapses /////////////////
// synapses
extern Synapses<double> synapses;
extern __device__ SynapticPathway<double> synapses_pre;

//////////////// random numbers /////////////////
extern float* dev_array_random_floats;
extern curandGenerator_t random_float_generator;

extern unsigned int num_blocks;
extern unsigned int max_threads_per_block;
extern unsigned int max_shared_mem_size;

extern unsigned int neurongroup_N;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif

