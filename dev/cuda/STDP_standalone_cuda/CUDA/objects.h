#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"

#include <curand.h>
#include <thrust/device_vector.h>

namespace brian {

extern int num_blocks_sequential;
extern int num_blocks_parallel;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;
extern Network magicnetwork;

//////////////// dynamic arrays ///////////
extern thrust::device_vector<double> _dynamic_array_ratemonitor_rate;
extern thrust::device_vector<double> _dynamic_array_ratemonitor_t;
extern thrust::device_vector<double>* _dynamic_array_statemonitor__recorded_w;
extern thrust::device_vector<double> _dynamic_array_statemonitor_t;
extern thrust::device_vector<double> _dynamic_array_synapses_Apost;
extern thrust::device_vector<double> _dynamic_array_synapses_Apre;
extern thrust::device_vector<double> _dynamic_array_synapses_lastupdate;
extern thrust::device_vector<double> _dynamic_array_synapses_w;
extern thrust::device_vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern thrust::device_vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern thrust::device_vector<double> _dynamic_array_synapses_post_delay;
extern thrust::device_vector<double> _dynamic_array_synapses_pre_delay;	

extern __device__ CudaVector<int32_t> **_dynamic_array_spikemonitor_i;
extern __device__ CudaVector<double> **_dynamic_array_spikemonitor_t;

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

/////////////// static arrays /////////////
extern int32_t *_static_array__array_statemonitor__indices;
extern const int _num__static_array__array_statemonitor__indices;

//////////////// synapses /////////////////
// synapses
extern Synapses<double> synapses;
extern __device__ SynapticPathway<double> synapses_post;
extern __device__ SynapticPathway<double> synapses_pre;

///////////// for random numbers ///////////
extern float* dev_array_rands;
extern curandGenerator_t gen;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif

