#include "objects.h"
#include "code_objects/neurongroup_resetter_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_neurongroup_resetter_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_spikespace = 4001;
const int _numw = 4000;
const int _numv = 4000;
const int _numnot_refractory = 4000;
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
	double * __restrict__ _ptr_array_neurongroup_w = _array_neurongroup_w;
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;
	bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;


	const int *_spikes = _ptr_array_neurongroup__spikespace;
	const int _num_spikes = _ptr_array_neurongroup__spikespace[4000];

	//// MAIN CODE ////////////
	for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
	{
		const int _idx = _spikes[_index_spikes];
		const int _vectorisation_idx = _idx;
		double w = _ptr_array_neurongroup_w[_idx];
		double v;
		v = 0.0;
		w = w + 0.0001;
		_ptr_array_neurongroup_w[_idx] = w;
		_ptr_array_neurongroup_v[_idx] = v;
	}
}


