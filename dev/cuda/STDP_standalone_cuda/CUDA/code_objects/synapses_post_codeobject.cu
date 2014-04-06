#include "objects.h"
#include "code_objects/synapses_post_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

////// SUPPORT CODE ///////
namespace {
	double _clip(const float value, const float a_min, const float a_max)
	{
	    if (value < a_min)
	        return a_min;
	    if (value > a_max)
	        return a_max;
	    return value;
	}
}

////// HASH DEFINES ///////


void _run_synapses_post_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_synapses_Apre = &_dynamic_array_synapses_Apre[0];
	const int _numApre = _dynamic_array_synapses_Apre.size();
	double* const _array_synapses_lastupdate = &_dynamic_array_synapses_lastupdate[0];
	const int _numlastupdate = _dynamic_array_synapses_lastupdate.size();
	double* const _array_synapses_Apost = &_dynamic_array_synapses_Apost[0];
	const int _numApost = _dynamic_array_synapses_Apost.size();
	double* const _array_synapses_w = &_dynamic_array_synapses_w[0];
	const int _numw = _dynamic_array_synapses_w.size();
	const double t = defaultclock.t_();
	int32_t* const _array_synapses__synaptic_pre = &_dynamic_array_synapses__synaptic_pre[0];
	const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_synapses_Apre = _array_synapses_Apre;
	double * __restrict__ _ptr_array_synapses_lastupdate = _array_synapses_lastupdate;
	double * __restrict__ _ptr_array_synapses_Apost = _array_synapses_Apost;
	double * __restrict__ _ptr_array_synapses_w = _array_synapses_w;
	int32_t * __restrict__ _ptr_array_synapses__synaptic_pre = _array_synapses__synaptic_pre;

    // This is only needed for the _debugmsg function below
	std::vector<int32_t> *_spiking_synapses = synapses_post.queue->peek();
	const unsigned int _num_spiking_synapses = _spiking_synapses->size();
	for(unsigned int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
		const int32_t _idx = (*_spiking_synapses)[_spiking_synapse_idx];
		const int32_t _vectorisation_idx = _idx;
		double Apre = _ptr_array_synapses_Apre[_idx];
		double lastupdate = _ptr_array_synapses_lastupdate[_idx];
		double Apost = _ptr_array_synapses_Apost[_idx];
		double w = _ptr_array_synapses_w[_idx];
		Apre = Apre * exp(-(t - lastupdate) / 0.02);
		Apost = Apost * exp(-(t - lastupdate) / 0.02);
		Apost += -0.000105;
		w = _clip(w + Apre, 0, 0.01);
		lastupdate = t;
		_ptr_array_synapses_Apre[_idx] = Apre;
		_ptr_array_synapses_lastupdate[_idx] = lastupdate;
		_ptr_array_synapses_Apost[_idx] = Apost;
		_ptr_array_synapses_w[_idx] = w;
	}
}

void _debugmsg_synapses_post_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

