#include "objects.h"
#include "code_objects/ratemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_ratemonitor_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const double _clock_t = defaultclock.t_();
const int _num_spikespace = 4001;
double* const _array_ratemonitor_rate = &_dynamic_array_ratemonitor_rate[0];
const int _numrate = _dynamic_array_ratemonitor_rate.size();
const double _clock_dt = defaultclock.dt_();
double* const _array_ratemonitor_t = &_dynamic_array_ratemonitor_t[0];
const int _numt = _dynamic_array_ratemonitor_t.size();
	///// POINTERS ////////////
 	
 int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
 double * __restrict__ _ptr_array_ratemonitor_rate = _array_ratemonitor_rate;
 double * __restrict__ _ptr_array_ratemonitor_t = _array_ratemonitor_t;



	int _num_spikes = _ptr_array_neurongroup__spikespace[_num_spikespace-1];
	// For subgroups, we do not want to record all spikes
    // We assume that spikes are ordered
    int _start_idx = 0;
    int _end_idx = - 1;
    for(int _j=0; _j<_num_spikes; _j++)
    {
        const int _idx = _ptr_array_neurongroup__spikespace[_j];
        if (_idx >= 0) {
            _start_idx = _j;
            break;
        }
    }
    for(int _j=_start_idx; _j<_num_spikes; _j++)
    {
        const int _idx = _ptr_array_neurongroup__spikespace[_j];
        if (_idx >= 4000) {
            _end_idx = _j;
            break;
        }
    }
    if (_end_idx == -1)
        _end_idx =_num_spikes;
    _num_spikes = _end_idx - _start_idx;
	_dynamic_array_ratemonitor_rate.push_back(1.0*_num_spikes/_clock_dt/4000);
	_dynamic_array_ratemonitor_t.push_back(_clock_t);
}


