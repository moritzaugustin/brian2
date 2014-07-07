#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_spikemonitor_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_count = 4000;
const int _num_spikespace = 4001;
double* const _array_spikemonitor_t = &_dynamic_array_spikemonitor_t[0];
const int _numt = _dynamic_array_spikemonitor_t.size();
int32_t* const _array_spikemonitor_i = &_dynamic_array_spikemonitor_i[0];
const int _numi = _dynamic_array_spikemonitor_i.size();
const double _clock_t = defaultclock.t_();
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_spikemonitor__count = _array_spikemonitor__count;
	int32_t * __restrict__ _ptr_array_neurongroup__spikespace = _array_neurongroup__spikespace;
	double * __restrict__ _ptr_array_spikemonitor_t = _array_spikemonitor_t;
	int32_t * __restrict__ _ptr_array_spikemonitor_i = _array_spikemonitor_i;

	//// MAIN CODE ////////////
	int _num_spikes = _ptr_array_neurongroup__spikespace[_num_spikespace-1];
    if (_num_spikes > 0)
    {
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
        if (_num_spikes > 0) {
        	for(int _j=_start_idx; _j<_end_idx; _j++)
        	{
        		const int _idx = _ptr_array_neurongroup__spikespace[_j];
        		_dynamic_array_spikemonitor_i.push_back(_idx-0);
        		_dynamic_array_spikemonitor_t.push_back(_clock_t);
        	}
        }
    }
}

void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;
	std::cout << "Number of spikes: " << _dynamic_array_spikemonitor_i.size() << endl;
}

