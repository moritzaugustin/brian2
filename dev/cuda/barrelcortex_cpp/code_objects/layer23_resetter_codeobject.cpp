#include "objects.h"
#include "code_objects/layer23_resetter_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_resetter_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_spikespace = 3077;
const int _numvt = 3076;
const int _numv = 3076;
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_layer23__spikespace = _array_layer23__spikespace;
 double * __restrict _ptr_array_layer23_vt = _array_layer23_vt;
 double * __restrict _ptr_array_layer23_v = _array_layer23_v;



	const int32_t *_spikes = _ptr_array_layer23__spikespace;
	const int32_t _num_spikes = _ptr_array_layer23__spikespace[3076];

	//// MAIN CODE ////////////	
	// scalar code
	const int _vectorisation_idx = -1;
 	

    
	
	for(int _index_spikes=0; _index_spikes<_num_spikes; _index_spikes++)
	{
	    // vector code
		const int _idx = _spikes[_index_spikes];
		const int _vectorisation_idx = _idx;
                
        double vt = _ptr_array_layer23_vt[_idx];
        double v;
        v = (-0.07);
        vt += 0.002;
        _ptr_array_layer23_vt[_idx] = vt;
        _ptr_array_layer23_v[_idx] = v;

	}
}


