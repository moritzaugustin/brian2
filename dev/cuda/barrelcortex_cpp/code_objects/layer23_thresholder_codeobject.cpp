#include "objects.h"
#include "code_objects/layer23_thresholder_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_thresholder_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_spikespace = 3077;
const int _numvt = 3076;
const double t = defaultclock.t_();
const int _numv = 3076;
const int _numlastspike = 3076;
const int _numnot_refractory = 3076;
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_layer23__spikespace = _array_layer23__spikespace;
 double * __restrict _ptr_array_layer23_vt = _array_layer23_vt;
 double * __restrict _ptr_array_layer23_v = _array_layer23_v;
 double * __restrict _ptr_array_layer23_lastspike = _array_layer23_lastspike;
 char * __restrict _ptr_array_layer23_not_refractory = _array_layer23_not_refractory;


	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	


	
	
    {
        long _count = 0;
        for(int _idx=0; _idx<3076; _idx++)
        {
            const int _vectorisation_idx = _idx;
                        
            const char not_refractory = _ptr_array_layer23_not_refractory[_idx];
            const double vt = _ptr_array_layer23_vt[_idx];
            const double v = _ptr_array_layer23_v[_idx];
            const double _cond = (v > vt) && not_refractory;

            if(_cond) {
                _ptr_array_layer23__spikespace[_count++] = _idx;
                // We have to use the pointer names directly here: The condition
                // might contain references to not_refractory or lastspike and in
                // that case the names will refer to a single entry.
                _ptr_array_layer23_not_refractory[_idx] = false;
                _ptr_array_layer23_lastspike[_idx] = t;
            }
        }
        _ptr_array_layer23__spikespace[3076] = _count;
    }
}


