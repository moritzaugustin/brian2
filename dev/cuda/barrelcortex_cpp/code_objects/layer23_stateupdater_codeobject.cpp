#include "objects.h"
#include "code_objects/layer23_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_stateupdater_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numge = 3076;
const int _numvt = 3076;
const double t = defaultclock.t_();
const int _numv = 3076;
const double dt = defaultclock.dt_();
const int _numlastspike = 3076;
const int _numgi = 3076;
const int _numnot_refractory = 3076;
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_layer23_ge = _array_layer23_ge;
 double * __restrict _ptr_array_layer23_vt = _array_layer23_vt;
 double * __restrict _ptr_array_layer23_v = _array_layer23_v;
 double * __restrict _ptr_array_layer23_lastspike = _array_layer23_lastspike;
 double * __restrict _ptr_array_layer23_gi = _array_layer23_gi;
 char * __restrict _ptr_array_layer23_not_refractory = _array_layer23_not_refractory;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	

	 
	for(int _idx=0; _idx<3076; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
                
        double ge = _ptr_array_layer23_ge[_idx];
        const double lastspike = _ptr_array_layer23_lastspike[_idx];
        double vt = _ptr_array_layer23_vt[_idx];
        double gi = _ptr_array_layer23_gi[_idx];
        double v = _ptr_array_layer23_v[_idx];
        char not_refractory;
        not_refractory = (t - lastspike) > 0.002;
        const double _ge = (ge * ((- dt) + 0.002)) / 0.002;
        const double _vt = ((dt * ((-0.055) - vt)) + (0.05 * vt)) / 0.05;
        const double _v = ((dt * ((((-0.07) + ge) + gi) - v)) + (0.01 * v)) / 0.01;
        const double _gi = (gi * ((- dt) + 0.025)) / 0.025;
        ge = _ge;
        vt = _vt;
        v = _v;
        gi = _gi;
        _ptr_array_layer23_ge[_idx] = ge;
        _ptr_array_layer23_not_refractory[_idx] = not_refractory;
        _ptr_array_layer23_vt[_idx] = vt;
        _ptr_array_layer23_gi[_idx] = gi;
        _ptr_array_layer23_v[_idx] = v;

	}
}


