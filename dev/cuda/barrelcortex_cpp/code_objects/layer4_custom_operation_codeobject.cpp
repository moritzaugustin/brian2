#include "objects.h"
#include "code_objects/layer4_custom_operation_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

////// SUPPORT CODE ///////
namespace {
 	
 double _rand(int vectorisation_idx)
 {
     return (double)rand()/RAND_MAX;
 }

}

////// HASH DEFINES ///////



void _run_layer4_custom_operation_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numstim_start_time = 1;
const int _numdirection = 1;
const int _numstim_start_y = 1;
const int _numstim_start_x = 1;
const double t = defaultclock.t_();
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_layer4_stim_start_time = _array_layer4_stim_start_time;
 double * __restrict _ptr_array_layer4_direction = _array_layer4_direction;
 double * __restrict _ptr_array_layer4_stim_start_y = _array_layer4_stim_start_y;
 double * __restrict _ptr_array_layer4_stim_start_x = _array_layer4_stim_start_x;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 double direction = _ptr_array_layer4_direction[0];
 double stim_start_time;
 double stim_start_y;
 double stim_start_x;
 direction = (_rand(_vectorisation_idx) * 2) * 3.141592653589793;
 stim_start_x = (2 / 2.0) - (cos(direction) * 6.0);
 stim_start_y = (2 / 2.0) - (sin(direction) * 6.0);
 stim_start_time = t;
 _ptr_array_layer4_stim_start_time[0] = stim_start_time;
 _ptr_array_layer4_direction[0] = direction;
 _ptr_array_layer4_stim_start_y[0] = stim_start_y;
 _ptr_array_layer4_stim_start_x[0] = stim_start_x;

	 
	for(int _idx=0; _idx<1936; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
                

	}
}


