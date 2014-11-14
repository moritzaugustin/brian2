#include "objects.h"
#include "code_objects/layer4_thresholder_codeobject.h"
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
 double _clip(const float value, const float a_min, const float a_max)
 {
     if (value < a_min)
         return a_min;
     if (value > a_max)
         return a_max;
     return value;
 }
 int int_(const bool value)
 {
     return value ? 1 : 0;
 }

}

////// HASH DEFINES ///////



void _run_layer4_thresholder_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numstim_start_x = 1;
const int _numselectivity = 1936;
const int _numstim_start_time = 1;
const int _numstim_start_y = 1;
const int _numbarrel_y = 1936;
const int _numbarrel_x = 1936;
const int _numdirection = 1;
const int _num_spikespace = 1937;
const double dt = defaultclock.dt_();
const double t = defaultclock.t_();
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_layer4_stim_start_x = _array_layer4_stim_start_x;
 double * __restrict _ptr_array_layer4_selectivity = _array_layer4_selectivity;
 double * __restrict _ptr_array_layer4_stim_start_time = _array_layer4_stim_start_time;
 double * __restrict _ptr_array_layer4_stim_start_y = _array_layer4_stim_start_y;
 int32_t * __restrict _ptr_array_layer4_barrel_y = _array_layer4_barrel_y;
 int32_t * __restrict _ptr_array_layer4_barrel_x = _array_layer4_barrel_x;
 double * __restrict _ptr_array_layer4_direction = _array_layer4_direction;
 int32_t * __restrict _ptr_array_layer4__spikespace = _array_layer4__spikespace;


	// not_refractory and lastspike are added as needed_variables in the
	// Thresholder class, we cannot use the USES_VARIABLE mechanism
	// conditionally

	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const double stim_start_time = _ptr_array_layer4_stim_start_time[0];
 const double direction = _ptr_array_layer4_direction[0];
 const double stim_start_y = _ptr_array_layer4_stim_start_y[0];
 const double stim_start_x = _ptr_array_layer4_stim_start_x[0];
 const double bar_y = ((sin(direction) * (t - stim_start_time)) / (5 * 0.001)) + stim_start_y;
 const double bar_x = ((cos(direction) * (t - stim_start_time)) / (5 * 0.001)) + stim_start_x;


	
	
    {
        long _count = 0;
        for(int _idx=0; _idx<1936; _idx++)
        {
            const int _vectorisation_idx = _idx;
                        
            const double selectivity = _ptr_array_layer4_selectivity[_idx];
            const int32_t barrel_y = _ptr_array_layer4_barrel_y[_idx];
            const double direction = _ptr_array_layer4_direction[0];
            const int32_t barrel_x = _ptr_array_layer4_barrel_x[_idx];
            const char is_active = fabs((((barrel_x + 0.5) - bar_x) * cos(direction)) + (((barrel_y + 0.5) - bar_y) * sin(direction))) < 0.5;
            const double rate = (int_(is_active) * _clip(cos(direction - selectivity), 0, INFINITY)) * 100.0;
            const double _cond = _rand(_vectorisation_idx) < (rate * dt);

            if(_cond) {
                _ptr_array_layer4__spikespace[_count++] = _idx;
            }
        }
        _ptr_array_layer4__spikespace[1936] = _count;
    }
}


