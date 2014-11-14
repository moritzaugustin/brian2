#include "objects.h"
#include "code_objects/recurrent_exc_synapses_create_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>
#include <stdint.h>
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace {
 	
 double _rand(int vectorisation_idx)
 {
     return (double)rand()/RAND_MAX;
 }

}

////// HASH DEFINES ///////



void _run_recurrent_exc_synapses_create_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numy_pre = 3076;
const int _numN_incoming = 3076;
int32_t* const _array_recurrent_exc__synaptic_post = &_dynamic_array_recurrent_exc__synaptic_post[0];
const int _num_synaptic_post = _dynamic_array_recurrent_exc__synaptic_post.size();
int32_t* const _array_recurrent_exc__synaptic_pre = &_dynamic_array_recurrent_exc__synaptic_pre[0];
const int _num_synaptic_pre = _dynamic_array_recurrent_exc__synaptic_pre.size();
const int _numN_outgoing = 2500;
const int _numy_post = 3076;
const int _numx_pre = 3076;
const int _num_all_pre = 2500;
const int _num_all_post = 3076;
const int _numj = 3076;
const int _numx_post = 3076;
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_layer23_y = _array_layer23_y;
 int32_t * __restrict _ptr_array_recurrent_exc_N_incoming = _array_recurrent_exc_N_incoming;
 int32_t * __restrict _ptr_array_recurrent_exc__synaptic_post = _array_recurrent_exc__synaptic_post;
 int32_t * __restrict _ptr_array_recurrent_exc__synaptic_pre = _array_recurrent_exc__synaptic_pre;
 int32_t * __restrict _ptr_array_recurrent_exc_N_outgoing = _array_recurrent_exc_N_outgoing;
 double * __restrict _ptr_array_layer23_x = _array_layer23_x;
 int32_t * __restrict _ptr_array_layer23_subgroup__sub_idx = _array_layer23_subgroup__sub_idx;
 int32_t * __restrict _ptr_array_layer23_i = _array_layer23_i;


    #include<iostream>
    
    // scalar code
    const int _vectorisation_idx = -1;
 	
 const int32_t _n = 1;

	
    for(int _i=0; _i<_num_all_pre; _i++)
	{
		for(int _j=0; _j<_num_all_post; _j++)
		{
		    const int _vectorisation_idx = _j;
   			
   const int32_t _all_post = _ptr_array_layer23_i[_j];
   const int32_t _all_pre = _ptr_array_layer23_subgroup__sub_idx[_i];
   const double y_pre = _ptr_array_layer23_y[_all_pre];
   const double y_post = _ptr_array_layer23_y[_all_post];
   const double x_pre = _ptr_array_layer23_x[_all_pre];
   const int32_t j = _ptr_array_layer23_i[_all_post];
   const double x_post = _ptr_array_layer23_x[_all_post];
   const int32_t _pre_idx = _all_pre;
   const int32_t _post_idx = _all_post;
   const char _cond = j < (4 * 625);
   const double _p = 0.15 * exp((-0.5) * ((pow((x_pre - x_post) / 0.4, 2)) + (pow((y_pre - y_post) / 0.4, 2))));

			// Add to buffer
			if(_cond)
			{
			    if (_p != 1.0) {
			        // We have to use _rand instead of rand to use our rand
			        // function, not the one from the C standard library
			        if (_rand(_vectorisation_idx) >= _p)
			            continue;
			    }
			    for (int _repetition=0; _repetition<_n; _repetition++) {
			        _ptr_array_recurrent_exc_N_outgoing[_pre_idx] += 1;
			        _ptr_array_recurrent_exc_N_incoming[_post_idx] += 1;
			    	_dynamic_array_recurrent_exc__synaptic_pre.push_back(_pre_idx);
			    	_dynamic_array_recurrent_exc__synaptic_post.push_back(_post_idx);
                }
			}
		}
	}

	// now we need to resize all registered variables
	const int32_t newsize = _dynamic_array_recurrent_exc__synaptic_pre.size();
	_dynamic_array_recurrent_exc__synaptic_post.resize(newsize);
	_dynamic_array_recurrent_exc__synaptic_pre.resize(newsize);
	_dynamic_array_recurrent_exc_pre_delay.resize(newsize);
	_dynamic_array_recurrent_exc_lastupdate.resize(newsize);
	_dynamic_array_recurrent_exc_w.resize(newsize);
	// Also update the total number of synapses
	recurrent_exc._N_value = newsize;
}


