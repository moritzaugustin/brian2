#include "objects.h"
#include "code_objects/feedforward_synapses_create_codeobject.h"
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



void _run_feedforward_synapses_create_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numN_incoming = 2500;
const int _numbarrel_x_pre = 1936;
const int _numbarrel_y_pre = 1936;
const int _num_all_post = 2500;
int32_t* const _array_feedforward__synaptic_post = &_dynamic_array_feedforward__synaptic_post[0];
const int _num_synaptic_post = _dynamic_array_feedforward__synaptic_post.size();
const int _numN_outgoing = 1936;
const int _num_all_pre = 1936;
int32_t* const _array_feedforward__synaptic_pre = &_dynamic_array_feedforward__synaptic_pre[0];
const int _num_synaptic_pre = _dynamic_array_feedforward__synaptic_pre.size();
const int _numbarrel_idx_post = 3076;
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_feedforward_N_incoming = _array_feedforward_N_incoming;
 int32_t * __restrict _ptr_array_layer4_barrel_x = _array_layer4_barrel_x;
 int32_t * __restrict _ptr_array_layer4_barrel_y = _array_layer4_barrel_y;
 int32_t * __restrict _ptr_array_layer23_subgroup__sub_idx = _array_layer23_subgroup__sub_idx;
 int32_t * __restrict _ptr_array_feedforward__synaptic_post = _array_feedforward__synaptic_post;
 int32_t * __restrict _ptr_array_feedforward_N_outgoing = _array_feedforward_N_outgoing;
 int32_t * __restrict _ptr_array_layer4_i = _array_layer4_i;
 int32_t * __restrict _ptr_array_feedforward__synaptic_pre = _array_feedforward__synaptic_pre;
 int32_t * __restrict _ptr_array_layer23_barrel_idx = _array_layer23_barrel_idx;


    #include<iostream>
    
    // scalar code
    const int _vectorisation_idx = -1;
 	
 const int32_t _n = 1;
 const double _p = 0.5;

	
    for(int _i=0; _i<_num_all_pre; _i++)
	{
		for(int _j=0; _j<_num_all_post; _j++)
		{
		    const int _vectorisation_idx = _j;
   			
   const int32_t _all_post = _ptr_array_layer23_subgroup__sub_idx[_j];
   const int32_t _all_pre = _ptr_array_layer4_i[_i];
   const int32_t barrel_y_pre = _ptr_array_layer4_barrel_y[_all_pre];
   const int32_t barrel_x_pre = _ptr_array_layer4_barrel_x[_all_pre];
   const int32_t barrel_idx_post = _ptr_array_layer23_barrel_idx[_all_post];
   const int32_t _pre_idx = _all_pre;
   const int32_t _post_idx = _all_post;
   const char _cond = (barrel_x_pre + (2 * barrel_y_pre)) == barrel_idx_post;

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
			        _ptr_array_feedforward_N_outgoing[_pre_idx] += 1;
			        _ptr_array_feedforward_N_incoming[_post_idx] += 1;
			    	_dynamic_array_feedforward__synaptic_pre.push_back(_pre_idx);
			    	_dynamic_array_feedforward__synaptic_post.push_back(_post_idx);
                }
			}
		}
	}

	// now we need to resize all registered variables
	const int32_t newsize = _dynamic_array_feedforward__synaptic_pre.size();
	_dynamic_array_feedforward__synaptic_post.resize(newsize);
	_dynamic_array_feedforward__synaptic_pre.resize(newsize);
	_dynamic_array_feedforward_A_source.resize(newsize);
	_dynamic_array_feedforward_A_target.resize(newsize);
	_dynamic_array_feedforward_pre_delay.resize(newsize);
	_dynamic_array_feedforward_post_delay.resize(newsize);
	_dynamic_array_feedforward_lastupdate.resize(newsize);
	_dynamic_array_feedforward_w.resize(newsize);
	// Also update the total number of synapses
	feedforward._N_value = newsize;
}


