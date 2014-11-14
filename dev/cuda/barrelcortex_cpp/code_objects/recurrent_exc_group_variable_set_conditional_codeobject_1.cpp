#include "objects.h"
#include "code_objects/recurrent_exc_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_recurrent_exc_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numj = 3076;
const int64_t N = recurrent_exc._N();
double* const _array_recurrent_exc_w = &_dynamic_array_recurrent_exc_w[0];
const int _numw = _dynamic_array_recurrent_exc_w.size();
int32_t* const _array_recurrent_exc__synaptic_post = &_dynamic_array_recurrent_exc__synaptic_post[0];
const int _num_postsynaptic_idx = _dynamic_array_recurrent_exc__synaptic_post.size();
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_layer23_i = _array_layer23_i;
 double * __restrict _ptr_array_recurrent_exc_w = _array_recurrent_exc_w;
 int32_t * __restrict _ptr_array_recurrent_exc__synaptic_post = _array_recurrent_exc__synaptic_post;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	

 	


	//We add the parallel flag because this is executed outside the main run loop
	
	for(int _idx=0; _idx<N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		
  const int32_t _postsynaptic_idx = _ptr_array_recurrent_exc__synaptic_post[_idx];
  const int32_t j = _ptr_array_layer23_i[_postsynaptic_idx];
  const char _cond = j >= (4 * 625);

		if (_cond)
		{
                        
            double w;
            w = 7.47674391 * 0.001;
            _ptr_array_recurrent_exc_w[_idx] = w;

        }
	}
}


