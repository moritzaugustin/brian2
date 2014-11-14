#include "objects.h"
#include "code_objects/layer23_subgroup_1_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_subgroup_1_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_sub_idx = 576;
const int _numx = 3076;
const int _num_source_i = 3076;
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_layer23_subgroup_1__sub_idx = _array_layer23_subgroup_1__sub_idx;
 double * __restrict _ptr_array_layer23_x = _array_layer23_x;
 int32_t * __restrict _ptr_array_layer23_i = _array_layer23_i;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const char _cond = true;

 	


	//We add the parallel flag because this is executed outside the main run loop
	
	for(int _idx=0; _idx<576; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		

		if (_cond)
		{
                        
            const int32_t _sub_idx = _ptr_array_layer23_subgroup_1__sub_idx[_idx];
            const int32_t _source_i = _ptr_array_layer23_i[_sub_idx];
            double x;
            const int32_t i = _source_i - 2500;
            x = (fmod(i, 2 * 12)) * (1.0 / 12);
            _ptr_array_layer23_x[_sub_idx] = x;

        }
	}
}


