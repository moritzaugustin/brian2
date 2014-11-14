#include "objects.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject_1.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_subgroup_group_variable_set_conditional_codeobject_1()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numy = 3076;
const int _numi = 3076;
const int _num_sub_idx = 2500;
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_layer23_y = _array_layer23_y;
 int32_t * __restrict _ptr_array_layer23_i = _array_layer23_i;
 int32_t * __restrict _ptr_array_layer23_subgroup__sub_idx = _array_layer23_subgroup__sub_idx;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const char _cond = true;

 	


	//We add the parallel flag because this is executed outside the main run loop
	
	for(int _idx=0; _idx<2500; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		

		if (_cond)
		{
                        
            const int32_t _sub_idx = _ptr_array_layer23_subgroup__sub_idx[_idx];
            const int32_t i = _ptr_array_layer23_i[_sub_idx];
            double y;
            y = (i / (2 * 25)) * (1.0 / 25);
            _ptr_array_layer23_y[_sub_idx] = y;

        }
	}
}


