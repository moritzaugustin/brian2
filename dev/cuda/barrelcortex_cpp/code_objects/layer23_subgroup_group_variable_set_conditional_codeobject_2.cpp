#include "objects.h"
#include "code_objects/layer23_subgroup_group_variable_set_conditional_codeobject_2.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_subgroup_group_variable_set_conditional_codeobject_2()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _num_sub_idx = 2500;
const int _numbarrel_idx = 3076;
const int _numy = 3076;
const int _numx = 3076;
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_layer23_subgroup__sub_idx = _array_layer23_subgroup__sub_idx;
 int32_t * __restrict _ptr_array_layer23_barrel_idx = _array_layer23_barrel_idx;
 double * __restrict _ptr_array_layer23_y = _array_layer23_y;
 double * __restrict _ptr_array_layer23_x = _array_layer23_x;


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
            const double y = _ptr_array_layer23_y[_sub_idx];
            const double x = _ptr_array_layer23_x[_sub_idx];
            int32_t barrel_idx;
            barrel_idx = floor(x) + (floor(y) * 2);
            _ptr_array_layer23_barrel_idx[_sub_idx] = barrel_idx;

        }
	}
}


