#include "objects.h"
#include "code_objects/layer4_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer4_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numbarrel_x = 1936;
const int _numi = 1936;
	///// POINTERS ////////////
 	
 int32_t * __restrict _ptr_array_layer4_barrel_x = _array_layer4_barrel_x;
 int32_t * __restrict _ptr_array_layer4_i = _array_layer4_i;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const char _cond = true;

 	


	//We add the parallel flag because this is executed outside the main run loop
	
	for(int _idx=0; _idx<1936; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		

		if (_cond)
		{
                        
            const int32_t i = _ptr_array_layer4_i[_idx];
            int32_t barrel_x;
            barrel_x = (fmod(i / 484, 2)) + 0.5;
            _ptr_array_layer4_barrel_x[_idx] = barrel_x;

        }
	}
}


