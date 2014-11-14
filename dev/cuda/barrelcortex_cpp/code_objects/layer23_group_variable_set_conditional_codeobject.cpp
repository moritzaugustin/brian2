#include "objects.h"
#include "code_objects/layer23_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_layer23_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numv = 3076;
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_layer23_v = _array_layer23_v;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const char _cond = true;

 	


	//We add the parallel flag because this is executed outside the main run loop
	
	for(int _idx=0; _idx<3076; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		

		if (_cond)
		{
                        
            double v;
            v = (-70.0) * 0.001;
            _ptr_array_layer23_v[_idx] = v;

        }
	}
}


