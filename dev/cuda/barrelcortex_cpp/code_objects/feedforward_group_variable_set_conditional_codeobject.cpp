#include "objects.h"
#include "code_objects/feedforward_group_variable_set_conditional_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_feedforward_group_variable_set_conditional_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_feedforward_w = &_dynamic_array_feedforward_w[0];
const int _numw = _dynamic_array_feedforward_w.size();
const int64_t N = feedforward._N();
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_feedforward_w = _array_feedforward_w;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const char _cond = true;

 	


	//We add the parallel flag because this is executed outside the main run loop
	
	for(int _idx=0; _idx<N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		

		if (_cond)
		{
                        
            double w;
            w = 3.73837195 * 0.001;
            _ptr_array_feedforward_w[_idx] = w;

        }
	}
}


