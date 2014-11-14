#include "objects.h"
#include "code_objects/synapses_group_variable_set_conditional_codeobject_2.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_synapses_group_variable_set_conditional_codeobject_2()
{
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_synapses_lastupdate = &_dynamic_array_synapses_lastupdate[0];
const int _numlastupdate = _dynamic_array_synapses_lastupdate.size();
const int64_t N = synapses._N();
	///// POINTERS ////////////
 	
 double * __restrict__ _ptr_array_synapses_lastupdate = _array_synapses_lastupdate;


	//// MAIN CODE ////////////
	// scalar code
	const int _vectorisation_idx = -1;
 	
 const bool _cond = true;

 	


	for(int _idx=0; _idx<N; _idx++)
	{
	    // vector code
		const int _vectorisation_idx = _idx;
  		

		if (_cond)
		{
                        
            double lastupdate;
            lastupdate = 0.0 * 1.0;
            _ptr_array_synapses_lastupdate[_idx] = lastupdate;

        }
	}
}


