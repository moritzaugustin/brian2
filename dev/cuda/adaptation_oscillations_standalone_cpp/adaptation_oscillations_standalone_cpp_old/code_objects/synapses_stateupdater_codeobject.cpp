#include "objects.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////


void _run_synapses_stateupdater_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int64_t N = synapses._N();
	///// POINTERS ////////////
	

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<N; _idx++)
	{
		const int _vectorisation_idx = _idx;
			
	}
}


