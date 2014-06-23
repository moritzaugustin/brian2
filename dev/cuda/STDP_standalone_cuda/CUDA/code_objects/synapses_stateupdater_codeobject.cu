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

__global__ void _run_synapses_stateupdater_codeobject_kernel()
{
	//DO NOTHING
}

////// HASH DEFINES ///////


void _run_synapses_stateupdater_codeobject()
{
	using namespace brian;

	//// MAIN CODE ////////////
	//_run_synapses_stateupdater_codeobject_kernel<<<1,1>>>();
}


