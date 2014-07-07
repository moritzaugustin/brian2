#include "objects.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


__global__ void _run_synapses_stateupdater_codeobject_kernel()
{
	//DO NOTHING
}

void _run_synapses_stateupdater_codeobject()
{
	using namespace brian;
	const int64_t N = synapses._N();

	//// MAIN CODE ////////////
	//_run_synapses_stateupdater_codeobject_kernel<<<1,N>>>();
}

