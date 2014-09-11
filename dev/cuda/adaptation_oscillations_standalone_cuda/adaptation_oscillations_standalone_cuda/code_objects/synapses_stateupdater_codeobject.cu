#include "objects.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#define THREADS 1024
#define BLOCKS(N) (N + THREADS -1)/THREADS

/*
__global__ void _run_synapses_stateupdater_codeobject_kernel()
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	//DO NOTHING IN THIS KERNEL
}
*/

void _run_synapses_stateupdater_codeobject()
{
	using namespace brian;

	const int64_t N = synapses._N();

	//_run_synapses_group_variable_set_conditional_codeobject_kernel<<<BLOCKS(N), THREADS>>>();
}

