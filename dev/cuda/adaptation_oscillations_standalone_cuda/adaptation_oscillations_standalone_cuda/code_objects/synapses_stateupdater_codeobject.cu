#include "objects.h"
#include "code_objects/synapses_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

//does nothing in this program, here just to provide a skeleton for this kind of kernel
__global__ void _run_synapses_stateupdater_codeobject_kernel()
{
	//DO NOTHING IN THIS KERNEL
}

void _run_synapses_stateupdater_codeobject()
{
	using namespace brian;

	const int64_t syn_N = synapses._N();

	unsigned int blocks = (syn_N + max_threads_per_block - 1)/max_threads_per_block;	// = ceil(N/num_threads)
	_run_synapses_stateupdater_codeobject_kernel<<<blocks, max_threads_per_block>>>();
}

