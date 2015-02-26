////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}

{# USES_VARIABLES { _spikespace } #}

#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

#define MEM_PER_THREAD (2*sizeof(int32_t) + sizeof(unsigned int))

__global__ void _run_{{codeobj_name}}_advance_kernel()
{
	using namespace brian;
	unsigned int tid = threadIdx.x;

	{{owner.name}}.queue->advance(
		tid);
}

__global__ void _run_{{codeobj_name}}_push_kernel(
	unsigned int sourceN,
	unsigned int _num_blocks,
	unsigned int _num_threads,
	int32_t* {{_spikespace}})
{
	using namespace brian;

	extern __shared__ char shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	for(int i = 0; i < sourceN;)
	{
		int32_t spiking_neuron = {{_spikespace}}[i];
		if(spiking_neuron != -1)
		{
			__syncthreads();
			i++;
			synapses_pre.queue->push(
				bid,
				tid,
				_num_threads,
				spiking_neuron,
				shared_mem);
		}
		else
		{
			//round to nearest multiple of N/num_blocks = start of next block
			i += sourceN/_num_blocks - i % (sourceN/_num_blocks);
		}
	}
}

void _run_{{codeobj_name}}()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////
    {{pointers_lines|autoindent}}

	_run_{{codeobj_name}}_advance_kernel<<<1, num_cuda_processors>>>();

	unsigned int num_threads = max_shared_mem_size / MEM_PER_THREAD;
	num_threads = num_threads < max_threads_per_block? num_threads : max_threads_per_block;	// get min of both
	_run_{{codeobj_name}}_push_kernel<<<num_cuda_processors, num_threads, num_threads*MEM_PER_THREAD>>>(
		_num_spikespace - 1,
		num_cuda_processors,
		num_threads,
		{{_spikespace}});

}
{% endmacro %}

////////////////////////////////////////////////////////////////////////////
//// HEADER FILE ///////////////////////////////////////////////////////////

{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

#endif
{% endmacro %}
