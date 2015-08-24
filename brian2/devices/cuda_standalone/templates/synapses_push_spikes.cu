////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{% macro cu_file() %}
{# USES_VARIABLES { _eventspace } #}

#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include<stdint.h>
#include "brianlib/common_math.h"

namespace {
	int _num_blocks(int num_objects)
    {
		static int needed_num_block = -1;
	    if(needed_num_block == -1)
		{
			needed_num_block = brian::num_parallel_blocks;
			while(needed_num_block * brian::max_threads_per_block < num_objects)
			{
				needed_num_block *= 2;
			}
		}
		return needed_num_block;
    }

	int _num_threads(int num_objects)
    {
		static int needed_num_threads = -1;
		if(needed_num_threads == -1)
		{
			int needed_num_block = _num_blocks(num_objects);
			needed_num_threads = min(brian::max_threads_per_block, (int)ceil(num_objects/(double)needed_num_block));
		}
		return needed_num_threads;
	}
}

#define MEM_PER_THREAD (sizeof(int32_t) + sizeof(unsigned int))

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
	unsigned int block_size,
	int32_t* {{_eventspace}})
{
	using namespace brian;

	extern __shared__ char shared_mem[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	//REMINDER: spikespace format: several blocks, each filled from the left with all spikes in this block, -1 ends list
	unsigned int start_index = {{owner.name}}.spikes_start - ({{owner.name}}.spikes_start % block_size);	//find start of last block
	for(int i = 0; i < {{owner.name}}.spikes_stop; i++)
	{
		int32_t spiking_neuron = {{_eventspace}}[i];
		if(spiking_neuron != -1 && spiking_neuron >= {{owner.name}}.spikes_start && spiking_neuron < {{owner.name}}.spikes_stop)
		{
			__syncthreads();
			{{owner.name}}.queue->push(
				bid,
				tid,
				_num_threads,
				spiking_neuron,
				shared_mem);
		}
		if(spiking_neuron == -1)
		{
			return;
		}
	}
}

void _run_{{codeobj_name}}()
{
	using namespace brian;
    ///// CONSTANTS ///////////
	%CONSTANTS%
	///// POINTERS ////////////

	_run_{{codeobj_name}}_advance_kernel<<<1, num_parallel_blocks>>>();

	unsigned int num_threads = max_shared_mem_size / MEM_PER_THREAD;
	num_threads = num_threads < max_threads_per_block? num_threads : max_threads_per_block; // get min of both
	_run_{{codeobj_name}}_push_kernel<<<num_parallel_blocks, num_threads, num_threads*MEM_PER_THREAD>>>(
		_num_eventspace - 1,
		num_parallel_blocks,
		num_threads,
		_num_threads(_num_eventspace - 1),
		{% set _event_space = get_array_name(owner.variables['eventspace'], access_data=False) %}
		dev{{_event_space}});
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
