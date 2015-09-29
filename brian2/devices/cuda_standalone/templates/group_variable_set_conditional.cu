{% macro cu_file() %}
#include "code_objects/{{codeobj_name}}.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

{% block extra_headers %}
{% endblock %}

////// SUPPORT CODE ///////
namespace {
	int num_blocks(int objects){
		return ceil(objects / (double)brian::max_threads_per_block);
	}
	int num_threads(int objects){
		return brian::max_threads_per_block;
	}
	{{support_code_lines|autoindent}}
}

__global__ void _kernel_{{codeobj_name}}(
	unsigned int _N,
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	{# USES_VARIABLES { N } #}
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;

	%KERNEL_VARIABLES%

	if(_idx < 0 || _idx >= _N)
	{
		return;
	}

	{{scalar_code['condition']|autoindent}}
	{{scalar_code['statement']|autoindent}}

	{{vector_code['condition']|autoindent}}
	if (_cond)
	{
        {{vector_code['statement']|autoindent}}
    }
}

////// HASH DEFINES ///////
{{hashdefine_lines|autoindent}}

void _run_{{codeobj_name}}()
{
    {# USES_VARIABLES { N } #}
    {# ALLOWS_SCALAR_WRITE #}
	using namespace brian;
	///// CONSTANTS ///////////
	const int _N = {{constant_or_scalar('N', variables['N'])}};
	%CONSTANTS%

	_kernel_{{codeobj_name}}<<<num_blocks(_N),num_threads(_N)>>>(
		_N,
		num_threads(_N),
		%HOST_PARAMETERS%
	);
}

{% block extra_functions_cu %}
{% endblock %}

{% endmacro %}


{% macro h_file() %}
#ifndef _INCLUDED_{{codeobj_name}}
#define _INCLUDED_{{codeobj_name}}

#include "objects.h"

void _run_{{codeobj_name}}();

{% block extra_functions_h %}
{% endblock %}

#endif
{% endmacro %}



