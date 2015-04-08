{% macro cu_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include<iostream>
#include<fstream>
#include <curand.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void _run_random_number_generation()
{
	using namespace brian;

	float mean = 0.0;
	float std_deviation = 1.0;

	{% for co in codeobj_with_rand %}
	curandGenerateUniform(random_float_generator, dev_{{co.name}}_random_uniform_floats, {{co.owner._N}} * {{co.rand_calls}});
	{% endfor %}
	{% for co in codeobj_with_randn %}
	curandGenerateNormal(random_float_generator, dev_{{co.name}}_random_normal_floats, {{co.owner._N}} * {{co.randn_calls}}, mean, std_deviation);
	{% endfor %}
}
{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_RAND_H
#define _BRIAN_RAND_H

#include <curand.h>

void _run_random_number_generation();

#endif


{% endmacro %}
