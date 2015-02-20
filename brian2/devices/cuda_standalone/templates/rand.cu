{% macro cu_file() %}

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
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
	curandGenerateNormal(random_float_generator, dev_array_random_normal_floats, num_random_normal_numbers, mean, std_deviation);
	curandGenerateUniform(random_float_generator, dev_array_random_uniform_floats, num_random_uniform_numbers);
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
