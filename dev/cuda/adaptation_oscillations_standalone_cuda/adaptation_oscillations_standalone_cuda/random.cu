#include "objects.h"
#include "random.h"

#include <curand.h>

#define neuron_N 4000

void _random_number_generation()
{
	using namespace brian;

	float mean = 0.0;
	float std_deviation = 1.0;
	curandGenerateNormal(random_float_generator, dev_array_random_floats, neuron_N, mean, std_deviation);
}
