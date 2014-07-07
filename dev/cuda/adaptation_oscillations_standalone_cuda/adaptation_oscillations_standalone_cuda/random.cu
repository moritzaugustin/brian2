#include "objects.h"
#include "random.h"
#include <curand.h>

#define N 4000

void _get_random()
{
	using namespace brian;
	curandGenerateNormal(gen, dev_array_random, N, 0.0, 1.0);
}
