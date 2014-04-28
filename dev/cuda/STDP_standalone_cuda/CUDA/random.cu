#include "objects.h"
#include "random.h"
#include <curand.h>

#define N 1000

void _get_rands()
{
	using namespace brian;
	curandGenerateUniform(gen, dev_array_rands, N);
}
