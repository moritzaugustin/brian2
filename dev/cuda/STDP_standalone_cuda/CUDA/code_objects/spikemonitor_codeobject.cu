#include "objects.h"
#include "code_objects/spikemonitor_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

////// HASH DEFINES ///////

#define N 1000

__global__ void _run_spikemonitor_codeobject_kernel(double par_clock_t, int32_t* par_array_poissongroup__spikespace)
{
	using namespace brian;

	int _num__array_poissongroup__spikespace = 1000;
	int32_t* _array_poissongroup__spikespace = par_array_poissongroup__spikespace;
	double clock_t = par_clock_t;

	for(int i = 0; i < _num__array_poissongroup__spikespace - 1; i++)
	{
		if(_array_poissongroup__spikespace[i] != -1)
		{
			_dynamic_array_spikemonitor_i->push(i);
			_dynamic_array_spikemonitor_t->push(clock_t);
		}
	}
}

void _run_spikemonitor_codeobject()
{
	using namespace brian;
	const double _clock_t = defaultclock.t_();

	_run_spikemonitor_codeobject_kernel<<<1,1>>>(_clock_t, dev_array_poissongroup__spikespace);
}

__global__ void _debugmsg_spikemonitor_codeobject_kernel()
{
	printf("Number of spikes: %d\n", brian::_dynamic_array_spikemonitor_i->size());
}


void _debugmsg_spikemonitor_codeobject()
{
	using namespace brian;
	_debugmsg_spikemonitor_codeobject_kernel<<<1,1>>>();
}

