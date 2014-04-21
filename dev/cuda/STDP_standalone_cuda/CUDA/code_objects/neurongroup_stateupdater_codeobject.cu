#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	
}

#define N 1

////// HASH DEFINES ///////

__global__ void _run_neurongroup_stateupdater_codeobject_kernel(int par_numge,
	int par_numv, double par_dt, double* par_array_neurongroup_ge,
	double* par_array_neurongroup_v)
{
	int tid = threadIdx.x;
	//const int _numge = par_numge;
	//const int _numv = par_numv;
	const double dt = par_dt;
	double *  _ptr_array_neurongroup_ge = par_array_neurongroup_ge;
	double *  _ptr_array_neurongroup_v = par_array_neurongroup_v;

	double ge = _ptr_array_neurongroup_ge[tid];
	double v = _ptr_array_neurongroup_v[tid];
	const double _ge = ge * exp(-(dt) / 0.005);
	const double _BA_v = -(0.0) * ge - -0.074 + ge * -0.06;
	const double _v = -(_BA_v) + (_BA_v + v) * exp(-(dt) / 0.01);
	ge = _ge;
	v = _v;
	_ptr_array_neurongroup_ge[tid] = ge;
	_ptr_array_neurongroup_v[tid] = v;
}

void _run_neurongroup_stateupdater_codeobject()
{
	using namespace brian;

	const double dt = defaultclock.dt_();
	const int _numge = 1;
	const int _numv = 1;

	_run_neurongroup_stateupdater_codeobject_kernel<<<1,N>>>(_numge, _numv,
		dt, dev_array_neurongroup_ge, dev_array_neurongroup_v);
}


