#include "objects.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>


////// SUPPORT CODE ///////
namespace {
	inline double _ranf()
	{
	    return (double)rand()/RAND_MAX;
	}
	double _randn(const int vectorisation_idx)
	{
	     double x1, x2, w;
	     static double y1, y2;
	     static bool need_values = true;
	     if (need_values)
	     {
	         do {
	                 x1 = 2.0 * _ranf() - 1.0;
	                 x2 = 2.0 * _ranf() - 1.0;
	                 w = x1 * x1 + x2 * x2;
	         } while ( w >= 1.0 );
	         w = sqrt( (-2.0 * log( w ) ) / w );
	         y1 = x1 * w;
	         y2 = x2 * w;
	         need_values = false;
	         return y1;
	     } else
	     {
	        need_values = true;
	        return y2;
	     }
	}
	int int_(const bool value)
	{
	    return value ? 1 : 0;
	}
}

////// HASH DEFINES ///////


void _run_neurongroup_stateupdater_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const double t = defaultclock.t_();
const int _numw = 4000;
const int _numv = 4000;
const double dt = defaultclock.dt_();
const int _numlastspike = 4000;
const int _numnot_refractory = 4000;
	///// POINTERS ////////////
	double * __restrict__ _ptr_array_neurongroup_w = _array_neurongroup_w;
	double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;
	double * __restrict__ _ptr_array_neurongroup_lastspike = _array_neurongroup_lastspike;
	bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;

	//// MAIN CODE ////////////
	for(int _idx=0; _idx<4000; _idx++)
	{
		const int _vectorisation_idx = _idx;
			double w = _ptr_array_neurongroup_w[_idx];
			double v = _ptr_array_neurongroup_v[_idx];
			const double lastspike = _ptr_array_neurongroup_lastspike[_idx];
			bool not_refractory = _ptr_array_neurongroup_not_refractory[_idx];
			not_refractory = t - lastspike > 0.0025;
			const double xi = pow(dt, 0.5) * _randn(_vectorisation_idx);
			const double _w = -(dt) * w * int_(not_refractory) / 0.2 + w;
			const double _v = dt * (0.14 * int_(not_refractory) - v * int_(not_refractory) / 0.01 - w * int_(not_refractory) / 0.01) + v + 0.002213594362117866 * xi * int_(not_refractory);
			if(not_refractory)
			    w = _w;
			if(not_refractory)
			    v = _v;
			_ptr_array_neurongroup_not_refractory[_idx] = not_refractory;
			_ptr_array_neurongroup_w[_idx] = w;
			_ptr_array_neurongroup_v[_idx] = v;
	}
}


