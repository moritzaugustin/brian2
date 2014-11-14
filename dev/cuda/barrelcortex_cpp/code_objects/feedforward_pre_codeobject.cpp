#include "objects.h"
#include "code_objects/feedforward_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>
#include <stdint.h>
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace {
 	
 double _clip(const float value, const float a_min, const float a_max)
 {
     if (value < a_min)
         return a_min;
     if (value > a_max)
         return a_max;
     return value;
 }

}

////// HASH DEFINES ///////



void _run_feedforward_pre_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_feedforward_lastupdate = &_dynamic_array_feedforward_lastupdate[0];
const int _numlastupdate = _dynamic_array_feedforward_lastupdate.size();
int32_t* const _array_feedforward__synaptic_post = &_dynamic_array_feedforward__synaptic_post[0];
const int _num_postsynaptic_idx = _dynamic_array_feedforward__synaptic_post.size();
int32_t* const _array_feedforward__synaptic_pre = &_dynamic_array_feedforward__synaptic_pre[0];
const int _num_synaptic_pre = _dynamic_array_feedforward__synaptic_pre.size();
const int _numge = 3076;
double* const _array_feedforward_A_source = &_dynamic_array_feedforward_A_source[0];
const int _numA_source = _dynamic_array_feedforward_A_source.size();
const double t = defaultclock.t_();
double* const _array_feedforward_w = &_dynamic_array_feedforward_w[0];
const int _numw = _dynamic_array_feedforward_w.size();
double* const _array_feedforward_A_target = &_dynamic_array_feedforward_A_target[0];
const int _numA_target = _dynamic_array_feedforward_A_target.size();
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_feedforward_lastupdate = _array_feedforward_lastupdate;
 int32_t * __restrict _ptr_array_feedforward__synaptic_post = _array_feedforward__synaptic_post;
 int32_t * __restrict _ptr_array_feedforward__synaptic_pre = _array_feedforward__synaptic_pre;
 double * __restrict _ptr_array_layer23_ge = _array_layer23_ge;
 double * __restrict _ptr_array_feedforward_A_source = _array_feedforward_A_source;
 double * __restrict _ptr_array_feedforward_w = _array_feedforward_w;
 double * __restrict _ptr_array_feedforward_A_target = _array_feedforward_A_target;



	// This is only needed for the _debugmsg function below	
	
	
	// scalar code
	const int _vectorisation_idx = -1;
 	

	
	std::vector<int> *_spiking_synapses = feedforward_pre.peek();
	const unsigned int _num_spiking_synapses = _spiking_synapses->size();

	
	{
		for(unsigned int _spiking_synapse_idx=0;
			_spiking_synapse_idx<_num_spiking_synapses;
			_spiking_synapse_idx++)
		{
			const int _idx = (*_spiking_synapses)[_spiking_synapse_idx];
			const int _vectorisation_idx = _idx;
   			
   const int32_t _postsynaptic_idx = _ptr_array_feedforward__synaptic_post[_idx];
   double lastupdate = _ptr_array_feedforward_lastupdate[_idx];
   double ge = _ptr_array_layer23_ge[_postsynaptic_idx];
   double A_source = _ptr_array_feedforward_A_source[_idx];
   double w = _ptr_array_feedforward_w[_idx];
   double A_target = _ptr_array_feedforward_A_target[_idx];
   A_source = A_source * exp((- (t - lastupdate)) / 0.005);
   A_target = A_target * exp((- (t - lastupdate)) / 0.025);
   ge += w;
   A_source += 0.00037383719530530515;
   w = _clip(w + A_target, 0, 0.007476743906106103);
   lastupdate = t;
   _ptr_array_feedforward_lastupdate[_idx] = lastupdate;
   _ptr_array_layer23_ge[_postsynaptic_idx] = ge;
   _ptr_array_feedforward_A_source[_idx] = A_source;
   _ptr_array_feedforward_w[_idx] = w;
   _ptr_array_feedforward_A_target[_idx] = A_target;

		}
	}

}

void _debugmsg_feedforward_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_feedforward__synaptic_pre.size() << endl;
}

