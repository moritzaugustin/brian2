#include "objects.h"
#include "code_objects/recurrent_exc_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>
#include <stdint.h>
#include "synapses_classes.h"

////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_recurrent_exc_pre_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_recurrent_exc_lastupdate = &_dynamic_array_recurrent_exc_lastupdate[0];
const int _numlastupdate = _dynamic_array_recurrent_exc_lastupdate.size();
int32_t* const _array_recurrent_exc__synaptic_pre = &_dynamic_array_recurrent_exc__synaptic_pre[0];
const int _num_synaptic_pre = _dynamic_array_recurrent_exc__synaptic_pre.size();
const int _numge = 3076;
const double t = defaultclock.t_();
double* const _array_recurrent_exc_w = &_dynamic_array_recurrent_exc_w[0];
const int _numw = _dynamic_array_recurrent_exc_w.size();
int32_t* const _array_recurrent_exc__synaptic_post = &_dynamic_array_recurrent_exc__synaptic_post[0];
const int _num_postsynaptic_idx = _dynamic_array_recurrent_exc__synaptic_post.size();
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_recurrent_exc_lastupdate = _array_recurrent_exc_lastupdate;
 int32_t * __restrict _ptr_array_recurrent_exc__synaptic_pre = _array_recurrent_exc__synaptic_pre;
 double * __restrict _ptr_array_layer23_ge = _array_layer23_ge;
 double * __restrict _ptr_array_recurrent_exc_w = _array_recurrent_exc_w;
 int32_t * __restrict _ptr_array_recurrent_exc__synaptic_post = _array_recurrent_exc__synaptic_post;



	// This is only needed for the _debugmsg function below	
	
	
	// scalar code
	const int _vectorisation_idx = -1;
 	

	
	std::vector<int> *_spiking_synapses = recurrent_exc_pre.peek();
	const unsigned int _num_spiking_synapses = _spiking_synapses->size();

	
	{
		for(unsigned int _spiking_synapse_idx=0;
			_spiking_synapse_idx<_num_spiking_synapses;
			_spiking_synapse_idx++)
		{
			const int _idx = (*_spiking_synapses)[_spiking_synapse_idx];
			const int _vectorisation_idx = _idx;
   			
   const int32_t _postsynaptic_idx = _ptr_array_recurrent_exc__synaptic_post[_idx];
   double ge = _ptr_array_layer23_ge[_postsynaptic_idx];
   const double w = _ptr_array_recurrent_exc_w[_idx];
   double lastupdate;
   ge += w;
   lastupdate = t;
   _ptr_array_recurrent_exc_lastupdate[_idx] = lastupdate;
   _ptr_array_layer23_ge[_postsynaptic_idx] = ge;

		}
	}

}

void _debugmsg_recurrent_exc_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_recurrent_exc__synaptic_pre.size() << endl;
}

