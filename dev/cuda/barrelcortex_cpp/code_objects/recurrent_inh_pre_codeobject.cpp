#include "objects.h"
#include "code_objects/recurrent_inh_pre_codeobject.h"
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



void _run_recurrent_inh_pre_codeobject()
{	
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_recurrent_inh_lastupdate = &_dynamic_array_recurrent_inh_lastupdate[0];
const int _numlastupdate = _dynamic_array_recurrent_inh_lastupdate.size();
int32_t* const _array_recurrent_inh__synaptic_pre = &_dynamic_array_recurrent_inh__synaptic_pre[0];
const int _num_synaptic_pre = _dynamic_array_recurrent_inh__synaptic_pre.size();
const double t = defaultclock.t_();
int32_t* const _array_recurrent_inh__synaptic_post = &_dynamic_array_recurrent_inh__synaptic_post[0];
const int _num_postsynaptic_idx = _dynamic_array_recurrent_inh__synaptic_post.size();
const int _numgi = 3076;
	///// POINTERS ////////////
 	
 double * __restrict _ptr_array_recurrent_inh_lastupdate = _array_recurrent_inh_lastupdate;
 int32_t * __restrict _ptr_array_recurrent_inh__synaptic_pre = _array_recurrent_inh__synaptic_pre;
 int32_t * __restrict _ptr_array_recurrent_inh__synaptic_post = _array_recurrent_inh__synaptic_post;
 double * __restrict _ptr_array_layer23_gi = _array_layer23_gi;



	// This is only needed for the _debugmsg function below	
	
	
	// scalar code
	const int _vectorisation_idx = -1;
 	

	
	std::vector<int> *_spiking_synapses = recurrent_inh_pre.peek();
	const unsigned int _num_spiking_synapses = _spiking_synapses->size();

	
	{
		for(unsigned int _spiking_synapse_idx=0;
			_spiking_synapse_idx<_num_spiking_synapses;
			_spiking_synapse_idx++)
		{
			const int _idx = (*_spiking_synapses)[_spiking_synapse_idx];
			const int _vectorisation_idx = _idx;
   			
   const int32_t _postsynaptic_idx = _ptr_array_recurrent_inh__synaptic_post[_idx];
   double gi = _ptr_array_layer23_gi[_postsynaptic_idx];
   double lastupdate;
   gi += (-0.0018420157493201931);
   lastupdate = t;
   _ptr_array_recurrent_inh_lastupdate[_idx] = lastupdate;
   _ptr_array_layer23_gi[_postsynaptic_idx] = gi;

		}
	}

}

void _debugmsg_recurrent_inh_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_recurrent_inh__synaptic_pre.size() << endl;
}

