#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

////// SUPPORT CODE ///////
namespace {
 	

}

////// HASH DEFINES ///////



void _run_synapses_pre_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	double* const _array_synapses_lastupdate = &_dynamic_array_synapses_lastupdate[0];
const int _numlastupdate = _dynamic_array_synapses_lastupdate.size();
double* const _array_synapses_c = &_dynamic_array_synapses_c[0];
const int _numc = _dynamic_array_synapses_c.size();
const double t = defaultclock.t_();
int32_t* const _array_synapses__synaptic_post = &_dynamic_array_synapses__synaptic_post[0];
const int _num_postsynaptic_idx = _dynamic_array_synapses__synaptic_post.size();
const int _numv = 4000;
int32_t* const _array_synapses__synaptic_pre = &_dynamic_array_synapses__synaptic_pre[0];
const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();
const int _numnot_refractory = 4000;
	///// POINTERS ////////////
 	
 double * __restrict__ _ptr_array_synapses_lastupdate = _array_synapses_lastupdate;
 double * __restrict__ _ptr_array_synapses_c = _array_synapses_c;
 int32_t * __restrict__ _ptr_array_synapses__synaptic_post = _array_synapses__synaptic_post;
 double * __restrict__ _ptr_array_neurongroup_v = _array_neurongroup_v;
 int32_t * __restrict__ _ptr_array_synapses__synaptic_pre = _array_synapses__synaptic_pre;
 bool * __restrict__ _ptr_array_neurongroup_not_refractory = _array_neurongroup_not_refractory;


    // This is only needed for the _debugmsg function below
	std::vector<int32_t> *_spiking_synapses = synapses_pre.queue->peek();

	// scalar code
	const int _vectorisation_idx = -1;
 	


	const unsigned int _num_spiking_synapses = _spiking_synapses->size();
	for(unsigned int _spiking_synapse_idx=0;
		_spiking_synapse_idx<_num_spiking_synapses;
		_spiking_synapse_idx++)
	{
	    // vector code
		const int32_t _idx = (*_spiking_synapses)[_spiking_synapse_idx];
		const int32_t _vectorisation_idx = _idx;
  		
  const int32_t _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
  const double c = _ptr_array_synapses_c[_idx];
  const bool not_refractory = _ptr_array_neurongroup_not_refractory[_postsynaptic_idx];
  double v = _ptr_array_neurongroup_v[_postsynaptic_idx];
  double lastupdate;
  if(not_refractory)
      v += c;
  lastupdate = t;
  _ptr_array_synapses_lastupdate[_idx] = lastupdate;
  _ptr_array_neurongroup_v[_postsynaptic_idx] = v;

	}
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

