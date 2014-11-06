#include "objects.h"
#include "code_objects/synapses_synapses_create_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

#define neuron_N 4000
#define max_syn_N (neuron_N * neuron_N)

void _run_synapses_synapses_create_codeobject()
{
	using namespace brian;

	//generate neuron_N * neuron_N random numbers
	float* _array_random_float_numbers;
	_array_random_float_numbers = (float*)malloc(sizeof(float)*max_syn_N);
	curandGenerator_t gen;
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, _array_random_float_numbers, max_syn_N);

	//these two vectors just cache everything on the CPU-side
	//data is copied to GPU at the end
	thrust::host_vector<int32_t> temp_synaptic_post;
	thrust::host_vector<int32_t> temp_synaptic_pre;

	int32_t * _ptr_array_synapses_N_incoming = _array_synapses_N_incoming;
	int32_t * _ptr_array_synapses_N_outgoing = _array_synapses_N_outgoing;
	int32_t * _ptr_array_neurongroup_i = _array_neurongroup_i;

	int syn_id = _dynamic_array_synapses__synaptic_pre.size();
	for(int i = 0; i < neuron_N; i++)
	{
		synapses_by_pre_neuron.push_back(syn_id);
		for(int j = 0; j < neuron_N; j++)
		{
			const int32_t _all_post = _ptr_array_neurongroup_i[j];
			const int32_t _all_pre = _ptr_array_neurongroup_i[i];
			const int32_t _pre_idx = _all_pre;
			const int32_t _post_idx = _all_post;
			const bool _cond = i != j;
			const int32_t _n = 1;
			const double _p = 0.05;
			// Add to buffer
			if(_cond)
			{
				if (_p != 1.0)
				{
					float r = _array_random_float_numbers[i*neuron_N + j];
					if (r >= _p)
					{
						continue;
					}
				}
				for (int _repetition = 0; _repetition < _n; _repetition++)
				{
					_ptr_array_synapses_N_outgoing[_pre_idx] += 1;
					_ptr_array_synapses_N_incoming[_post_idx] += 1;
				    	temp_synaptic_pre.push_back(_pre_idx);
				    	temp_synaptic_post.push_back(_post_idx);
					syn_id++;
				}
			}
		}
	}
	synapses_by_pre_neuron.push_back(syn_id);

	//copy data to GPU
	_dynamic_array_synapses__synaptic_post = temp_synaptic_post;
	_dynamic_array_synapses__synaptic_pre = temp_synaptic_pre;

	// now we need to resize all registered variables
	int newsize = _dynamic_array_synapses__synaptic_pre.size();
	_dynamic_array_synapses__synaptic_post.resize(newsize);
	_dynamic_array_synapses__synaptic_pre.resize(newsize);
	_dynamic_array_synapses_c.resize(newsize);
	_dynamic_array_synapses_pre_delay.resize(newsize);
	_dynamic_array_synapses_lastupdate.resize(newsize);
	//Also update the total number of synapses
	synapses._N_value = newsize;

	curandDestroyGenerator(gen);
	free(_array_random_float_numbers);
}


