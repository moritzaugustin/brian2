#include "objects.h"
#include "code_objects/synapses_synapses_create_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

#define N 4000
#define N_squared (N*N)
#define ceil(M, num) ((M + num-1)/num)

void _run_synapses_synapses_create_codeobject()
{
	using namespace brian;

	float* _array_random;
	_array_random = (float*)malloc(sizeof(float)*N_squared);
	curandGenerator_t gen;
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, _array_random, N_squared);

	///// CONSTANTS ///////////
	const int _num_all_post = N;
	const int _num_synaptic_post = _dynamic_array_synapses__synaptic_post.size();
	const int _num_all_pre = N;
	const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();
	///// POINTERS ////////////
	int32_t * __restrict__ _ptr_array_synapses_N_incoming = _array_synapses_N_incoming;
	int32_t * __restrict__ _ptr_array_synapses_N_outgoing = _array_synapses_N_outgoing;
	int32_t * __restrict__ _ptr_array_neurongroup_i = _array_neurongroup_i;

	std::vector<int32_t> temp_pos;
	std::vector<int32_t> temp_pre;
	std::vector<int32_t> temp_post;

	int _synapse_idx = _dynamic_array_synapses__synaptic_pre.size();
	for(int i=0; i<_num_all_pre; i++)
	{
		temp_pos.push_back(_synapse_idx);
		for(int j=0; j<_num_all_post; j++)
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
					// We have to use _rand instead of rand to use our rand
					// function, not the one from the C standard library
					if (_array_random[i*_num_all_post + j] >= _p)
					    continue;
				}

				for (int _repetition=0; _repetition<_n; _repetition++)
				{
					_ptr_array_synapses_N_outgoing[_pre_idx] += 1;
					_ptr_array_synapses_N_incoming[_post_idx] += 1;
					temp_pre.push_back(_pre_idx);
					temp_post.push_back(_post_idx);
					_synapse_idx++;
				}
			}
		}
	}
	temp_pos.push_back(_synapse_idx);

	_dynamic_array_synapses__pos = temp_pos;
	_dynamic_array_synapses__synaptic_post = temp_post;
	_dynamic_array_synapses__synaptic_pre = temp_pre;

	for(int i = 0; i < N; i++)
	{
		int num_syn = _ptr_array_synapses_N_outgoing[i];
		int num_per_block = ceil(num_syn, num_blocks_sequential);
		int cur_block = 0;
		for(int j = 0; j < _ptr_array_synapses_N_outgoing[i]; j += num_per_block)
		{
			_array_neurongroup_queue_bounds[i*num_blocks_sequential + cur_block] = temp_pre[temp_pos[i] + j];
			cur_block++;
		}
		_array_neurongroup_queue_bounds[i*num_blocks_sequential + num_blocks_sequential - 1] = temp_pre[temp_pos[i + 1] - 1];
	}
	cudaMemcpy(dev_array_neurongroup_queue_bounds, _array_neurongroup_queue_bounds, sizeof(int32_t)*N*(num_blocks_sequential + 1), cudaMemcpyHostToDevice);
	
	// now we need to resize all registered variables
	const int newsize = _dynamic_array_synapses__synaptic_pre.size();
	_dynamic_array_synapses__synaptic_post.resize(newsize);
	_dynamic_array_synapses__synaptic_pre.resize(newsize);
	_dynamic_array_synapses_c.resize(newsize);
	_dynamic_array_synapses_pre_delay.resize(newsize);
	_dynamic_array_synapses_lastupdate.resize(newsize);
	// Also update the total number of synapses
	synapses._N_value = newsize;

	curandDestroyGenerator(gen);
	free(_array_random);
}


