#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

#define neuron_N 4000

__global__ void _run_synapses_pre_pre_codeobject_kernel(int par_num_threads, int par_num_pre)
{
	using namespace brian;

	int bid = blockIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	synapses_pre.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	int num_threads = par_num_threads;
	int num_pre = par_num_pre;
	float num_per_thread = (float)num_pre/(float)num_threads;
	int lower = bid;
	int upper = (bid + 1);

	for(int i = 0; i < synapses_pre.queue->num_parallel; i++)
	{
		for(int j = 0; j < pre_neuron_queue[i].size(); j++)
		{
			int32_t pre_idx = pre_neuron_queue[i].get(j);
			if(pre_idx >= lower && pre_idx < upper)
			{
				//DO NOTHING
			}
		}
	}
}

__global__ void _run_synapses_pre_syn_codeobject_kernel(int par_num_threads, int par_num_syn, double par_t, double* par_array_synapses_lastupdate)
{
	using namespace brian;
	
	int bid = blockIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	synapses_pre.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	float num_per_thread = (float)num_syn/(float)num_threads; 
	int lower = bid;
	int upper = (bid + 1);

	double* _array_synapses_lastupdate = par_array_synapses_lastupdate;
	double t = par_t;

	for(int i = 0; i < synapses_pre.queue->num_parallel; i++)
	{
		for(int j = 0; j < synapses_queue[i].size(); j++)
		{
			int32_t syn_idx = synapses_queue[i].get(j);
			if(syn_idx >= lower && syn_idx < upper)
			{
				double lastupdate;
				lastupdate = t;
				_array_synapses_lastupdate[syn_idx] = lastupdate;
			}
		}
	}
}

__global__ void _run_synapses_pre_post_codeobject_kernel(int par_num_threads, int par_num_post, double* par_array_synapses_c, bool* par_array_neurongroup_not_refractory, double* par_array_neurongroup_v)
{
	using namespace brian;
	
	int bid = blockIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	synapses_pre.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	int num_threads = par_num_threads;
	int num_post = par_num_post;
	float num_per_thread = (float)num_post/(float)num_threads;
	int lower = bid;
	int upper = (bid + 1);

	double* array_synapses_c = par_array_synapses_c;
	bool* array_neurongroup_not_refractory = par_array_neurongroup_not_refractory;
	double* array_neurongroup_v = par_array_neurongroup_v;

	//we are only working on part of the queue
	for(int i = lower; i < upper; i++)
	{
		for(int j = 0; j < post_neuron_queue[i].size(); j++)
		{
			int syn_idx = synapses_queue[i].get(j);
			int32_t post_idx = post_neuron_queue[i].get(j);
			const double c = array_synapses_c[syn_idx];
			const bool not_refractory = array_neurongroup_not_refractory[post_idx];
			double v = array_neurongroup_v[post_idx];
			if(not_refractory)
			{
				//v += c;
			}
			array_neurongroup_v[post_idx] = v;
		}
	}
}

void _run_synapses_pre_codeobject()
{
	using namespace brian;

	double t = defaultclock.t_();
	int syn_N = synapses._N();
	double* dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);
	double* dev_array_synapses_c = thrust::raw_pointer_cast(&_dynamic_array_synapses_c[0]);

	//_run_synapses_pre_pre_codeobject_kernel<<<num_blocks_sequential,1>>>(num_blocks_sequential, neuron_N);

	_run_synapses_pre_syn_codeobject_kernel<<<num_blocks_sequential,1>>>(num_blocks_sequential, syn_N, t, dev_array_synapses_lastupdate);

	_run_synapses_pre_post_codeobject_kernel<<<num_blocks_sequential,1>>>(num_blocks_sequential, neuron_N, dev_array_synapses_c, dev_array_neurongroup_not_refractory, dev_array_neurongroup_v);
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

