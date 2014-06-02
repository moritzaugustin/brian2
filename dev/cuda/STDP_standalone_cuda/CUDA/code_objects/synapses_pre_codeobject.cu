#include "objects.h"
#include "code_objects/synapses_pre_codeobject.h"
#include<math.h>
#include "brianlib/common_math.h"
#include<stdint.h>
#include<iostream>
#include<fstream>

#include <inttypes.h>
#include "brianlib/synapses.h"

#include <thrust/device_vector.h>

////// SUPPORT CODE ///////
namespace {
	__device__ double _clip(const float value, const float a_min, const float a_max)
	{
	    if (value < a_min)
	        return a_min;
	    if (value > a_max)
	        return a_max;
	    return value;
	}
}

////// HASH DEFINES ///////

__global__ void _run_synapses_pre_pre_codeobject_kernel(int par_num_threads, int par_num_syn)
{
	int bid = blockIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	brian::synapses_pre.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	float num_per_thread = (float)num_syn/(float)num_threads;
	int lower = bid*(num_per_thread);
	int upper = (bid + 1)*(num_per_thread);

	for(int i = 0; i < brian::synapses_pre.queue->num_parallel; i++)
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

__global__ void _run_synapses_pre_syn_codeobject_kernel(int par_num_threads, int par_num_syn, double* par_array_synapses_Apre, double* par_array_synapses_lastupdate, double* par_array_synapses_Apost, double* par_array_synapses_w, double* par_array_neurongroup_ge, int32_t* par_array_synapses__synaptic_pre, int32_t* par_array_synapses__synaptic_post, int par_numApre, int par_numlastupdate, int par_numApost, int par_numw, int par_numge, double par_t, int par_num_postsynaptic_idx, int par_num_synaptic_pre)
{
	int bid = blockIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;
	brian::synapses_pre.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	float num_per_thread = (float)num_syn/(float)num_threads;
	int lower = bid*(num_per_thread);
	int upper = (bid + 1)*(num_per_thread);

	double * _ptr_array_synapses_Apre = par_array_synapses_Apre;
	double * _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;
	double * _ptr_array_synapses_Apost = par_array_synapses_Apost;
	double * _ptr_array_synapses_w = par_array_synapses_w;
	//double * _ptr_array_neurongroup_ge = par_array_neurongroup_ge;
	//int32_t * _ptr_array_synapses__synaptic_post = par_array_synapses__synaptic_post;
	//int32_t * _ptr_array_synapses__synaptic_pre = par_array_synapses__synaptic_pre;
	//const int _numApre = par_numApre;
	//const int _numlastupdate = par_numlastupdate;
	//const int _numApost = par_numApost;
	//const int _numw = par_numw;
	//const int _numge = par_numge;
	double t = par_t;
	//const int _num_postsynaptic_idx = par_num_postsynaptic_idx;
	//const int _num_synaptic_pre = par_num_synaptic_pre;

	//iterate over all queues
	for(int i = 0; i < brian::synapses_pre.queue->num_parallel; i++)
	{
		//and over all elements in each queue
		for(int j = 0; j < synapses_queue[i].size(); j++)
		{
			int32_t syn_idx = synapses_queue[i].get(j);
			//we are only responsible for parts of the work
			if(syn_idx >= lower && syn_idx < upper)
			{
				double Apre = _ptr_array_synapses_Apre[syn_idx];
				double lastupdate = _ptr_array_synapses_lastupdate[syn_idx];
				double Apost = _ptr_array_synapses_Apost[syn_idx];
				double w = _ptr_array_synapses_w[syn_idx];
				Apre = Apre * exp(-(t - lastupdate) / 0.02);
				Apost = Apost * exp(-(t - lastupdate) / 0.02);
				Apre += 0.0001;
				w = _clip(w + Apost, 0, 0.01);
				lastupdate = t;
				_ptr_array_synapses_Apre[syn_idx] = Apre;
				_ptr_array_synapses_lastupdate[syn_idx] = lastupdate;
				_ptr_array_synapses_Apost[syn_idx] = Apost;
				_ptr_array_synapses_w[syn_idx] = w;
			}
		}
	}
}

__global__ void _run_synapses_pre_post_codeobject_kernel(int par_num_threads, int par_num_syn, double* par_array_neurongroup_ge, double* par_array_synapses_w)
{
	int bid = blockIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	brian::synapses_pre.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	float num_per_thread = (float)num_syn/(float)num_threads;
	int lower = bid*(num_per_thread);
	int upper = (bid + 1)*(num_per_thread);

	double * _ptr_array_neurongroup_ge = par_array_neurongroup_ge;
	double * _ptr_array_synapses_w = par_array_synapses_w;


	for(int i = 0; i < brian::synapses_pre.queue->num_parallel; i++)
	{
		for(int j = 0; j < post_neuron_queue[i].size(); j++)
		{
			int32_t post_idx = post_neuron_queue[i].get(j);
			if(post_idx >= lower && post_idx < upper)
			{
				int32_t syn_idx = synapses_queue[i].get(j);
				double ge = _ptr_array_neurongroup_ge[post_idx];
				double w = _ptr_array_synapses_w[syn_idx];
				ge += w;
				_ptr_array_neurongroup_ge[post_idx] = ge;
			}
		}
	}
}

void _run_synapses_pre_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numApre = _dynamic_array_synapses_Apre.size();
	const int _numlastupdate = _dynamic_array_synapses_lastupdate.size();
	const int _numApost = _dynamic_array_synapses_Apost.size();
	const int _numw = _dynamic_array_synapses_w.size();
	const int _numge = 1;
	const double t = defaultclock.t_();
	const int _num_postsynaptic_idx = _dynamic_array_synapses__synaptic_post.size();
	const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();

	int max_num_threads = num_blocks_sequential;

	double* dev_array_synapses_Apre = thrust::raw_pointer_cast(&_dynamic_array_synapses_Apre[0]);
	double* dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);
	double* dev_array_synapses_Apost = thrust::raw_pointer_cast(&_dynamic_array_synapses_Apost[0]);
	double* dev_array_synapses_w = thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0]);
	int32_t* dev_array_synapses__synaptic_pre = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]);
	int32_t* dev_array_synapses__synaptic_post = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post[0]);

	//_run_synapses_pre_pre_codeobject_kernel<<<max_num_threads,1>>>(max_num_threads, 1000);

	_run_synapses_pre_syn_codeobject_kernel<<<max_num_threads, 1>>>(max_num_threads, 1000, dev_array_synapses_Apre,
		dev_array_synapses_lastupdate, dev_array_synapses_Apost, dev_array_synapses_w,
		dev_array_neurongroup_ge, dev_array_synapses__synaptic_pre,
		dev_array_synapses__synaptic_post, _numApre, _numlastupdate, _numApost, _numw,
		_numge, t, _num_postsynaptic_idx, _num_synaptic_pre);

	_run_synapses_pre_post_codeobject_kernel<<<max_num_threads, 1>>>(max_num_threads, 1, dev_array_neurongroup_ge, dev_array_synapses_w);

}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}
