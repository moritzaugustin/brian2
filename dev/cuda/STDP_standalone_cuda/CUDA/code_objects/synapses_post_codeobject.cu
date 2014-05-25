#include "objects.h"
#include "code_objects/synapses_post_codeobject.h"
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


__global__ void _run_synapses_post_pre_codeobject_kernel(int par_num_threads, int par_num_syn)
{
	int tid = threadIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	int lower = tid*(num_syn/num_threads);
	int upper = (tid + 1)*(num_syn/num_threads);

	brian::synapses_post.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	for(int i = 0; i < brian::synapses_post.queue->num_parallel; i++)
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

__global__ void _run_synapses_post_syn_codeobject_kernel(int par_num_threads, int par_num_syn, double* par_array_synapses_Apre, int par_numApre,
	double* par_array_synapses_lastupdate, int par_numlastupdate,
	double* par_array_synapses_Apost, int par_numApost, double* par_array_synapses_w,
	int par_numw, int32_t* par_array_synapses__synaptic_pre, int par_num_synaptic_pre,
	double par_t)
{
	int tid = threadIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	int lower = tid*(num_syn/num_threads);
	int upper = (tid + 1)*(num_syn/num_threads);

	brian::synapses_post.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	double * _ptr_array_synapses_Apre = par_array_synapses_Apre;
	//int _numApre = par_numApre;
	double * _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;
	//int _numlastupdate = par_numlastupdate;
	double * _ptr_array_synapses_Apost = par_array_synapses_Apost;
	//int _numApost = par_numApost;
	double * _ptr_array_synapses_w = par_array_synapses_w;
	//int _numw = par_numw;
	//int32_t * _ptr_array_synapses__synaptic_pre = par_array_synapses__synaptic_pre;
	//int _num_synaptic_pre = par_num_synaptic_pre;
	double t = par_t;

	//iterate over all queues
	for(int i = 0; i < brian::synapses_post.queue->num_parallel; i++)
	{
		//and over all elements in each queue
		for(int j = 0; j < synapses_queue[i].size(); j++)
		{
			int32_t syn_idx = synapses_queue[i].get(j);
			//we are only responsible for parts of the work
			if(syn_idx >= lower && syn_idx < upper)
			{
				const int32_t syn_idx = synapses_queue[i].get(j);
				double Apre = _ptr_array_synapses_Apre[syn_idx];
				double lastupdate = _ptr_array_synapses_lastupdate[syn_idx];
				double Apost = _ptr_array_synapses_Apost[syn_idx];
				double w = _ptr_array_synapses_w[syn_idx];
				Apre = Apre * exp(-(t - lastupdate) / 0.02);
				Apost = Apost * exp(-(t - lastupdate) / 0.02);
				Apost += -0.000105;
				w = _clip(w + Apre, 0, 0.01);
				lastupdate = t;
				_ptr_array_synapses_Apre[syn_idx] = Apre;
				_ptr_array_synapses_lastupdate[syn_idx] = lastupdate;
				_ptr_array_synapses_Apost[syn_idx] = Apost;
				_ptr_array_synapses_w[syn_idx] = w;
			}
		}
	}
}

__global__ void _run_synapses_post_post_codeobject_kernel(int par_num_threads, int par_num_syn)
{
	int tid = threadIdx.x;

	CudaVector<int32_t>* synapses_queue;
	CudaVector<int32_t>* pre_neuron_queue;
	CudaVector<int32_t>* post_neuron_queue;

	int num_threads = par_num_threads;
	int num_syn = par_num_syn;
	int lower = tid*(num_syn/num_threads);
	int upper = (tid + 1)*(num_syn/num_threads);

	brian::synapses_post.queue->peek(&synapses_queue, &pre_neuron_queue, &post_neuron_queue);

	for(int i = 0; i < brian::synapses_post.queue->num_parallel; i++)
	{
		for(int j = 0; j < post_neuron_queue[i].size(); j++)
		{
			int32_t post_idx = post_neuron_queue[i].get(j);
			if(post_idx >= lower && post_idx < upper)
			{
				//DO NOTHING
			}
		}
	}
}

void _run_synapses_post_codeobject()
{
	using namespace brian;
	///// CONSTANTS ///////////
	const int _numApre = _dynamic_array_synapses_Apre.size();
	const int _numlastupdate = _dynamic_array_synapses_lastupdate.size();
	const int _numApost = _dynamic_array_synapses_Apost.size();
	const int _numw = _dynamic_array_synapses_w.size();
	const int _num_synaptic_pre = _dynamic_array_synapses__synaptic_pre.size();
	const double t = defaultclock.t_();

	int max_num_threads = num_blocks_sequential;

	// This is only needed for the _debugmsg function below
	double* dev_array_synapses_Apre = thrust::raw_pointer_cast(&_dynamic_array_synapses_Apre[0]);
	double* dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);
	double* dev_array_synapses_Apost = thrust::raw_pointer_cast(&_dynamic_array_synapses_Apost[0]);
	double* dev_array_synapses_w = thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0]);
	int32_t* dev_array_synapses__synaptic_pre = thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre[0]);

	_run_synapses_post_pre_codeobject_kernel<<<max_num_threads, 1>>>(max_num_threads, 1000);

	_run_synapses_post_syn_codeobject_kernel<<<max_num_threads, 1>>>(max_num_threads, 1, dev_array_synapses_Apre,
		_numApre, dev_array_synapses_lastupdate, _numlastupdate, dev_array_synapses_Apost,
		_numApost, dev_array_synapses_w, _numw, dev_array_synapses__synaptic_pre,
		_num_synaptic_pre, t);

	_run_synapses_post_post_codeobject_kernel<<<max_num_threads,1>>>(max_num_threads, 1);
}

void _debugmsg_synapses_post_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

