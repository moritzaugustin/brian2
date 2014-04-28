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

__global__ void _run_synapses_pre_codeobject_kernel(double* par_array_synapses_Apre, double* par_array_synapses_lastupdate, double* par_array_synapses_Apost, double* par_array_synapses_w, double* par_array_neurongroup_ge, int32_t* par_array_synapses__synaptic_pre, int32_t* par_array_synapses__synaptic_post, 	int par_numApre, int par_numlastupdate, int par_numApost, int par_numw, int par_numge, double par_t, int par_num_postsynaptic_idx, int par_num_synaptic_pre, int32_t * par_spiking_synapses, int par_num_spiking_synapses)
{
	int tid = threadIdx.x;

	double * _ptr_array_synapses_Apre = par_array_synapses_Apre;
	double * _ptr_array_synapses_lastupdate = par_array_synapses_lastupdate;
	double * _ptr_array_synapses_Apost = par_array_synapses_Apost;
	double * _ptr_array_synapses_w = par_array_synapses_w;
	double * _ptr_array_neurongroup_ge = par_array_neurongroup_ge;
	int32_t * _ptr_array_synapses__synaptic_post = par_array_synapses__synaptic_post;
	//int32_t * _ptr_array_synapses__synaptic_pre = par_array_synapses__synaptic_pre;
	//const int _numApre = par_numApre;
	//const int _numlastupdate = par_numlastupdate;
	//const int _numApost = par_numApost;
	//const int _numw = par_numw;
	//const int _numge = par_numge;
	double t = par_t;
	//const int _num_postsynaptic_idx = par_num_postsynaptic_idx;
	//const int _num_synaptic_pre = par_num_synaptic_pre;
	int32_t * _spiking_synapses = par_spiking_synapses;
	int num_spiking_synapses = par_num_spiking_synapses;

	int32_t _idx = _spiking_synapses[tid];

	int32_t _postsynaptic_idx = _ptr_array_synapses__synaptic_post[_idx];
	double Apre = _ptr_array_synapses_Apre[_idx];
	double lastupdate = _ptr_array_synapses_lastupdate[_idx];
	double Apost = _ptr_array_synapses_Apost[_idx];
	double w = _ptr_array_synapses_w[_idx];
	Apre = Apre * exp(-(t - lastupdate) / 0.02);
	Apost = Apost * exp(-(t - lastupdate) / 0.02);
	Apre += 0.0001;
	w = _clip(w + Apost, 0, 0.01);
	lastupdate = t;
	_ptr_array_synapses_Apre[_idx] = Apre;
	_ptr_array_synapses_lastupdate[_idx] = lastupdate;
	_ptr_array_synapses_Apost[_idx] = Apost;
	_ptr_array_synapses_w[_idx] = w;

	for(int i = 0; i < num_spiking_synapses; i++)
	{
		__syncthreads();
		if(tid == i)
		{
			double ge = _ptr_array_neurongroup_ge[_postsynaptic_idx];
			ge += w;
			_ptr_array_neurongroup_ge[_postsynaptic_idx] = ge;
		}
		__syncthreads();
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

	double* dev_array_synapses_Apre = thrust::raw_pointer_cast(&_dynamic_array_synapses_Apre[0]);
	double* dev_array_synapses_lastupdate = thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate[0]);
	double* dev_array_synapses_Apost = thrust::raw_pointer_cast(&_dynamic_array_synapses_Apost[0]);
	double* dev_array_synapses_w = thrust::raw_pointer_cast(&_dynamic_array_synapses_w[0]);
	int32_t* dev_array_synapses__synaptic_pre;
	int32_t* dev_array_synapses__synaptic_post;

	cudaMalloc((void**)&dev_array_synapses__synaptic_pre, sizeof(int32_t)*_num_synaptic_pre);
	cudaMalloc((void**)&dev_array_synapses__synaptic_post, sizeof(int32_t)*_num_postsynaptic_idx);

	cudaMemcpy(dev_array_synapses__synaptic_pre, &_dynamic_array_synapses__synaptic_pre[0], sizeof(int32_t)*_num_synaptic_pre, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_synapses__synaptic_post, &_dynamic_array_synapses__synaptic_post[0], sizeof(int32_t)*_num_postsynaptic_idx, cudaMemcpyHostToDevice);

    // This is only needed for the _debugmsg function below
	thrust::device_vector<int32_t> _spiking_synapses = synapses_pre.queue->peek();
	int32_t* dev_spiking_synapses = thrust::raw_pointer_cast(_spiking_synapses.data());
	const unsigned int _num_spiking_synapses = _spiking_synapses.size();

	_run_synapses_pre_codeobject_kernel<<<1, _num_spiking_synapses>>>(dev_array_synapses_Apre,
		dev_array_synapses_lastupdate, dev_array_synapses_Apost, dev_array_synapses_w,
		dev_array_neurongroup_ge, dev_array_synapses__synaptic_pre,
		dev_array_synapses__synaptic_post, _numApre, _numlastupdate, _numApost, _numw,
		_numge, t, _num_postsynaptic_idx, _num_synaptic_pre, dev_spiking_synapses,
		_num_spiking_synapses);

	cudaMemcpy(&_dynamic_array_synapses__synaptic_pre[0], dev_array_synapses__synaptic_pre, sizeof(int32_t)*_num_synaptic_pre, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_dynamic_array_synapses__synaptic_post[0], dev_array_synapses__synaptic_post, sizeof(int32_t)*_num_postsynaptic_idx, cudaMemcpyDeviceToHost);

	cudaFree(dev_array_synapses__synaptic_pre);
	cudaFree(dev_array_synapses__synaptic_post);
}

void _debugmsg_synapses_pre_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}
