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

__global__ void _run_synapses_post_codeobject_kernel(double* par_array_synapses_Apre,
	int par_numApre, double* par_array_synapses_lastupdate, int par_numlastupdate,
	double* par_array_synapses_Apost, int par_numApost, double* par_array_synapses_w,
	int par_numw, int32_t* par_array_synapses__synaptic_pre,
	int par_num_synaptic_pre, double par_t, int32_t* par_spiking_synapses,
	int par_num_spiking_synapses)
{
	int tid = threadIdx.x;
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
	int32_t * _spiking_synapses = par_spiking_synapses;
	//int num_spiking_synapses = par_num_spiking_synapses;


	const int32_t _idx = _spiking_synapses[tid];
	double Apre = _ptr_array_synapses_Apre[_idx];
	double lastupdate = _ptr_array_synapses_lastupdate[_idx];
	double Apost = _ptr_array_synapses_Apost[_idx];
	double w = _ptr_array_synapses_w[_idx];
	Apre = Apre * exp(-(t - lastupdate) / 0.02);
	Apost = Apost * exp(-(t - lastupdate) / 0.02);
	Apost += -0.000105;
	w = _clip(w + Apre, 0, 0.01);
	lastupdate = t;
	_ptr_array_synapses_Apre[_idx] = Apre;
	_ptr_array_synapses_lastupdate[_idx] = lastupdate;
	_ptr_array_synapses_Apost[_idx] = Apost;
	_ptr_array_synapses_w[_idx] = w;
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

	// This is only needed for the _debugmsg function below
	thrust::device_vector<int32_t> _spiking_synapses = synapses_post.queue->peek();
	int32_t* dev_spiking_synapses = thrust::raw_pointer_cast(_spiking_synapses.data());
	const unsigned int _num_spiking_synapses = _spiking_synapses.size();

	double* dev_array_synapses_Apre;
	double* dev_array_synapses_lastupdate;
	double* dev_array_synapses_Apost;
	double* dev_array_synapses_w;
	int32_t* dev_array_synapses__synaptic_pre;

	cudaMalloc((void**)&dev_array_synapses_Apre, sizeof(double)*_numApre);
	cudaMalloc((void**)&dev_array_synapses_lastupdate, sizeof(double)*_numlastupdate);
	cudaMalloc((void**)&dev_array_synapses_Apost, sizeof(double)*_numApost);
	cudaMalloc((void**)&dev_array_synapses_w, sizeof(double)*_numw);
	cudaMalloc((void**)&dev_array_synapses__synaptic_pre, sizeof(int32_t)*_num_synaptic_pre);

	cudaMemcpy(dev_array_synapses_Apre, &_dynamic_array_synapses_Apre[0], sizeof(double)*_numApre, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_synapses_lastupdate, &_dynamic_array_synapses_lastupdate[0], sizeof(double)*_numlastupdate, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_synapses_Apost, &_dynamic_array_synapses_Apost[0], sizeof(double)*_numApost, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_synapses_w, &_dynamic_array_synapses_w[0], sizeof(double)*_numw, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_synapses__synaptic_pre, &_dynamic_array_synapses__synaptic_pre[0], sizeof(int32_t)*_num_synaptic_pre, cudaMemcpyHostToDevice);

	_run_synapses_post_codeobject_kernel<<<1, _num_spiking_synapses>>>(
		dev_array_synapses_Apre, _numApre, dev_array_synapses_lastupdate,
		_numlastupdate, dev_array_synapses_Apost, _numApost, dev_array_synapses_w,
		_numw, dev_array_synapses__synaptic_pre, _num_synaptic_pre, t,
		dev_spiking_synapses, _num_spiking_synapses);

	cudaMemcpy(&_dynamic_array_synapses_Apre[0], dev_array_synapses_Apre, sizeof(double)*_numApre, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_dynamic_array_synapses_lastupdate[0], dev_array_synapses_lastupdate, sizeof(double)*_numlastupdate, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_dynamic_array_synapses_Apost[0], dev_array_synapses_Apost, sizeof(double)*_numApost, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_dynamic_array_synapses_w[0], dev_array_synapses_w, sizeof(double)*_numw, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_dynamic_array_synapses__synaptic_pre[0], dev_array_synapses__synaptic_pre, sizeof(int32_t)*_num_synaptic_pre, cudaMemcpyDeviceToHost);

	cudaFree(dev_array_synapses_Apre);
	cudaFree(dev_array_synapses_lastupdate);
	cudaFree(dev_array_synapses_Apost);
	cudaFree(dev_array_synapses_w);
	cudaFree(dev_array_synapses__synaptic_pre);
}

void _debugmsg_synapses_post_codeobject()
{
	using namespace brian;
	std::cout << "Number of synapses: " << _dynamic_array_synapses__synaptic_pre.size() << endl;
}

