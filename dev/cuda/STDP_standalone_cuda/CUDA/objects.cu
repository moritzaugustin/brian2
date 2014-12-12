#include<stdint.h>
#include<vector>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include "objects.h"
#include "brianlib/synapses.h"
#include "brianlib/spikequeue.h"
#include "brianlib/CudaVector.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"
#include<iostream>
#include<fstream>

int brian::num_blocks_sequential;
int brian::num_blocks_parallel;

//////////////// clocks ///////////////////
Clock brian::defaultclock(0.0001);

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 2;
int32_t * brian::dev_array_neurongroup__spikespace;

double * brian::_array_neurongroup_ge;
const int brian::_num__array_neurongroup_ge = 1;
double * brian::dev_array_neurongroup_ge;

int32_t * brian::_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 1;
//wird nicht weiter verwendet

double * brian::_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 1;
double * brian::dev_array_neurongroup_v;

int32_t * brian::_array_poissongroup__spikespace;
const int brian::_num__array_poissongroup__spikespace = 1001;
int32_t * brian::dev_array_poissongroup__spikespace;

int32_t * brian::_array_poissongroup_i;
const int brian::_num__array_poissongroup_i = 1000;
//wird nicht weiter verwendet

double * brian::_array_poissongroup_rates;
const int brian::_num__array_poissongroup_rates = 1000;
double * brian::dev_array_poissongroup_rates;

int32_t * brian::_array_spikemonitor__count;
const int brian::_num__array_spikemonitor__count = 1000;
//wird nicht weiter verwendet

int32_t * brian::_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 2;

double * brian::_array_statemonitor__recorded_w;
const int brian::_num__array_statemonitor__recorded_w = 2;

int32_t * brian::_array_synapses_N_incoming;
const int brian::_num__array_synapses_N_incoming = 1;
//wird nicht weiter verwendet

int32_t * brian::_array_synapses_N_outgoing;
const int brian::_num__array_synapses_N_outgoing = 1000;
//wird nicht weiter verwendet

//////////////// dynamic arrays 1d /////////
thrust::device_vector<double> brian::_dynamic_array_ratemonitor_rate;
thrust::device_vector<double> brian::_dynamic_array_ratemonitor_t;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor__recorded_w;
thrust::device_vector<double> brian::_dynamic_array_statemonitor_t;
thrust::device_vector<double> brian::_dynamic_array_synapses_Apost;
thrust::device_vector<double> brian::_dynamic_array_synapses_Apre;
thrust::device_vector<double> brian::_dynamic_array_synapses_lastupdate;
thrust::device_vector<double> brian::_dynamic_array_synapses_w;
thrust::device_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<double> brian::_dynamic_array_synapses_post_delay;
thrust::device_vector<double> brian::_dynamic_array_synapses_pre_delay;

__device__ CudaVector<int32_t>** brian::_dynamic_array_spikemonitor_i;
__device__ CudaVector<double>** brian::_dynamic_array_spikemonitor_t;

/////////////// static arrays /////////////
int32_t * brian::_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 2;

//////////////// synapses /////////////////
// synapses
Synapses<double> brian::synapses(1000, 1);
__device__ SynapticPathway<double> brian::synapses_post;
__device__ SynapticPathway<double> brian::synapses_pre;

///////////// for random numbers ///////////
float* brian::dev_array_rands;
curandGenerator_t brian::gen;

__global__ void init_kernel(int num_mps)
{
	int tid = threadIdx.x;

	if(tid == 0)
	{
		brian::synapses_post.init(
			1,
			1, 1000,
			0.0001,
			0, 1
			);
		brian::synapses_pre.init(
			num_mps,
			1000, 1,
			0.0001,
			0, 1000
			);

		brian::_dynamic_array_spikemonitor_i = new CudaVector<int32_t>*[num_mps];
		brian::_dynamic_array_spikemonitor_t = new CudaVector<double>*[num_mps];
	}
	__syncthreads();
	brian::_dynamic_array_spikemonitor_i[tid] = new CudaVector<int32_t>;
	brian::_dynamic_array_spikemonitor_t[tid] = new CudaVector<double>;
}

void _init_arrays()
{
	using namespace brian;
	cudaMalloc((void**)&dev_array_rands, sizeof(float)*1000);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	num_blocks_sequential = props.multiProcessorCount * 4;

	init_kernel<<<1,num_blocks_sequential>>>(num_blocks_sequential);

    // Arrays initialized to 0
	_array_spikemonitor__count = new int32_t[1000];
	for(int i=0; i<1000; i++) _array_spikemonitor__count[i] = 0;

	_array_statemonitor__indices = new int32_t[2];
	for(int i=0; i<2; i++) _array_statemonitor__indices[i] = 0;

	_array_poissongroup__spikespace = new int32_t[1001];
	for(int i=0; i<1001; i++) _array_poissongroup__spikespace[i] = 0;
	cudaMalloc((void**)&dev_array_poissongroup__spikespace, sizeof(int32_t)*_num__array_poissongroup__spikespace);
	cudaMemcpy(dev_array_poissongroup__spikespace, _array_poissongroup__spikespace, sizeof(int32_t)*_num__array_poissongroup__spikespace, cudaMemcpyHostToDevice);

	_array_neurongroup__spikespace = new int32_t[2];
	for(int i=0; i<2; i++) _array_neurongroup__spikespace[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace);
	cudaMemcpy(dev_array_neurongroup__spikespace, _array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);

	_array_neurongroup_ge = new double[1];
	for(int i=0; i<1; i++) _array_neurongroup_ge[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_ge, sizeof(double)*_num__array_neurongroup_ge);
	cudaMemcpy(dev_array_neurongroup_ge, _array_neurongroup_ge, sizeof(double)*_num__array_neurongroup_ge, cudaMemcpyHostToDevice);

	_array_poissongroup_i = new int32_t[1000];
	for(int i=0; i<1000; i++) _array_poissongroup_i[i] = 0;

	_array_neurongroup_i = new int32_t[1];
	for(int i=0; i<1; i++) _array_neurongroup_i[i] = 0;

	_array_synapses_N_incoming = new int32_t[1];
	for(int i=0; i<1; i++) _array_synapses_N_incoming[i] = 0;

	_array_synapses_N_outgoing = new int32_t[1000];
	for(int i=0; i<1000; i++) _array_synapses_N_outgoing[i] = 0;

	_array_poissongroup_rates = new double[1000];
	for(int i=0; i<1000; i++) _array_poissongroup_rates[i] = 0;
	cudaMalloc((void**)&dev_array_poissongroup_rates, sizeof(double)*_num__array_poissongroup_rates);
	cudaMemcpy(dev_array_poissongroup_rates, _array_poissongroup_rates, sizeof(double)*_num__array_poissongroup_rates, cudaMemcpyHostToDevice);

	_array_neurongroup_v = new double[1];
	for(int i=0; i<1; i++) _array_neurongroup_v[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v);
	cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice);

	// Arrays initialized to an "arange"
	_array_poissongroup_i = new int32_t[1000];
	for(int i=0; i<1000; i++) _array_poissongroup_i[i] = 0 + i;
	_array_neurongroup_i = new int32_t[1];
	for(int i=0; i<1; i++) _array_neurongroup_i[i] = 0 + i;

	// static arrays
	_static_array__array_statemonitor__indices = new int32_t[2];

	_dynamic_array_statemonitor__recorded_w = new thrust::device_vector<double>[_num__array_statemonitor__recorded_w];
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_statemonitor__indices;
	f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor__indices.is_open())
	{
		f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 2*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_ge, dev_array_neurongroup_ge, sizeof(double)*_num__array_neurongroup_ge, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_poissongroup__spikespace, dev_array_poissongroup__spikespace, sizeof(int32_t)*_num__array_poissongroup__spikespace, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_poissongroup_rates, dev_array_poissongroup_rates, sizeof(double)*_num__array_poissongroup_rates, cudaMemcpyDeviceToHost);

	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 2*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_ge;
	outfile__array_neurongroup_ge.open("results/_array_neurongroup_ge", ios::binary | ios::out);
	if(outfile__array_neurongroup_ge.is_open())
	{
		outfile__array_neurongroup_ge.write(reinterpret_cast<char*>(_array_neurongroup_ge), 1*sizeof(_array_neurongroup_ge[0]));
		outfile__array_neurongroup_ge.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_ge." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 1*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 1*sizeof(_array_neurongroup_v[0]));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	ofstream outfile__array_poissongroup__spikespace;
	outfile__array_poissongroup__spikespace.open("results/_array_poissongroup__spikespace", ios::binary | ios::out);
	if(outfile__array_poissongroup__spikespace.is_open())
	{
		outfile__array_poissongroup__spikespace.write(reinterpret_cast<char*>(_array_poissongroup__spikespace), 1001*sizeof(_array_poissongroup__spikespace[0]));
		outfile__array_poissongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_poissongroup__spikespace." << endl;
	}
	ofstream outfile__array_poissongroup_i;
	outfile__array_poissongroup_i.open("results/_array_poissongroup_i", ios::binary | ios::out);
	if(outfile__array_poissongroup_i.is_open())
	{
		outfile__array_poissongroup_i.write(reinterpret_cast<char*>(_array_poissongroup_i), 1000*sizeof(_array_poissongroup_i[0]));
		outfile__array_poissongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_poissongroup_i." << endl;
	}
	ofstream outfile__array_poissongroup_rates;
	outfile__array_poissongroup_rates.open("results/_array_poissongroup_rates", ios::binary | ios::out);
	if(outfile__array_poissongroup_rates.is_open())
	{
		outfile__array_poissongroup_rates.write(reinterpret_cast<char*>(_array_poissongroup_rates), 1000*sizeof(_array_poissongroup_rates[0]));
		outfile__array_poissongroup_rates.close();
	} else
	{
		std::cout << "Error writing output file for _array_poissongroup_rates." << endl;
	}
	ofstream outfile__array_spikemonitor__count;
	outfile__array_spikemonitor__count.open("results/_array_spikemonitor__count", ios::binary | ios::out);
	if(outfile__array_spikemonitor__count.is_open())
	{
		outfile__array_spikemonitor__count.write(reinterpret_cast<char*>(_array_spikemonitor__count), 1000*sizeof(_array_spikemonitor__count[0]));
		outfile__array_spikemonitor__count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__count." << endl;
	}
	ofstream outfile__array_statemonitor__indices;
	outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices", ios::binary | ios::out);
	if(outfile__array_statemonitor__indices.is_open())
	{
		outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 2*sizeof(_array_statemonitor__indices[0]));
		outfile__array_statemonitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
	}
	ofstream outfile__array_synapses_N_incoming;
	outfile__array_synapses_N_incoming.open("results/_array_synapses_N_incoming", ios::binary | ios::out);
	if(outfile__array_synapses_N_incoming.is_open())
	{
		outfile__array_synapses_N_incoming.write(reinterpret_cast<char*>(_array_synapses_N_incoming), 1*sizeof(_array_synapses_N_incoming[0]));
		outfile__array_synapses_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N_incoming." << endl;
	}
	ofstream outfile__array_synapses_N_outgoing;
	outfile__array_synapses_N_outgoing.open("results/_array_synapses_N_outgoing", ios::binary | ios::out);
	if(outfile__array_synapses_N_outgoing.is_open())
	{
		outfile__array_synapses_N_outgoing.write(reinterpret_cast<char*>(_array_synapses_N_outgoing), 1000*sizeof(_array_synapses_N_outgoing[0]));
		outfile__array_synapses_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N_outgoing." << endl;
	}
	ofstream outfile__dynamic_array_ratemonitor_rate;
	outfile__dynamic_array_ratemonitor_rate.open("results/_dynamic_array_ratemonitor_rate", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_rate.is_open())
	{
		thrust::host_vector<double> _dynamic_array_ratemonitor_rate_host = _dynamic_array_ratemonitor_rate;
		outfile__dynamic_array_ratemonitor_rate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_rate_host[0])), _dynamic_array_ratemonitor_rate_host.size()*sizeof(_dynamic_array_ratemonitor_rate_host[0]));
		outfile__dynamic_array_ratemonitor_rate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_rate." << endl;
	}
	ofstream outfile__dynamic_array_ratemonitor_t;
	outfile__dynamic_array_ratemonitor_t.open("results/_dynamic_array_ratemonitor_t", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_t.is_open())
	{
		thrust::host_vector<double> _dynamic_array_ratemonitor_t_host = _dynamic_array_ratemonitor_t;
		outfile__dynamic_array_ratemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_ratemonitor_t_host[0])), _dynamic_array_ratemonitor_t_host.size()*sizeof(_dynamic_array_ratemonitor_t_host[0]));
		outfile__dynamic_array_ratemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i", ios::binary | ios::out);
/*
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
		thrust::host_vector<int32_t> _dynamic_array_spikemonitor_i_host = _dynamic_array_spikemonitor_i;
		outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_i_host[0])), _dynamic_array_spikemonitor_i_host.size()*sizeof(_dynamic_array_spikemonitor_i_host[0]));
		outfile__dynamic_array_spikemonitor_i.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
*/
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t", ios::binary | ios::out);
/*
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
		thrust::host_vector<double> _dynamic_array_spikemonitor_t_host = _dynamic_array_spikemonitor_t;
		outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_spikemonitor_t_host[0])), _dynamic_array_spikemonitor_t_host.size()*sizeof(_dynamic_array_spikemonitor_t_host[0]));
		outfile__dynamic_array_spikemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}
*/
	ofstream outfile__dynamic_array_statemonitor_t;
	outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_t.is_open())
	{
		thrust::host_vector<double> _dynamic_array_statemonitor_t_host = _dynamic_array_statemonitor_t;
		outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor_t_host[0])), _dynamic_array_statemonitor_t_host.size()*sizeof(_dynamic_array_statemonitor_t_host[0]));
		outfile__dynamic_array_statemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_post;
	outfile__dynamic_array_synapses__synaptic_post.open("results/_dynamic_array_synapses__synaptic_post", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_post.is_open())
	{
		thrust::host_vector<int32_t> _dynamic_array_synapses__synaptic_post_host = _dynamic_array_synapses__synaptic_post;
		outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_post_host[0])), _dynamic_array_synapses__synaptic_post_host.size()*sizeof(_dynamic_array_synapses__synaptic_post_host[0]));
		outfile__dynamic_array_synapses__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
		thrust::host_vector<int32_t> _dynamic_array_synapses__synaptic_pre_host = _dynamic_array_synapses__synaptic_pre;
		outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses__synaptic_pre_host[0])), _dynamic_array_synapses__synaptic_pre_host.size()*sizeof(_dynamic_array_synapses__synaptic_pre_host[0]));
		outfile__dynamic_array_synapses__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_Apost;
	outfile__dynamic_array_synapses_Apost.open("results/_dynamic_array_synapses_Apost", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_Apost.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_Apost_host = _dynamic_array_synapses_Apost;
		outfile__dynamic_array_synapses_Apost.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_Apost_host[0])), _dynamic_array_synapses_Apost_host.size()*sizeof(_dynamic_array_synapses_Apost_host[0]));
		outfile__dynamic_array_synapses_Apost.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_Apost." << endl;
	}
	ofstream outfile__dynamic_array_synapses_Apre;
	outfile__dynamic_array_synapses_Apre.open("results/_dynamic_array_synapses_Apre", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_Apre.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_Apre_host = _dynamic_array_synapses_Apre;
		outfile__dynamic_array_synapses_Apre.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_Apre_host[0])), _dynamic_array_synapses_Apre_host.size()*sizeof(_dynamic_array_synapses_Apre_host[0]));
		outfile__dynamic_array_synapses_Apost.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_Apre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_lastupdate;
	outfile__dynamic_array_synapses_lastupdate.open("results/_dynamic_array_synapses_lastupdate", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_lastupdate.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_lastupdate_host = _dynamic_array_synapses_lastupdate;
		outfile__dynamic_array_synapses_lastupdate.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_lastupdate_host[0])), _dynamic_array_synapses_lastupdate_host.size()*sizeof(_dynamic_array_synapses_lastupdate_host[0]));
		outfile__dynamic_array_synapses_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_post_delay;
	outfile__dynamic_array_synapses_post_delay.open("results/_dynamic_array_synapses_post_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_post_delay.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_post_delay_host = _dynamic_array_synapses_post_delay;
		outfile__dynamic_array_synapses_post_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_post_delay_host[0])), _dynamic_array_synapses_post_delay_host.size()*sizeof(_dynamic_array_synapses_post_delay_host[0]));
		outfile__dynamic_array_synapses_post_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_post_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_pre_delay;
	outfile__dynamic_array_synapses_pre_delay.open("results/_dynamic_array_synapses_pre_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_pre_delay.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_pre_delay_host = _dynamic_array_synapses_pre_delay;
		outfile__dynamic_array_synapses_pre_delay.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_pre_delay_host[0])), _dynamic_array_synapses_pre_delay_host.size()*sizeof(_dynamic_array_synapses_pre_delay_host[0]));
		outfile__dynamic_array_synapses_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_pre_delay." << endl;
	}
	ofstream outfile__dynamic_array_synapses_w;
	outfile__dynamic_array_synapses_w.open("results/_dynamic_array_synapses_w", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_w.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_w_host = _dynamic_array_synapses_w;
		outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_w_host[0])), _dynamic_array_synapses_w_host.size()*sizeof(_dynamic_array_synapses_w_host[0]));
		outfile__dynamic_array_synapses_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor__recorded_w;
	outfile__dynamic_array_statemonitor__recorded_w.open("results/_dynamic_array_statemonitor__recorded_w", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor__recorded_w.is_open())
	{
		thrust::host_vector<double>* _dynamic_array_statemonitor__recorded_w_host_array = new thrust::host_vector<double>[_num__array_statemonitor__recorded_w];
		for(int i = 0; i < _num__array_statemonitor__recorded_w; i++)
		{
			_dynamic_array_statemonitor__recorded_w_host_array[i] = _dynamic_array_statemonitor__recorded_w[i];
		}
		for(int j = 0; j < _dynamic_array_statemonitor__recorded_w_host_array[0].size(); j++)
		{
			for(int n = 0; n < _num__array_statemonitor__recorded_w; n++)
			{
				outfile__dynamic_array_statemonitor__recorded_w.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor__recorded_w_host_array[n][j])), sizeof(_dynamic_array_statemonitor__recorded_w_host_array[0][0]));
			}
		}
		outfile__dynamic_array_statemonitor__recorded_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor__recorded_w." << endl;
	}
}

__global__ void dealloc_kernel(int par_num_threads)
{
	int tid = threadIdx.x;

	if(tid == 0)
	{
		brian::synapses_post.destroy();
		brian::synapses_pre.destroy();
	}

	delete brian::_dynamic_array_spikemonitor_i[tid];
	delete brian::_dynamic_array_spikemonitor_t[tid];
}

void _dealloc_arrays()
{
	using namespace brian;

	curandDestroyGenerator(gen);
	cudaFree(dev_array_rands);
	
	dealloc_kernel<<<1,num_blocks_sequential>>>(num_blocks_sequential);
	
	_dynamic_array_ratemonitor_rate.clear();
	thrust::device_vector<double>().swap(_dynamic_array_ratemonitor_rate);
	_dynamic_array_ratemonitor_t.clear();
	thrust::device_vector<double>().swap(_dynamic_array_ratemonitor_t);
	_dynamic_array_statemonitor_t.clear();
	thrust::device_vector<double>().swap(_dynamic_array_statemonitor_t);
	_dynamic_array_synapses_Apost.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_Apost);
	_dynamic_array_synapses_Apre.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_Apre);
	_dynamic_array_synapses_lastupdate.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_lastupdate);
	_dynamic_array_synapses_w.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_w);
	_dynamic_array_synapses_post_delay.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_post_delay);
	_dynamic_array_synapses_pre_delay.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_pre_delay);
	_dynamic_array_synapses__synaptic_post.clear();
	thrust::device_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
	_dynamic_array_synapses__synaptic_pre.clear();
	thrust::device_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);

	for(int i = 0; i < 2; i++)
	{
		_dynamic_array_statemonitor__recorded_w[i].clear();
		thrust::device_vector<double>().swap(_dynamic_array_statemonitor__recorded_w[i]);
	}
	
	if(_dynamic_array_statemonitor__recorded_w)
	{
		delete [] _dynamic_array_statemonitor__recorded_w;
	}
	
	if(_array_neurongroup__spikespace!=0)
	{
		delete [] _array_neurongroup__spikespace;
		_array_neurongroup__spikespace = 0;
	}
	cudaFree(dev_array_neurongroup__spikespace);
	if(_array_neurongroup_ge!=0)
	{
		delete [] _array_neurongroup_ge;
		_array_neurongroup_ge = 0;
	}
	cudaFree(dev_array_neurongroup_ge);

	if(_array_neurongroup_i!=0)
	{
		delete [] _array_neurongroup_i;
		_array_neurongroup_i = 0;
	}

	if(_array_neurongroup_v!=0)
	{
		delete [] _array_neurongroup_v;
		_array_neurongroup_v = 0;
	}
	cudaFree(dev_array_neurongroup_v);

	if(_array_poissongroup__spikespace!=0)
	{
		delete [] _array_poissongroup__spikespace;
		_array_poissongroup__spikespace = 0;
	}
	cudaFree(dev_array_poissongroup__spikespace);

	if(_array_poissongroup_i!=0)
	{
		delete [] _array_poissongroup_i;
		_array_poissongroup_i = 0;
	}

	if(_array_poissongroup_rates!=0)
	{
		delete [] _array_poissongroup_rates;
		_array_poissongroup_rates = 0;
	}
	cudaFree(dev_array_poissongroup_rates);

	if(_array_spikemonitor__count!=0)
	{
		delete [] _array_spikemonitor__count;
		_array_spikemonitor__count = 0;
	}

	if(_array_statemonitor__indices!=0)
	{
		delete [] _array_statemonitor__indices;
		_array_statemonitor__indices = 0;
	}

	if(_array_statemonitor__recorded_w!=0)
	{
		delete [] _array_statemonitor__recorded_w;
		_array_statemonitor__recorded_w = 0;
	}

	if(_array_synapses_N_incoming!=0)
	{
		delete [] _array_synapses_N_incoming;
		_array_synapses_N_incoming = 0;
	}

	if(_array_synapses_N_outgoing!=0)
	{
		delete [] _array_synapses_N_outgoing;
		_array_synapses_N_outgoing = 0;
	}

	// static arrays
	if(_static_array__array_statemonitor__indices!=0)
	{
		delete [] _static_array__array_statemonitor__indices;
		_static_array__array_statemonitor__indices = 0;
	}
}
