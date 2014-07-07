#include<stdint.h>
#include<vector>
#include "objects.h"
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"
#include<iostream>
#include<fstream>

#include <curand.h>
#include <thrust/device_vector.h>

#define N 4000

//////////////// clocks ///////////////////
Clock brian::defaultclock(0.0001);

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = N+1;
int32_t * brian::dev_array_neurongroup__spikespace;

int32_t * brian::_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = N;
int32_t * brian::dev_array_neurongroup_i;

double * brian::_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = N;
double * brian::dev_array_neurongroup_lastspike;

bool * brian::_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = N;
bool * brian::dev_array_neurongroup_not_refractory;

double * brian::_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = N;
double * brian::dev_array_neurongroup_v;

double * brian::_array_neurongroup_w;
const int brian::_num__array_neurongroup_w = N;
double * brian::dev_array_neurongroup_w;

int32_t * brian::_array_spikemonitor__count;
const int brian::_num__array_spikemonitor__count = N;

int32_t * brian::_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 1;

double * brian::_array_statemonitor__recorded_v;
const int brian::_num__array_statemonitor__recorded_v = (0, 1);

double * brian::_array_statemonitor__recorded_w;
const int brian::_num__array_statemonitor__recorded_w = (0, 1);

int32_t * brian::_array_synapses_N_incoming;
const int brian::_num__array_synapses_N_incoming = N;

int32_t * brian::_array_synapses_N_outgoing;
const int brian::_num__array_synapses_N_outgoing = N;

//////////////// dynamic arrays 1d /////////
std::vector<double> brian::_dynamic_array_statemonitor_t;

thrust::device_vector<double> brian::_dynamic_array_ratemonitor_rate;
thrust::device_vector<double> brian::_dynamic_array_ratemonitor_t;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor__recorded_v;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor__recorded_w;
thrust::device_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<double> brian::_dynamic_array_synapses_c;
thrust::device_vector<double> brian::_dynamic_array_synapses_lastupdate;
thrust::device_vector<double> brian::_dynamic_array_synapses_pre_delay;

__device__ CudaVector<int32_t>** brian::_dynamic_array_spikemonitor_i;
__device__ CudaVector<double>** brian::_dynamic_array_spikemonitor_t;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_lastspike;
const int brian::_num__static_array__array_neurongroup_lastspike = N;

bool * brian::_static_array__array_neurongroup_not_refractory;
const int brian::_num__static_array__array_neurongroup_not_refractory = N;

int32_t * brian::_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 1;

//////////////// synapses /////////////////
// synapses
Synapses<double> brian::synapses(N, N);
__device__ SynapticPathway<double> brian::synapses_pre;

///////////// random number generation //////

float* brian::dev_array_random;
curandGenerator_t brian::gen;

int brian::num_blocks_sequential = 8;

__global__ void init_kernel(int num_mps)
{
	using namespace brian;

	int tid = threadIdx.x;

	if(tid == 0)
	{
		synapses_pre.init(
			num_mps,
			N, N,
			0.0001,
			0, N
			);

		_dynamic_array_spikemonitor_i = new CudaVector<int32_t>*[num_mps];
		_dynamic_array_spikemonitor_t = new CudaVector<double>*[num_mps];
	}
	__syncthreads();
	_dynamic_array_spikemonitor_i[tid] = new CudaVector<int32_t>;
	_dynamic_array_spikemonitor_t[tid] = new CudaVector<double>;
}

void _init_arrays()
{
	using namespace brian;

	cudaMalloc((void**)&dev_array_random, sizeof(float)*N);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));

	init_kernel<<<1,num_blocks_sequential>>>(num_blocks_sequential);

    // Arrays initialized to 0
	_array_spikemonitor__count = new int32_t[N];
	for(int i=0; i<N; i++) _array_spikemonitor__count[i] = 0;

	_array_statemonitor__indices = new int32_t[1];
	for(int i=0; i<1; i++) _array_statemonitor__indices[i] = 0;

	_array_neurongroup__spikespace = new int32_t[N+1];
	for(int i=0; i<N+1; i++) _array_neurongroup__spikespace[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace);
	cudaMemcpy(dev_array_neurongroup__spikespace, _array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);

	_array_neurongroup_i = new int32_t[N];
	for(int i=0; i<N; i++) _array_neurongroup_i[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i);
	cudaMemcpy(dev_array_neurongroup_i, _array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyHostToDevice);

	_array_neurongroup_lastspike = new double[N];
	for(int i=0; i<N; i++) _array_neurongroup_lastspike[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike);
	cudaMemcpy(dev_array_neurongroup_lastspike, _array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice);

	_array_synapses_N_incoming = new int32_t[N];
	for(int i=0; i<N; i++) _array_synapses_N_incoming[i] = 0;

	_array_synapses_N_outgoing = new int32_t[N];
	for(int i=0; i<N; i++) _array_synapses_N_outgoing[i] = 0;

	_array_neurongroup_not_refractory = new bool[N];
	for(int i=0; i<N; i++) _array_neurongroup_not_refractory[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_not_refractory, sizeof(bool)*_num__array_neurongroup_not_refractory);
	cudaMemcpy(dev_array_neurongroup_not_refractory, _array_neurongroup_not_refractory, sizeof(bool)*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice);

	_array_neurongroup_v = new double[N];
	for(int i=0; i<N; i++) _array_neurongroup_v[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v);
	cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice);

	_array_neurongroup_w = new double[N];
	for(int i=0; i<N; i++) _array_neurongroup_w[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_w, sizeof(double)*_num__array_neurongroup_w);
	cudaMemcpy(dev_array_neurongroup_w, _array_neurongroup_w, sizeof(double)*_num__array_neurongroup_w, cudaMemcpyHostToDevice);

	// Arrays initialized to an "arange"
	_array_neurongroup_i = new int32_t[N];
	for(int i=0; i<N; i++) _array_neurongroup_i[i] = 0 + i;

	// static arrays
	_static_array__array_neurongroup_lastspike = new double[N];
	_static_array__array_neurongroup_not_refractory = new bool[N];
	_static_array__array_statemonitor__indices = new int32_t[1];

	_dynamic_array_statemonitor__recorded_v = new thrust::device_vector<double>[_num__array_statemonitor__recorded_v];
	_dynamic_array_statemonitor__recorded_w = new thrust::device_vector<double>[_num__array_statemonitor__recorded_w];
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_neurongroup_lastspike;
	f_static_array__array_neurongroup_lastspike.open("static_arrays/_static_array__array_neurongroup_lastspike", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_lastspike.is_open())
	{
		f_static_array__array_neurongroup_lastspike.read(reinterpret_cast<char*>(_static_array__array_neurongroup_lastspike), N*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_lastspike." << endl;
	}
	ifstream f_static_array__array_neurongroup_not_refractory;
	f_static_array__array_neurongroup_not_refractory.open("static_arrays/_static_array__array_neurongroup_not_refractory", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_not_refractory.is_open())
	{
		f_static_array__array_neurongroup_not_refractory.read(reinterpret_cast<char*>(_static_array__array_neurongroup_not_refractory), N*sizeof(bool));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_not_refractory." << endl;
	}
	ifstream f_static_array__array_statemonitor__indices;
	f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor__indices.is_open())
	{
		f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 1*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
	}
}

__global__ void get_CudaVector_data(int32_t** i_data, int* i_sizes, double** t_data, int* t_sizes)
{
	using namespace brian;

	int tid = threadIdx.x;

	i_data[tid] = _dynamic_array_spikemonitor_i[tid]->content();
	i_sizes[tid] = _dynamic_array_spikemonitor_i[tid]->size();
	t_data[tid] = _dynamic_array_spikemonitor_t[tid]->content();
	t_sizes[tid] = _dynamic_array_spikemonitor_t[tid]->size();
}

void _write_arrays()
{
	using namespace brian;

	int32_t* spikemonitor_i_data = (int32_t*)malloc(sizeof(int*)*num_blocks_sequential);
	int* spikemonitor_i_sizes = (int*)malloc(sizeof(int*)*num_blocks_sequential);
	double* spikemonitor_t_data = (double*)malloc(sizeof(double*)*num_blocks_sequential);
	int* spikemonitor_t_sizes = (int*)malloc(sizeof(int*)*num_blocks_sequential);

	int32_t** dev_spikemonitor_i_data;
	cudaMalloc((void**)&dev_spikemonitor_i_data, sizeof(int32_t**)*num_blocks_sequential);
	int* dev_spikemonitor_i_sizes;
	cudaMalloc((void**)&dev_spikemonitor_i_sizes, sizeof(int*)*num_blocks_sequential);
	double** dev_spikemonitor_t_data;
	cudaMalloc((void**)&dev_spikemonitor_t_data, sizeof(double**)*num_blocks_sequential);
	int* dev_spikemonitor_t_sizes;
	cudaMalloc((void**)&dev_spikemonitor_t_sizes, sizeof(int*)*num_blocks_sequential);

	get_CudaVector_data<<<1, num_blocks_sequential>>>(dev_spikemonitor_i_data, dev_spikemonitor_i_sizes, dev_spikemonitor_t_data, dev_spikemonitor_t_sizes);

	cudaMemcpy(spikemonitor_i_data, dev_spikemonitor_i_data, sizeof(int32_t)*num_blocks_sequential, cudaMemcpyDeviceToHost);
	cudaMemcpy(spikemonitor_i_sizes, dev_spikemonitor_i_sizes, sizeof(int)*num_blocks_sequential, cudaMemcpyDeviceToHost);
	cudaMemcpy(spikemonitor_t_data, dev_spikemonitor_t_data, sizeof(double)*num_blocks_sequential, cudaMemcpyDeviceToHost);
	cudaMemcpy(spikemonitor_t_sizes, dev_spikemonitor_t_sizes, sizeof(int)*num_blocks_sequential, cudaMemcpyDeviceToHost);

	cudaMemcpy(_array_neurongroup__spikespace, dev_array_neurongroup__spikespace, sizeof(int32_t)*_num__array_neurongroup__spikespace, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_i, dev_array_neurongroup_i, sizeof(int32_t)*_num__array_neurongroup_i, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_lastspike, dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_not_refractory, dev_array_neurongroup_not_refractory, sizeof(bool)*_num__array_neurongroup_not_refractory, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_v, dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyDeviceToHost);
	cudaMemcpy(_array_neurongroup_w, dev_array_neurongroup_w, sizeof(double)*_num__array_neurongroup_w, cudaMemcpyDeviceToHost);



	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), N+1*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}



	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), N*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}



	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), N*sizeof(_array_neurongroup_lastspike[0]));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}



	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), N*sizeof(_array_neurongroup_not_refractory[0]));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}



	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), N*sizeof(_array_neurongroup_v[0]));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}



	ofstream outfile__array_neurongroup_w;
	outfile__array_neurongroup_w.open("results/_array_neurongroup_w", ios::binary | ios::out);
	if(outfile__array_neurongroup_w.is_open())
	{
		outfile__array_neurongroup_w.write(reinterpret_cast<char*>(_array_neurongroup_w), N*sizeof(_array_neurongroup_w[0]));
		outfile__array_neurongroup_w.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_w." << endl;
	}



	ofstream outfile__array_spikemonitor__count;
	outfile__array_spikemonitor__count.open("results/_array_spikemonitor__count", ios::binary | ios::out);
	if(outfile__array_spikemonitor__count.is_open())
	{
		outfile__array_spikemonitor__count.write(reinterpret_cast<char*>(_array_spikemonitor__count), N*sizeof(_array_spikemonitor__count[0]));
		outfile__array_spikemonitor__count.close();
	} else
	{
		std::cout << "Error writing output file for _array_spikemonitor__count." << endl;
	}



	ofstream outfile__array_statemonitor__indices;
	outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices", ios::binary | ios::out);
	if(outfile__array_statemonitor__indices.is_open())
	{
		outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 1*sizeof(_array_statemonitor__indices[0]));
		outfile__array_statemonitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
	}



	ofstream outfile__array_synapses_N_incoming;
	outfile__array_synapses_N_incoming.open("results/_array_synapses_N_incoming", ios::binary | ios::out);
	if(outfile__array_synapses_N_incoming.is_open())
	{
		outfile__array_synapses_N_incoming.write(reinterpret_cast<char*>(_array_synapses_N_incoming), N*sizeof(_array_synapses_N_incoming[0]));
		outfile__array_synapses_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N_incoming." << endl;
	}



	ofstream outfile__array_synapses_N_outgoing;
	outfile__array_synapses_N_outgoing.open("results/_array_synapses_N_outgoing", ios::binary | ios::out);
	if(outfile__array_synapses_N_outgoing.is_open())
	{
		outfile__array_synapses_N_outgoing.write(reinterpret_cast<char*>(_array_synapses_N_outgoing), N*sizeof(_array_synapses_N_outgoing[0]));
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
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
		for(int i = 0; i < num_blocks_sequential; i++)
		{
			outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&spikemonitor_i_data[i]), spikemonitor_i_sizes[i]*sizeof(int32_t));
		}
		outfile__dynamic_array_spikemonitor_i.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}



	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
		for(int i = 0; i < num_blocks_sequential; i++)
		{
			outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&spikemonitor_t_data[i]), spikemonitor_t_sizes[i]*sizeof(double));
		}
		outfile__dynamic_array_spikemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_t." << endl;
	}



	ofstream outfile__dynamic_array_statemonitor_t;
	outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_t.is_open())
	{
		outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
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



	ofstream outfile__dynamic_array_synapses_c;
	outfile__dynamic_array_synapses_c.open("results/_dynamic_array_synapses_c", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_c.is_open())
	{
		thrust::host_vector<double> _dynamic_array_synapses_c_host = _dynamic_array_synapses_c;
		outfile__dynamic_array_synapses_c.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_synapses_c_host[0])), _dynamic_array_synapses_c_host.size()*sizeof(_dynamic_array_synapses_c_host[0]));
		outfile__dynamic_array_synapses_c.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_c." << endl;
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



	ofstream outfile__dynamic_array_statemonitor__recorded_v;
	outfile__dynamic_array_statemonitor__recorded_v.open("results/_dynamic_array_statemonitor__recorded_v", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor__recorded_v.is_open())
	{
		thrust::host_vector<double>* _dynamic_array_statemonitor__recorded_v_host_array = new thrust::host_vector<double>[_num__array_statemonitor__recorded_v];
		for(int i = 0; i < _num__array_statemonitor__recorded_v; i++)
		{
			_dynamic_array_statemonitor__recorded_v_host_array[i] = _dynamic_array_statemonitor__recorded_v[i];
		}
		for(int j = 0; j < _dynamic_array_statemonitor__recorded_v_host_array[0].size(); j++)
		{
			for(int n = 0; n < _num__array_statemonitor__recorded_v; n++)
			{
				outfile__dynamic_array_statemonitor__recorded_v.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&_dynamic_array_statemonitor__recorded_v_host_array[n][j])), sizeof(_dynamic_array_statemonitor__recorded_v_host_array[0][0]));
			}
		}
		outfile__dynamic_array_statemonitor__recorded_v.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor__recorded_v." << endl;
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


	free(spikemonitor_i_data);
	free(spikemonitor_i_sizes);
	free(spikemonitor_t_data);
	free(spikemonitor_t_sizes);
	cudaFree(dev_spikemonitor_i_data);
	cudaFree(dev_spikemonitor_i_sizes);
	cudaFree(dev_spikemonitor_t_data);
	cudaFree(dev_spikemonitor_t_sizes);
}

__global__ void dealloc_kernel(int par_num_threads)
{
	using namespace brian;

	int tid = threadIdx.x;

	if(tid == 0)
	{
		synapses_pre.destroy();
	}
	delete _dynamic_array_spikemonitor_i[tid];
	delete _dynamic_array_spikemonitor_t[tid];
}

void _dealloc_arrays()
{
	using namespace brian;

	curandDestroyGenerator(gen);
	cudaFree(dev_array_random);

	int num_blocks_sequential = 1;
	dealloc_kernel<<<1, num_blocks_sequential>>>(num_blocks_sequential);


	_dynamic_array_ratemonitor_rate.clear();
	thrust::device_vector<double>().swap(_dynamic_array_ratemonitor_rate);
	_dynamic_array_ratemonitor_t.clear();
	thrust::device_vector<double>().swap(_dynamic_array_ratemonitor_t);
	_dynamic_array_synapses__synaptic_post.clear();
	thrust::device_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);
	_dynamic_array_synapses__synaptic_pre.clear();
	thrust::device_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);
	_dynamic_array_synapses_c.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_c);
	_dynamic_array_synapses_lastupdate.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_lastupdate);
	_dynamic_array_synapses_pre_delay.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_pre_delay);

	if(_array_neurongroup__spikespace!=0)
	{
		delete [] _array_neurongroup__spikespace;
		_array_neurongroup__spikespace = 0;
		cudaFree(dev_array_neurongroup__spikespace);
	}

	if(_array_neurongroup_i!=0)
	{
		delete [] _array_neurongroup_i;
		_array_neurongroup_i = 0;
		cudaFree(dev_array_neurongroup_i);
	}

	if(_array_neurongroup_lastspike!=0)
	{
		delete [] _array_neurongroup_lastspike;
		_array_neurongroup_lastspike = 0;
		cudaFree(dev_array_neurongroup_lastspike);
	}

	if(_array_neurongroup_not_refractory!=0)
	{
		delete [] _array_neurongroup_not_refractory;
		_array_neurongroup_not_refractory = 0;
		cudaFree(dev_array_neurongroup_not_refractory);
	}

	if(_array_neurongroup_v!=0)
	{
		delete [] _array_neurongroup_v;
		_array_neurongroup_v = 0;
		cudaFree(dev_array_neurongroup_v);
	}

	if(_array_neurongroup_w!=0)
	{
		delete [] _array_neurongroup_w;
		_array_neurongroup_w = 0;
		cudaFree(dev_array_neurongroup_w);
	}

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

	if(_array_statemonitor__recorded_v!=0)
	{
		delete [] _array_statemonitor__recorded_v;
		_array_statemonitor__recorded_v = 0;
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
	if(_static_array__array_neurongroup_lastspike!=0)
	{
		delete [] _static_array__array_neurongroup_lastspike;
		_static_array__array_neurongroup_lastspike = 0;
	}
	if(_static_array__array_neurongroup_not_refractory!=0)
	{
		delete [] _static_array__array_neurongroup_not_refractory;
		_static_array__array_neurongroup_not_refractory = 0;
	}
	if(_static_array__array_statemonitor__indices!=0)
	{
		delete [] _static_array__array_statemonitor__indices;
		_static_array__array_statemonitor__indices = 0;
	}
}

