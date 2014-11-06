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

#define neuron_N 4000

//////////////// clocks ///////////////////
Clock brian::defaultclock(0.0001);

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
int32_t * brian::_array_neurongroup__spikespace;
int32_t * brian::dev_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 4001;

int32_t * brian::_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 4000;

double * brian::_array_neurongroup_lastspike;
double * brian::dev_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = 4000;

bool * brian::_array_neurongroup_not_refractory;
bool * brian::dev_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = 4000;

double * brian::_array_neurongroup_v;
double * brian::dev_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 4000;

double * brian::_array_neurongroup_w;
double * brian::dev_array_neurongroup_w;
const int brian::_num__array_neurongroup_w = 4000;

int32_t * brian::_array_spikemonitor__count;
const int brian::_num__array_spikemonitor__count = 4000;

int32_t * brian::_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 1;

int32_t * brian::_array_synapses_N_incoming;
const int brian::_num__array_synapses_N_incoming = 4000;

int32_t * brian::_array_synapses_N_outgoing;
const int brian::_num__array_synapses_N_outgoing = 4000;

unsigned* brian::dev_size_by_pre;
int32_t** brian::dev_synapses_id_by_pre;
int32_t** brian::dev_post_neuron_by_pre;
unsigned int** brian::dev_delay_by_pre;

//////////////// dynamic arrays 1d /////////
std::vector<double> brian::_dynamic_array_ratemonitor_rate;
std::vector<double> brian::_dynamic_array_ratemonitor_t;
thrust::device_vector<double> brian::_dynamic_array_synapses_c;
thrust::device_vector<double> brian::_dynamic_array_synapses_lastupdate;
thrust::device_vector<double> brian::_dynamic_array_synapses_pre_delay;
thrust::device_vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
thrust::device_vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
thrust::device_vector<int32_t> brian::synapses_by_pre_neuron;
std::vector<int32_t> brian::_dynamic_array_spikemonitor_i;
std::vector<double> brian::_dynamic_array_spikemonitor_t;
std::vector<double> brian::_dynamic_array_statemonitor_t;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor__recorded_v;
thrust::device_vector<double>* brian::_dynamic_array_statemonitor__recorded_w;

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_lastspike;
const int brian::_num__static_array__array_neurongroup_lastspike = 4000;

bool * brian::_static_array__array_neurongroup_not_refractory;
const int brian::_num__static_array__array_neurongroup_not_refractory = 4000;

int32_t * brian::_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 1;

unsigned int brian::num_blocks;

//////////////// synapses /////////////////
// synapses
Synapses<double> brian::synapses(4000, 4000);
__device__ SynapticPathway<double> brian::synapses_pre;

//////////////// random numbers /////////////////
float* brian::dev_array_random_floats;
curandGenerator_t brian::random_float_generator;

__global__ void deviceside_init(
	unsigned int num_blocks_sequential)
{
	using namespace brian;

	synapses_pre.init();
}

void _init_arrays()
{
	using namespace brian;

	num_blocks = 10;

	deviceside_init<<<1,1>>>(
		num_blocks);

	cudaMalloc((void**)&dev_array_random_floats, sizeof(float)*neuron_N);
	curandCreateGenerator(&random_float_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(random_float_generator, time(0));

	cudaMalloc((void**)&dev_size_by_pre, sizeof(unsigned int)*neuron_N*num_blocks);
	cudaMalloc((void**)&dev_synapses_id_by_pre, sizeof(int32_t*)*neuron_N*num_blocks);
	cudaMalloc((void**)&dev_post_neuron_by_pre, sizeof(int32_t*)*neuron_N*num_blocks);
	cudaMalloc((void**)&dev_delay_by_pre, sizeof(unsigned int*)*neuron_N*num_blocks);

    	// Arrays initialized to 0
	_array_spikemonitor__count = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_spikemonitor__count[i] = 0;

	_array_statemonitor__indices = new int32_t[1];
	for(int i=0; i<1; i++) _array_statemonitor__indices[i] = 0;

	_array_neurongroup__spikespace = new int32_t[4001];
	for(int i=0; i<4001; i++) _array_neurongroup__spikespace[i] = -1;
	cudaMalloc((void**)&dev_array_neurongroup__spikespace, sizeof(double)*_num__array_neurongroup__spikespace);
	cudaMemcpy(dev_array_neurongroup__spikespace, _array_neurongroup__spikespace, sizeof(double)*_num__array_neurongroup__spikespace, cudaMemcpyHostToDevice);

	_array_neurongroup_i = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_i[i] = 0;

	_array_neurongroup_lastspike = new double[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_lastspike[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike);
	cudaMemcpy(dev_array_neurongroup_lastspike, _array_neurongroup_lastspike, sizeof(double)*_num__array_neurongroup_lastspike, cudaMemcpyHostToDevice);

	_array_synapses_N_incoming = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_synapses_N_incoming[i] = 0;

	_array_synapses_N_outgoing = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_synapses_N_outgoing[i] = 0;

	_array_neurongroup_not_refractory = new bool[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_not_refractory[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_not_refractory, sizeof(double)*_num__array_neurongroup_not_refractory);
	cudaMemcpy(dev_array_neurongroup_not_refractory, _array_neurongroup_not_refractory, sizeof(double)*_num__array_neurongroup_not_refractory, cudaMemcpyHostToDevice);

	_array_neurongroup_v = new double[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_v[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v);
	cudaMemcpy(dev_array_neurongroup_v, _array_neurongroup_v, sizeof(double)*_num__array_neurongroup_v, cudaMemcpyHostToDevice);

	_array_neurongroup_w = new double[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_w[i] = 0;
	cudaMalloc((void**)&dev_array_neurongroup_w, sizeof(double)*_num__array_neurongroup_w);
	cudaMemcpy(dev_array_neurongroup_w, _array_neurongroup_w, sizeof(double)*_num__array_neurongroup_w, cudaMemcpyHostToDevice);

	// Arrays initialized to an "arange"
	_array_neurongroup_i = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_i[i] = 0 + i;

	_dynamic_array_statemonitor__recorded_v = new thrust::device_vector<double>[_num__array_statemonitor__indices];
	_dynamic_array_statemonitor__recorded_w = new thrust::device_vector<double>[_num__array_statemonitor__indices];

	// static arrays
	_static_array__array_neurongroup_lastspike = new double[4000];
	_static_array__array_neurongroup_not_refractory = new bool[4000];
	_static_array__array_statemonitor__indices = new int32_t[1];
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_neurongroup_lastspike;
	f_static_array__array_neurongroup_lastspike.open("static_arrays/_static_array__array_neurongroup_lastspike", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_lastspike.is_open())
	{
		f_static_array__array_neurongroup_lastspike.read(reinterpret_cast<char*>(_static_array__array_neurongroup_lastspike), 4000*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_lastspike." << endl;
	}



	ifstream f_static_array__array_neurongroup_not_refractory;
	f_static_array__array_neurongroup_not_refractory.open("static_arrays/_static_array__array_neurongroup_not_refractory", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_not_refractory.is_open())
	{
		f_static_array__array_neurongroup_not_refractory.read(reinterpret_cast<char*>(_static_array__array_neurongroup_not_refractory), 4000*sizeof(bool));
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

void _write_arrays()
{
/*
	using namespace brian;

	ofstream outfile__array_neurongroup__spikespace;
	outfile__array_neurongroup__spikespace.open("results/_array_neurongroup__spikespace", ios::binary | ios::out);
	if(outfile__array_neurongroup__spikespace.is_open())
	{
		outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 4001*sizeof(_array_neurongroup__spikespace[0]));
		outfile__array_neurongroup__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 4000*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_lastspike;
	outfile__array_neurongroup_lastspike.open("results/_array_neurongroup_lastspike", ios::binary | ios::out);
	if(outfile__array_neurongroup_lastspike.is_open())
	{
		outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 4000*sizeof(_array_neurongroup_lastspike[0]));
		outfile__array_neurongroup_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
	}
	ofstream outfile__array_neurongroup_not_refractory;
	outfile__array_neurongroup_not_refractory.open("results/_array_neurongroup_not_refractory", ios::binary | ios::out);
	if(outfile__array_neurongroup_not_refractory.is_open())
	{
		outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 4000*sizeof(_array_neurongroup_not_refractory[0]));
		outfile__array_neurongroup_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
	}
	ofstream outfile__array_neurongroup_v;
	outfile__array_neurongroup_v.open("results/_array_neurongroup_v", ios::binary | ios::out);
	if(outfile__array_neurongroup_v.is_open())
	{
		outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 4000*sizeof(_array_neurongroup_v[0]));
		outfile__array_neurongroup_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_v." << endl;
	}
	ofstream outfile__array_neurongroup_w;
	outfile__array_neurongroup_w.open("results/_array_neurongroup_w", ios::binary | ios::out);
	if(outfile__array_neurongroup_w.is_open())
	{
		outfile__array_neurongroup_w.write(reinterpret_cast<char*>(_array_neurongroup_w), 4000*sizeof(_array_neurongroup_w[0]));
		outfile__array_neurongroup_w.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_w." << endl;
	}
	ofstream outfile__array_spikemonitor__count;
	outfile__array_spikemonitor__count.open("results/_array_spikemonitor__count", ios::binary | ios::out);
	if(outfile__array_spikemonitor__count.is_open())
	{
		outfile__array_spikemonitor__count.write(reinterpret_cast<char*>(_array_spikemonitor__count), 4000*sizeof(_array_spikemonitor__count[0]));
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
		outfile__array_synapses_N_incoming.write(reinterpret_cast<char*>(_array_synapses_N_incoming), 4000*sizeof(_array_synapses_N_incoming[0]));
		outfile__array_synapses_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N_incoming." << endl;
	}
	ofstream outfile__array_synapses_N_outgoing;
	outfile__array_synapses_N_outgoing.open("results/_array_synapses_N_outgoing", ios::binary | ios::out);
	if(outfile__array_synapses_N_outgoing.is_open())
	{
		outfile__array_synapses_N_outgoing.write(reinterpret_cast<char*>(_array_synapses_N_outgoing), 4000*sizeof(_array_synapses_N_outgoing[0]));
		outfile__array_synapses_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _array_synapses_N_outgoing." << endl;
	}

	ofstream outfile__dynamic_array_ratemonitor_rate;
	outfile__dynamic_array_ratemonitor_rate.open("results/_dynamic_array_ratemonitor_rate", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_rate.is_open())
	{
		outfile__dynamic_array_ratemonitor_rate.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_rate[0]), _dynamic_array_ratemonitor_rate.size()*sizeof(_dynamic_array_ratemonitor_rate[0]));
		outfile__dynamic_array_ratemonitor_rate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_rate." << endl;
	}
	ofstream outfile__dynamic_array_ratemonitor_t;
	outfile__dynamic_array_ratemonitor_t.open("results/_dynamic_array_ratemonitor_t", ios::binary | ios::out);
	if(outfile__dynamic_array_ratemonitor_t.is_open())
	{
		outfile__dynamic_array_ratemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_ratemonitor_t[0]), _dynamic_array_ratemonitor_t.size()*sizeof(_dynamic_array_ratemonitor_t[0]));
		outfile__dynamic_array_ratemonitor_t.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_ratemonitor_t." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_i;
	outfile__dynamic_array_spikemonitor_i.open("results/_dynamic_array_spikemonitor_i", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_i.is_open())
	{
		outfile__dynamic_array_spikemonitor_i.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_i[0]), _dynamic_array_spikemonitor_i.size()*sizeof(_dynamic_array_spikemonitor_i[0]));
		outfile__dynamic_array_spikemonitor_i.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_spikemonitor_i." << endl;
	}
	ofstream outfile__dynamic_array_spikemonitor_t;
	outfile__dynamic_array_spikemonitor_t.open("results/_dynamic_array_spikemonitor_t", ios::binary | ios::out);
	if(outfile__dynamic_array_spikemonitor_t.is_open())
	{
		outfile__dynamic_array_spikemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_spikemonitor_t[0]), _dynamic_array_spikemonitor_t.size()*sizeof(_dynamic_array_spikemonitor_t[0]));
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
		outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
		outfile__dynamic_array_synapses__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_synapses__synaptic_pre;
	outfile__dynamic_array_synapses__synaptic_pre.open("results/_dynamic_array_synapses__synaptic_pre", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
	{
		outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
		outfile__dynamic_array_synapses__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_synapses_c;
	outfile__dynamic_array_synapses_c.open("results/_dynamic_array_synapses_c", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_c.is_open())
	{
		outfile__dynamic_array_synapses_c.write(reinterpret_cast<char*>(&_dynamic_array_synapses_c[0]), _dynamic_array_synapses_c.size()*sizeof(_dynamic_array_synapses_c[0]));
		outfile__dynamic_array_synapses_c.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_c." << endl;
	}
	ofstream outfile__dynamic_array_synapses_lastupdate;
	outfile__dynamic_array_synapses_lastupdate.open("results/_dynamic_array_synapses_lastupdate", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_lastupdate.is_open())
	{
		outfile__dynamic_array_synapses_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_synapses_lastupdate[0]), _dynamic_array_synapses_lastupdate.size()*sizeof(_dynamic_array_synapses_lastupdate[0]));
		outfile__dynamic_array_synapses_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_synapses_pre_delay;
	outfile__dynamic_array_synapses_pre_delay.open("results/_dynamic_array_synapses_pre_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_synapses_pre_delay.is_open())
	{
		outfile__dynamic_array_synapses_pre_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_pre_delay[0]), _dynamic_array_synapses_pre_delay.size()*sizeof(_dynamic_array_synapses_pre_delay[0]));
		outfile__dynamic_array_synapses_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_synapses_pre_delay." << endl;
	}

	ofstream outfile__dynamic_array_statemonitor__recorded_v;
	outfile__dynamic_array_statemonitor__recorded_v.open("results/_dynamic_array_statemonitor__recorded_v", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor__recorded_v.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor__recorded_v.n; n++)
        {
            outfile__dynamic_array_statemonitor__recorded_v.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor__recorded_v(n, 0)), _dynamic_array_statemonitor__recorded_v.m*sizeof(_dynamic_array_statemonitor__recorded_v(0, 0)));
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
        for (int n=0; n<_dynamic_array_statemonitor__recorded_w.n; n++)
        {
            outfile__dynamic_array_statemonitor__recorded_w.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor__recorded_w(n, 0)), _dynamic_array_statemonitor__recorded_w.m*sizeof(_dynamic_array_statemonitor__recorded_w(0, 0)));
        }
        outfile__dynamic_array_statemonitor__recorded_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor__recorded_w." << endl;
	}
*/
}

__global__ void deviceside_destroy()
{
	using namespace brian;

	synapses_pre.destroy();
}

void _dealloc_arrays()
{
	using namespace brian;

	deviceside_destroy<<<1,1>>>();

	//temp array of device pointers
	int32_t** temp_synapses_by_pre_id = new int32_t*[num_blocks*neuron_N];
	int32_t** temp_post_neuron_by_pre_id = new int32_t*[num_blocks*neuron_N];
	unsigned int** temp_delay_by_pre_id = new unsigned int*[num_blocks*neuron_N];

	cudaMemcpy(temp_synapses_by_pre_id, dev_synapses_id_by_pre, sizeof(int32_t*)*neuron_N*num_blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_post_neuron_by_pre_id, dev_post_neuron_by_pre, sizeof(int32_t*)*neuron_N*num_blocks, cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_delay_by_pre_id, dev_delay_by_pre, sizeof(unsigned int*)*neuron_N*num_blocks, cudaMemcpyDeviceToHost);

	for(int i = 0; i < num_blocks*neuron_N; i++)
	{
		cudaFree(temp_synapses_by_pre_id[i]);
		cudaFree(temp_post_neuron_by_pre_id[i]);
		cudaFree(temp_delay_by_pre_id[i]);
	}

	delete [] temp_synapses_by_pre_id;
	delete [] temp_post_neuron_by_pre_id;
	delete [] temp_delay_by_pre_id;

	cudaFree(dev_size_by_pre);
	cudaFree(dev_synapses_id_by_pre);
	cudaFree(dev_post_neuron_by_pre);
	cudaFree(dev_delay_by_pre);

	curandDestroyGenerator(random_float_generator);
	cudaFree(dev_array_random_floats);

	synapses_by_pre_neuron.clear();
	thrust::device_vector<int32_t>().swap(synapses_by_pre_neuron);

	_dynamic_array_synapses_c.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_c);

	_dynamic_array_synapses_lastupdate.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_lastupdate);

	_dynamic_array_synapses_pre_delay.clear();
	thrust::device_vector<double>().swap(_dynamic_array_synapses_pre_delay);

	_dynamic_array_synapses__synaptic_post.clear();
	thrust::device_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_post);

	_dynamic_array_synapses__synaptic_pre.clear();
	thrust::device_vector<int32_t>().swap(_dynamic_array_synapses__synaptic_pre);

	if(_array_neurongroup__spikespace!=0)
	{
		delete [] _array_neurongroup__spikespace;
		_array_neurongroup__spikespace = 0;
		cudaFree(dev_array_neurongroup__spikespace);
		dev_array_neurongroup__spikespace = 0;
	}

	if(_array_neurongroup_i!=0)
	{
		delete [] _array_neurongroup_i;
		_array_neurongroup_i = 0;
	}

	if(_array_neurongroup_lastspike!=0)
	{
		delete [] _array_neurongroup_lastspike;
		_array_neurongroup_lastspike = 0;
		cudaFree(dev_array_neurongroup_lastspike);
		dev_array_neurongroup_lastspike = 0;
	}

	if(_array_neurongroup_not_refractory!=0)
	{
		delete [] _array_neurongroup_not_refractory;
		_array_neurongroup_not_refractory = 0;
		cudaFree(dev_array_neurongroup_not_refractory);
		dev_array_neurongroup_not_refractory = 0;
	}

	if(_array_neurongroup_v!=0)
	{
		delete [] _array_neurongroup_v;
		_array_neurongroup_v = 0;
		cudaFree(dev_array_neurongroup_v);
		dev_array_neurongroup_v = 0;
	}

	if(_array_neurongroup_w!=0)
	{
		delete [] _array_neurongroup_w;
		_array_neurongroup_w = 0;
		cudaFree(dev_array_neurongroup_w);
		dev_array_neurongroup_w = 0;
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

	if(_dynamic_array_statemonitor__recorded_v)
	{
		delete [] _dynamic_array_statemonitor__recorded_v;
		_dynamic_array_statemonitor__recorded_v = 0;
	}

	if(_dynamic_array_statemonitor__recorded_w)
	{
		delete [] _dynamic_array_statemonitor__recorded_w;
		_dynamic_array_statemonitor__recorded_w = 0;
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

