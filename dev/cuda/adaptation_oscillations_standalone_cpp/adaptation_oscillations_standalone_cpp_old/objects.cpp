
#include<stdint.h>
#include<vector>
#include "objects.h"
#include "brianlib/synapses.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/network.h"
#include<iostream>
#include<fstream>

//////////////// clocks ///////////////////
Clock brian::defaultclock(0.0001);

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
int32_t * brian::_array_neurongroup__spikespace;
const int brian::_num__array_neurongroup__spikespace = 4001;
int32_t * brian::_array_neurongroup_i;
const int brian::_num__array_neurongroup_i = 4000;
double * brian::_array_neurongroup_lastspike;
const int brian::_num__array_neurongroup_lastspike = 4000;
bool * brian::_array_neurongroup_not_refractory;
const int brian::_num__array_neurongroup_not_refractory = 4000;
double * brian::_array_neurongroup_v;
const int brian::_num__array_neurongroup_v = 4000;
double * brian::_array_neurongroup_w;
const int brian::_num__array_neurongroup_w = 4000;
int32_t * brian::_array_spikemonitor__count;
const int brian::_num__array_spikemonitor__count = 4000;
int32_t * brian::_array_statemonitor__indices;
const int brian::_num__array_statemonitor__indices = 1;
double * brian::_array_statemonitor__recorded_v;
const int brian::_num__array_statemonitor__recorded_v = (0, 1);
double * brian::_array_statemonitor__recorded_w;
const int brian::_num__array_statemonitor__recorded_w = (0, 1);
int32_t * brian::_array_synapses_N_incoming;
const int brian::_num__array_synapses_N_incoming = 4000;
int32_t * brian::_array_synapses_N_outgoing;
const int brian::_num__array_synapses_N_outgoing = 4000;

//////////////// dynamic arrays 1d /////////
std::vector<double> brian::_dynamic_array_ratemonitor_rate;
std::vector<double> brian::_dynamic_array_ratemonitor_t;
std::vector<int32_t> brian::_dynamic_array_spikemonitor_i;
std::vector<double> brian::_dynamic_array_spikemonitor_t;
std::vector<double> brian::_dynamic_array_statemonitor_t;
std::vector<int32_t> brian::_dynamic_array_synapses__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_synapses__synaptic_pre;
std::vector<double> brian::_dynamic_array_synapses_c;
std::vector<double> brian::_dynamic_array_synapses_lastupdate;
std::vector<double> brian::_dynamic_array_synapses_pre_delay;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> brian::_dynamic_array_statemonitor__recorded_v;
DynamicArray2D<double> brian::_dynamic_array_statemonitor__recorded_w;

/////////////// static arrays /////////////
double * brian::_static_array__array_neurongroup_lastspike;
const int brian::_num__static_array__array_neurongroup_lastspike = 4000;
bool * brian::_static_array__array_neurongroup_not_refractory;
const int brian::_num__static_array__array_neurongroup_not_refractory = 4000;
int32_t * brian::_static_array__array_statemonitor__indices;
const int brian::_num__static_array__array_statemonitor__indices = 1;

//////////////// synapses /////////////////
// synapses
Synapses<double> brian::synapses(4000, 4000);
SynapticPathway<double> brian::synapses_pre(
		4000, 4000,
		_dynamic_array_synapses_pre_delay,
		_dynamic_array_synapses__synaptic_pre,
		0.0001,
		0, 4000
		);


void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_spikemonitor__count = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_spikemonitor__count[i] = 0;
	_array_statemonitor__indices = new int32_t[1];
	for(int i=0; i<1; i++) _array_statemonitor__indices[i] = 0;
	_array_neurongroup__spikespace = new int32_t[4001];
	for(int i=0; i<4001; i++) _array_neurongroup__spikespace[i] = 0;
	_array_neurongroup_i = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_i[i] = 0;
	_array_neurongroup_lastspike = new double[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_lastspike[i] = 0;
	_array_synapses_N_incoming = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_synapses_N_incoming[i] = 0;
	_array_synapses_N_outgoing = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_synapses_N_outgoing[i] = 0;
	_array_neurongroup_not_refractory = new bool[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_not_refractory[i] = 0;
	_array_neurongroup_v = new double[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_v[i] = 0;
	_array_neurongroup_w = new double[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_w[i] = 0;

	// Arrays initialized to an "arange"
	_array_neurongroup_i = new int32_t[4000];
	for(int i=0; i<4000; i++) _array_neurongroup_i[i] = 0 + i;

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
}

void _dealloc_arrays()
{
	using namespace brian;

	if(_array_neurongroup__spikespace!=0)
	{
		delete [] _array_neurongroup__spikespace;
		_array_neurongroup__spikespace = 0;
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
	}
	if(_array_neurongroup_not_refractory!=0)
	{
		delete [] _array_neurongroup_not_refractory;
		_array_neurongroup_not_refractory = 0;
	}
	if(_array_neurongroup_v!=0)
	{
		delete [] _array_neurongroup_v;
		_array_neurongroup_v = 0;
	}
	if(_array_neurongroup_w!=0)
	{
		delete [] _array_neurongroup_w;
		_array_neurongroup_w = 0;
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

