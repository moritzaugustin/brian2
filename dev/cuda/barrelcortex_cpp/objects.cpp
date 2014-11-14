
#include<stdint.h>
#include<vector>
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "network.h"
#include<iostream>
#include<fstream>

//////////////// clocks ///////////////////
Clock brian::defaultclock(0.0001);
Clock brian::layer4_custom_operation_clock(0.06);

//////////////// networks /////////////////
Network brian::magicnetwork;

//////////////// arrays ///////////////////
int32_t * brian::_array_feedforward_N_incoming;
const int brian::_num__array_feedforward_N_incoming = 2500;
int32_t * brian::_array_feedforward_N_outgoing;
const int brian::_num__array_feedforward_N_outgoing = 1936;
int32_t * brian::_array_layer23__spikespace;
const int brian::_num__array_layer23__spikespace = 3077;
int32_t * brian::_array_layer23_barrel_idx;
const int brian::_num__array_layer23_barrel_idx = 3076;
double * brian::_array_layer23_ge;
const int brian::_num__array_layer23_ge = 3076;
double * brian::_array_layer23_gi;
const int brian::_num__array_layer23_gi = 3076;
int32_t * brian::_array_layer23_i;
const int brian::_num__array_layer23_i = 3076;
double * brian::_array_layer23_lastspike;
const int brian::_num__array_layer23_lastspike = 3076;
char * brian::_array_layer23_not_refractory;
const int brian::_num__array_layer23_not_refractory = 3076;
int32_t * brian::_array_layer23_subgroup_1__sub_idx;
const int brian::_num__array_layer23_subgroup_1__sub_idx = 576;
int32_t * brian::_array_layer23_subgroup__sub_idx;
const int brian::_num__array_layer23_subgroup__sub_idx = 2500;
double * brian::_array_layer23_v;
const int brian::_num__array_layer23_v = 3076;
double * brian::_array_layer23_vt;
const int brian::_num__array_layer23_vt = 3076;
double * brian::_array_layer23_x;
const int brian::_num__array_layer23_x = 3076;
double * brian::_array_layer23_y;
const int brian::_num__array_layer23_y = 3076;
int32_t * brian::_array_layer4__spikespace;
const int brian::_num__array_layer4__spikespace = 1937;
int32_t * brian::_array_layer4_barrel_x;
const int brian::_num__array_layer4_barrel_x = 1936;
int32_t * brian::_array_layer4_barrel_y;
const int brian::_num__array_layer4_barrel_y = 1936;
double * brian::_array_layer4_direction;
const int brian::_num__array_layer4_direction = 1;
int32_t * brian::_array_layer4_i;
const int brian::_num__array_layer4_i = 1936;
double * brian::_array_layer4_selectivity;
const int brian::_num__array_layer4_selectivity = 1936;
double * brian::_array_layer4_stim_start_time;
const int brian::_num__array_layer4_stim_start_time = 1;
double * brian::_array_layer4_stim_start_x;
const int brian::_num__array_layer4_stim_start_x = 1;
double * brian::_array_layer4_stim_start_y;
const int brian::_num__array_layer4_stim_start_y = 1;
int32_t * brian::_array_recurrent_exc_N_incoming;
const int brian::_num__array_recurrent_exc_N_incoming = 3076;
int32_t * brian::_array_recurrent_exc_N_outgoing;
const int brian::_num__array_recurrent_exc_N_outgoing = 2500;
int32_t * brian::_array_recurrent_inh_N_incoming;
const int brian::_num__array_recurrent_inh_N_incoming = 2500;
int32_t * brian::_array_recurrent_inh_N_outgoing;
const int brian::_num__array_recurrent_inh_N_outgoing = 3076;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> brian::_dynamic_array_feedforward__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_feedforward__synaptic_pre;
std::vector<double> brian::_dynamic_array_feedforward_A_source;
std::vector<double> brian::_dynamic_array_feedforward_A_target;
std::vector<double> brian::_dynamic_array_feedforward_lastupdate;
std::vector<double> brian::_dynamic_array_feedforward_post_delay;
std::vector<double> brian::_dynamic_array_feedforward_pre_delay;
std::vector<double> brian::_dynamic_array_feedforward_w;
std::vector<int32_t> brian::_dynamic_array_recurrent_exc__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_recurrent_exc__synaptic_pre;
std::vector<double> brian::_dynamic_array_recurrent_exc_lastupdate;
std::vector<double> brian::_dynamic_array_recurrent_exc_pre_delay;
std::vector<double> brian::_dynamic_array_recurrent_exc_w;
std::vector<int32_t> brian::_dynamic_array_recurrent_inh__synaptic_post;
std::vector<int32_t> brian::_dynamic_array_recurrent_inh__synaptic_pre;
std::vector<double> brian::_dynamic_array_recurrent_inh_lastupdate;
std::vector<double> brian::_dynamic_array_recurrent_inh_pre_delay;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
double * brian::_static_array__array_layer23_lastspike;
const int brian::_num__static_array__array_layer23_lastspike = 3076;
char * brian::_static_array__array_layer23_not_refractory;
const int brian::_num__static_array__array_layer23_not_refractory = 3076;

//////////////// synapses /////////////////
// feedforward
Synapses<double> brian::feedforward(1936, 2500);
SynapticPathway<double> brian::feedforward_post(
		2500, 1936,
		_dynamic_array_feedforward_post_delay,
		_dynamic_array_feedforward__synaptic_post,
		0.0001,
		0, 2500);
SynapticPathway<double> brian::feedforward_pre(
		1936, 2500,
		_dynamic_array_feedforward_pre_delay,
		_dynamic_array_feedforward__synaptic_pre,
		0.0001,
		0, 1936);
// recurrent_exc
Synapses<double> brian::recurrent_exc(2500, 3076);
SynapticPathway<double> brian::recurrent_exc_pre(
		2500, 3076,
		_dynamic_array_recurrent_exc_pre_delay,
		_dynamic_array_recurrent_exc__synaptic_pre,
		0.0001,
		0, 2500);
// recurrent_inh
Synapses<double> brian::recurrent_inh(576, 2500);
SynapticPathway<double> brian::recurrent_inh_pre(
		576, 2500,
		_dynamic_array_recurrent_inh_pre_delay,
		_dynamic_array_recurrent_inh__synaptic_pre,
		0.0001,
		2500, 3076);


void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_layer4__spikespace = new int32_t[1937];
	
	for(int i=0; i<1937; i++) _array_layer4__spikespace[i] = 0;
	_array_layer23__spikespace = new int32_t[3077];
	
	for(int i=0; i<3077; i++) _array_layer23__spikespace[i] = 0;
	_array_layer23_subgroup__sub_idx = new int32_t[2500];
	
	for(int i=0; i<2500; i++) _array_layer23_subgroup__sub_idx[i] = 0;
	_array_layer23_subgroup_1__sub_idx = new int32_t[576];
	
	for(int i=0; i<576; i++) _array_layer23_subgroup_1__sub_idx[i] = 0;
	_array_layer23_barrel_idx = new int32_t[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_barrel_idx[i] = 0;
	_array_layer4_barrel_x = new int32_t[1936];
	
	for(int i=0; i<1936; i++) _array_layer4_barrel_x[i] = 0;
	_array_layer4_barrel_y = new int32_t[1936];
	
	for(int i=0; i<1936; i++) _array_layer4_barrel_y[i] = 0;
	_array_layer4_direction = new double[1];
	
	for(int i=0; i<1; i++) _array_layer4_direction[i] = 0;
	_array_layer23_ge = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_ge[i] = 0;
	_array_layer23_gi = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_gi[i] = 0;
	_array_layer4_i = new int32_t[1936];
	
	for(int i=0; i<1936; i++) _array_layer4_i[i] = 0;
	_array_layer23_i = new int32_t[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_i[i] = 0;
	_array_layer23_lastspike = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_lastspike[i] = 0;
	_array_feedforward_N_incoming = new int32_t[2500];
	
	for(int i=0; i<2500; i++) _array_feedforward_N_incoming[i] = 0;
	_array_recurrent_exc_N_incoming = new int32_t[3076];
	
	for(int i=0; i<3076; i++) _array_recurrent_exc_N_incoming[i] = 0;
	_array_recurrent_inh_N_incoming = new int32_t[2500];
	
	for(int i=0; i<2500; i++) _array_recurrent_inh_N_incoming[i] = 0;
	_array_feedforward_N_outgoing = new int32_t[1936];
	
	for(int i=0; i<1936; i++) _array_feedforward_N_outgoing[i] = 0;
	_array_recurrent_exc_N_outgoing = new int32_t[2500];
	
	for(int i=0; i<2500; i++) _array_recurrent_exc_N_outgoing[i] = 0;
	_array_recurrent_inh_N_outgoing = new int32_t[3076];
	
	for(int i=0; i<3076; i++) _array_recurrent_inh_N_outgoing[i] = 0;
	_array_layer23_not_refractory = new char[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_not_refractory[i] = 0;
	_array_layer4_selectivity = new double[1936];
	
	for(int i=0; i<1936; i++) _array_layer4_selectivity[i] = 0;
	_array_layer4_stim_start_time = new double[1];
	
	for(int i=0; i<1; i++) _array_layer4_stim_start_time[i] = 0;
	_array_layer4_stim_start_x = new double[1];
	
	for(int i=0; i<1; i++) _array_layer4_stim_start_x[i] = 0;
	_array_layer4_stim_start_y = new double[1];
	
	for(int i=0; i<1; i++) _array_layer4_stim_start_y[i] = 0;
	_array_layer23_v = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_v[i] = 0;
	_array_layer23_vt = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_vt[i] = 0;
	_array_layer23_x = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_x[i] = 0;
	_array_layer23_y = new double[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_y[i] = 0;

	// Arrays initialized to an "arange"
	_array_layer23_subgroup_1__sub_idx = new int32_t[576];
	
	for(int i=0; i<576; i++) _array_layer23_subgroup_1__sub_idx[i] = 2500 + i;
	_array_layer23_subgroup__sub_idx = new int32_t[2500];
	
	for(int i=0; i<2500; i++) _array_layer23_subgroup__sub_idx[i] = 0 + i;
	_array_layer4_i = new int32_t[1936];
	
	for(int i=0; i<1936; i++) _array_layer4_i[i] = 0 + i;
	_array_layer23_i = new int32_t[3076];
	
	for(int i=0; i<3076; i++) _array_layer23_i[i] = 0 + i;

	// static arrays
	_static_array__array_layer23_lastspike = new double[3076];
	_static_array__array_layer23_not_refractory = new char[3076];
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_layer23_lastspike;
	f_static_array__array_layer23_lastspike.open("static_arrays/_static_array__array_layer23_lastspike", ios::in | ios::binary);
	if(f_static_array__array_layer23_lastspike.is_open())
	{
		f_static_array__array_layer23_lastspike.read(reinterpret_cast<char*>(_static_array__array_layer23_lastspike), 3076*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_layer23_lastspike." << endl;
	}
	ifstream f_static_array__array_layer23_not_refractory;
	f_static_array__array_layer23_not_refractory.open("static_arrays/_static_array__array_layer23_not_refractory", ios::in | ios::binary);
	if(f_static_array__array_layer23_not_refractory.is_open())
	{
		f_static_array__array_layer23_not_refractory.read(reinterpret_cast<char*>(_static_array__array_layer23_not_refractory), 3076*sizeof(char));
	} else
	{
		std::cout << "Error opening static array _static_array__array_layer23_not_refractory." << endl;
	}
}	

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_feedforward_N_incoming;
	outfile__array_feedforward_N_incoming.open("results/_array_feedforward_N_incoming", ios::binary | ios::out);
	if(outfile__array_feedforward_N_incoming.is_open())
	{
		outfile__array_feedforward_N_incoming.write(reinterpret_cast<char*>(_array_feedforward_N_incoming), 2500*sizeof(_array_feedforward_N_incoming[0]));
		outfile__array_feedforward_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _array_feedforward_N_incoming." << endl;
	}
	ofstream outfile__array_feedforward_N_outgoing;
	outfile__array_feedforward_N_outgoing.open("results/_array_feedforward_N_outgoing", ios::binary | ios::out);
	if(outfile__array_feedforward_N_outgoing.is_open())
	{
		outfile__array_feedforward_N_outgoing.write(reinterpret_cast<char*>(_array_feedforward_N_outgoing), 1936*sizeof(_array_feedforward_N_outgoing[0]));
		outfile__array_feedforward_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _array_feedforward_N_outgoing." << endl;
	}
	ofstream outfile__array_layer23__spikespace;
	outfile__array_layer23__spikespace.open("results/_array_layer23__spikespace", ios::binary | ios::out);
	if(outfile__array_layer23__spikespace.is_open())
	{
		outfile__array_layer23__spikespace.write(reinterpret_cast<char*>(_array_layer23__spikespace), 3077*sizeof(_array_layer23__spikespace[0]));
		outfile__array_layer23__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23__spikespace." << endl;
	}
	ofstream outfile__array_layer23_barrel_idx;
	outfile__array_layer23_barrel_idx.open("results/_array_layer23_barrel_idx", ios::binary | ios::out);
	if(outfile__array_layer23_barrel_idx.is_open())
	{
		outfile__array_layer23_barrel_idx.write(reinterpret_cast<char*>(_array_layer23_barrel_idx), 3076*sizeof(_array_layer23_barrel_idx[0]));
		outfile__array_layer23_barrel_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_barrel_idx." << endl;
	}
	ofstream outfile__array_layer23_ge;
	outfile__array_layer23_ge.open("results/_array_layer23_ge", ios::binary | ios::out);
	if(outfile__array_layer23_ge.is_open())
	{
		outfile__array_layer23_ge.write(reinterpret_cast<char*>(_array_layer23_ge), 3076*sizeof(_array_layer23_ge[0]));
		outfile__array_layer23_ge.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_ge." << endl;
	}
	ofstream outfile__array_layer23_gi;
	outfile__array_layer23_gi.open("results/_array_layer23_gi", ios::binary | ios::out);
	if(outfile__array_layer23_gi.is_open())
	{
		outfile__array_layer23_gi.write(reinterpret_cast<char*>(_array_layer23_gi), 3076*sizeof(_array_layer23_gi[0]));
		outfile__array_layer23_gi.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_gi." << endl;
	}
	ofstream outfile__array_layer23_i;
	outfile__array_layer23_i.open("results/_array_layer23_i", ios::binary | ios::out);
	if(outfile__array_layer23_i.is_open())
	{
		outfile__array_layer23_i.write(reinterpret_cast<char*>(_array_layer23_i), 3076*sizeof(_array_layer23_i[0]));
		outfile__array_layer23_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_i." << endl;
	}
	ofstream outfile__array_layer23_lastspike;
	outfile__array_layer23_lastspike.open("results/_array_layer23_lastspike", ios::binary | ios::out);
	if(outfile__array_layer23_lastspike.is_open())
	{
		outfile__array_layer23_lastspike.write(reinterpret_cast<char*>(_array_layer23_lastspike), 3076*sizeof(_array_layer23_lastspike[0]));
		outfile__array_layer23_lastspike.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_lastspike." << endl;
	}
	ofstream outfile__array_layer23_not_refractory;
	outfile__array_layer23_not_refractory.open("results/_array_layer23_not_refractory", ios::binary | ios::out);
	if(outfile__array_layer23_not_refractory.is_open())
	{
		outfile__array_layer23_not_refractory.write(reinterpret_cast<char*>(_array_layer23_not_refractory), 3076*sizeof(_array_layer23_not_refractory[0]));
		outfile__array_layer23_not_refractory.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_not_refractory." << endl;
	}
	ofstream outfile__array_layer23_subgroup_1__sub_idx;
	outfile__array_layer23_subgroup_1__sub_idx.open("results/_array_layer23_subgroup_1__sub_idx", ios::binary | ios::out);
	if(outfile__array_layer23_subgroup_1__sub_idx.is_open())
	{
		outfile__array_layer23_subgroup_1__sub_idx.write(reinterpret_cast<char*>(_array_layer23_subgroup_1__sub_idx), 576*sizeof(_array_layer23_subgroup_1__sub_idx[0]));
		outfile__array_layer23_subgroup_1__sub_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_subgroup_1__sub_idx." << endl;
	}
	ofstream outfile__array_layer23_subgroup__sub_idx;
	outfile__array_layer23_subgroup__sub_idx.open("results/_array_layer23_subgroup__sub_idx", ios::binary | ios::out);
	if(outfile__array_layer23_subgroup__sub_idx.is_open())
	{
		outfile__array_layer23_subgroup__sub_idx.write(reinterpret_cast<char*>(_array_layer23_subgroup__sub_idx), 2500*sizeof(_array_layer23_subgroup__sub_idx[0]));
		outfile__array_layer23_subgroup__sub_idx.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_subgroup__sub_idx." << endl;
	}
	ofstream outfile__array_layer23_v;
	outfile__array_layer23_v.open("results/_array_layer23_v", ios::binary | ios::out);
	if(outfile__array_layer23_v.is_open())
	{
		outfile__array_layer23_v.write(reinterpret_cast<char*>(_array_layer23_v), 3076*sizeof(_array_layer23_v[0]));
		outfile__array_layer23_v.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_v." << endl;
	}
	ofstream outfile__array_layer23_vt;
	outfile__array_layer23_vt.open("results/_array_layer23_vt", ios::binary | ios::out);
	if(outfile__array_layer23_vt.is_open())
	{
		outfile__array_layer23_vt.write(reinterpret_cast<char*>(_array_layer23_vt), 3076*sizeof(_array_layer23_vt[0]));
		outfile__array_layer23_vt.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_vt." << endl;
	}
	ofstream outfile__array_layer23_x;
	outfile__array_layer23_x.open("results/_array_layer23_x", ios::binary | ios::out);
	if(outfile__array_layer23_x.is_open())
	{
		outfile__array_layer23_x.write(reinterpret_cast<char*>(_array_layer23_x), 3076*sizeof(_array_layer23_x[0]));
		outfile__array_layer23_x.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_x." << endl;
	}
	ofstream outfile__array_layer23_y;
	outfile__array_layer23_y.open("results/_array_layer23_y", ios::binary | ios::out);
	if(outfile__array_layer23_y.is_open())
	{
		outfile__array_layer23_y.write(reinterpret_cast<char*>(_array_layer23_y), 3076*sizeof(_array_layer23_y[0]));
		outfile__array_layer23_y.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer23_y." << endl;
	}
	ofstream outfile__array_layer4__spikespace;
	outfile__array_layer4__spikespace.open("results/_array_layer4__spikespace", ios::binary | ios::out);
	if(outfile__array_layer4__spikespace.is_open())
	{
		outfile__array_layer4__spikespace.write(reinterpret_cast<char*>(_array_layer4__spikespace), 1937*sizeof(_array_layer4__spikespace[0]));
		outfile__array_layer4__spikespace.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4__spikespace." << endl;
	}
	ofstream outfile__array_layer4_barrel_x;
	outfile__array_layer4_barrel_x.open("results/_array_layer4_barrel_x", ios::binary | ios::out);
	if(outfile__array_layer4_barrel_x.is_open())
	{
		outfile__array_layer4_barrel_x.write(reinterpret_cast<char*>(_array_layer4_barrel_x), 1936*sizeof(_array_layer4_barrel_x[0]));
		outfile__array_layer4_barrel_x.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_barrel_x." << endl;
	}
	ofstream outfile__array_layer4_barrel_y;
	outfile__array_layer4_barrel_y.open("results/_array_layer4_barrel_y", ios::binary | ios::out);
	if(outfile__array_layer4_barrel_y.is_open())
	{
		outfile__array_layer4_barrel_y.write(reinterpret_cast<char*>(_array_layer4_barrel_y), 1936*sizeof(_array_layer4_barrel_y[0]));
		outfile__array_layer4_barrel_y.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_barrel_y." << endl;
	}
	ofstream outfile__array_layer4_direction;
	outfile__array_layer4_direction.open("results/_array_layer4_direction", ios::binary | ios::out);
	if(outfile__array_layer4_direction.is_open())
	{
		outfile__array_layer4_direction.write(reinterpret_cast<char*>(_array_layer4_direction), 1*sizeof(_array_layer4_direction[0]));
		outfile__array_layer4_direction.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_direction." << endl;
	}
	ofstream outfile__array_layer4_i;
	outfile__array_layer4_i.open("results/_array_layer4_i", ios::binary | ios::out);
	if(outfile__array_layer4_i.is_open())
	{
		outfile__array_layer4_i.write(reinterpret_cast<char*>(_array_layer4_i), 1936*sizeof(_array_layer4_i[0]));
		outfile__array_layer4_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_i." << endl;
	}
	ofstream outfile__array_layer4_selectivity;
	outfile__array_layer4_selectivity.open("results/_array_layer4_selectivity", ios::binary | ios::out);
	if(outfile__array_layer4_selectivity.is_open())
	{
		outfile__array_layer4_selectivity.write(reinterpret_cast<char*>(_array_layer4_selectivity), 1936*sizeof(_array_layer4_selectivity[0]));
		outfile__array_layer4_selectivity.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_selectivity." << endl;
	}
	ofstream outfile__array_layer4_stim_start_time;
	outfile__array_layer4_stim_start_time.open("results/_array_layer4_stim_start_time", ios::binary | ios::out);
	if(outfile__array_layer4_stim_start_time.is_open())
	{
		outfile__array_layer4_stim_start_time.write(reinterpret_cast<char*>(_array_layer4_stim_start_time), 1*sizeof(_array_layer4_stim_start_time[0]));
		outfile__array_layer4_stim_start_time.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_stim_start_time." << endl;
	}
	ofstream outfile__array_layer4_stim_start_x;
	outfile__array_layer4_stim_start_x.open("results/_array_layer4_stim_start_x", ios::binary | ios::out);
	if(outfile__array_layer4_stim_start_x.is_open())
	{
		outfile__array_layer4_stim_start_x.write(reinterpret_cast<char*>(_array_layer4_stim_start_x), 1*sizeof(_array_layer4_stim_start_x[0]));
		outfile__array_layer4_stim_start_x.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_stim_start_x." << endl;
	}
	ofstream outfile__array_layer4_stim_start_y;
	outfile__array_layer4_stim_start_y.open("results/_array_layer4_stim_start_y", ios::binary | ios::out);
	if(outfile__array_layer4_stim_start_y.is_open())
	{
		outfile__array_layer4_stim_start_y.write(reinterpret_cast<char*>(_array_layer4_stim_start_y), 1*sizeof(_array_layer4_stim_start_y[0]));
		outfile__array_layer4_stim_start_y.close();
	} else
	{
		std::cout << "Error writing output file for _array_layer4_stim_start_y." << endl;
	}
	ofstream outfile__array_recurrent_exc_N_incoming;
	outfile__array_recurrent_exc_N_incoming.open("results/_array_recurrent_exc_N_incoming", ios::binary | ios::out);
	if(outfile__array_recurrent_exc_N_incoming.is_open())
	{
		outfile__array_recurrent_exc_N_incoming.write(reinterpret_cast<char*>(_array_recurrent_exc_N_incoming), 3076*sizeof(_array_recurrent_exc_N_incoming[0]));
		outfile__array_recurrent_exc_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _array_recurrent_exc_N_incoming." << endl;
	}
	ofstream outfile__array_recurrent_exc_N_outgoing;
	outfile__array_recurrent_exc_N_outgoing.open("results/_array_recurrent_exc_N_outgoing", ios::binary | ios::out);
	if(outfile__array_recurrent_exc_N_outgoing.is_open())
	{
		outfile__array_recurrent_exc_N_outgoing.write(reinterpret_cast<char*>(_array_recurrent_exc_N_outgoing), 2500*sizeof(_array_recurrent_exc_N_outgoing[0]));
		outfile__array_recurrent_exc_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _array_recurrent_exc_N_outgoing." << endl;
	}
	ofstream outfile__array_recurrent_inh_N_incoming;
	outfile__array_recurrent_inh_N_incoming.open("results/_array_recurrent_inh_N_incoming", ios::binary | ios::out);
	if(outfile__array_recurrent_inh_N_incoming.is_open())
	{
		outfile__array_recurrent_inh_N_incoming.write(reinterpret_cast<char*>(_array_recurrent_inh_N_incoming), 2500*sizeof(_array_recurrent_inh_N_incoming[0]));
		outfile__array_recurrent_inh_N_incoming.close();
	} else
	{
		std::cout << "Error writing output file for _array_recurrent_inh_N_incoming." << endl;
	}
	ofstream outfile__array_recurrent_inh_N_outgoing;
	outfile__array_recurrent_inh_N_outgoing.open("results/_array_recurrent_inh_N_outgoing", ios::binary | ios::out);
	if(outfile__array_recurrent_inh_N_outgoing.is_open())
	{
		outfile__array_recurrent_inh_N_outgoing.write(reinterpret_cast<char*>(_array_recurrent_inh_N_outgoing), 3076*sizeof(_array_recurrent_inh_N_outgoing[0]));
		outfile__array_recurrent_inh_N_outgoing.close();
	} else
	{
		std::cout << "Error writing output file for _array_recurrent_inh_N_outgoing." << endl;
	}

	ofstream outfile__dynamic_array_feedforward__synaptic_post;
	outfile__dynamic_array_feedforward__synaptic_post.open("results/_dynamic_array_feedforward__synaptic_post", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward__synaptic_post.is_open())
	{
		outfile__dynamic_array_feedforward__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_feedforward__synaptic_post[0]), _dynamic_array_feedforward__synaptic_post.size()*sizeof(_dynamic_array_feedforward__synaptic_post[0]));
		outfile__dynamic_array_feedforward__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_feedforward__synaptic_pre;
	outfile__dynamic_array_feedforward__synaptic_pre.open("results/_dynamic_array_feedforward__synaptic_pre", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward__synaptic_pre.is_open())
	{
		outfile__dynamic_array_feedforward__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_feedforward__synaptic_pre[0]), _dynamic_array_feedforward__synaptic_pre.size()*sizeof(_dynamic_array_feedforward__synaptic_pre[0]));
		outfile__dynamic_array_feedforward__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_feedforward_A_source;
	outfile__dynamic_array_feedforward_A_source.open("results/_dynamic_array_feedforward_A_source", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward_A_source.is_open())
	{
		outfile__dynamic_array_feedforward_A_source.write(reinterpret_cast<char*>(&_dynamic_array_feedforward_A_source[0]), _dynamic_array_feedforward_A_source.size()*sizeof(_dynamic_array_feedforward_A_source[0]));
		outfile__dynamic_array_feedforward_A_source.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward_A_source." << endl;
	}
	ofstream outfile__dynamic_array_feedforward_A_target;
	outfile__dynamic_array_feedforward_A_target.open("results/_dynamic_array_feedforward_A_target", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward_A_target.is_open())
	{
		outfile__dynamic_array_feedforward_A_target.write(reinterpret_cast<char*>(&_dynamic_array_feedforward_A_target[0]), _dynamic_array_feedforward_A_target.size()*sizeof(_dynamic_array_feedforward_A_target[0]));
		outfile__dynamic_array_feedforward_A_target.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward_A_target." << endl;
	}
	ofstream outfile__dynamic_array_feedforward_lastupdate;
	outfile__dynamic_array_feedforward_lastupdate.open("results/_dynamic_array_feedforward_lastupdate", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward_lastupdate.is_open())
	{
		outfile__dynamic_array_feedforward_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_feedforward_lastupdate[0]), _dynamic_array_feedforward_lastupdate.size()*sizeof(_dynamic_array_feedforward_lastupdate[0]));
		outfile__dynamic_array_feedforward_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_feedforward_post_delay;
	outfile__dynamic_array_feedforward_post_delay.open("results/_dynamic_array_feedforward_post_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward_post_delay.is_open())
	{
		outfile__dynamic_array_feedforward_post_delay.write(reinterpret_cast<char*>(&_dynamic_array_feedforward_post_delay[0]), _dynamic_array_feedforward_post_delay.size()*sizeof(_dynamic_array_feedforward_post_delay[0]));
		outfile__dynamic_array_feedforward_post_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward_post_delay." << endl;
	}
	ofstream outfile__dynamic_array_feedforward_pre_delay;
	outfile__dynamic_array_feedforward_pre_delay.open("results/_dynamic_array_feedforward_pre_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward_pre_delay.is_open())
	{
		outfile__dynamic_array_feedforward_pre_delay.write(reinterpret_cast<char*>(&_dynamic_array_feedforward_pre_delay[0]), _dynamic_array_feedforward_pre_delay.size()*sizeof(_dynamic_array_feedforward_pre_delay[0]));
		outfile__dynamic_array_feedforward_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward_pre_delay." << endl;
	}
	ofstream outfile__dynamic_array_feedforward_w;
	outfile__dynamic_array_feedforward_w.open("results/_dynamic_array_feedforward_w", ios::binary | ios::out);
	if(outfile__dynamic_array_feedforward_w.is_open())
	{
		outfile__dynamic_array_feedforward_w.write(reinterpret_cast<char*>(&_dynamic_array_feedforward_w[0]), _dynamic_array_feedforward_w.size()*sizeof(_dynamic_array_feedforward_w[0]));
		outfile__dynamic_array_feedforward_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_feedforward_w." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_exc__synaptic_post;
	outfile__dynamic_array_recurrent_exc__synaptic_post.open("results/_dynamic_array_recurrent_exc__synaptic_post", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_exc__synaptic_post.is_open())
	{
		outfile__dynamic_array_recurrent_exc__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_exc__synaptic_post[0]), _dynamic_array_recurrent_exc__synaptic_post.size()*sizeof(_dynamic_array_recurrent_exc__synaptic_post[0]));
		outfile__dynamic_array_recurrent_exc__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_exc__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_exc__synaptic_pre;
	outfile__dynamic_array_recurrent_exc__synaptic_pre.open("results/_dynamic_array_recurrent_exc__synaptic_pre", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_exc__synaptic_pre.is_open())
	{
		outfile__dynamic_array_recurrent_exc__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_exc__synaptic_pre[0]), _dynamic_array_recurrent_exc__synaptic_pre.size()*sizeof(_dynamic_array_recurrent_exc__synaptic_pre[0]));
		outfile__dynamic_array_recurrent_exc__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_exc__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_exc_lastupdate;
	outfile__dynamic_array_recurrent_exc_lastupdate.open("results/_dynamic_array_recurrent_exc_lastupdate", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_exc_lastupdate.is_open())
	{
		outfile__dynamic_array_recurrent_exc_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_exc_lastupdate[0]), _dynamic_array_recurrent_exc_lastupdate.size()*sizeof(_dynamic_array_recurrent_exc_lastupdate[0]));
		outfile__dynamic_array_recurrent_exc_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_exc_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_exc_pre_delay;
	outfile__dynamic_array_recurrent_exc_pre_delay.open("results/_dynamic_array_recurrent_exc_pre_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_exc_pre_delay.is_open())
	{
		outfile__dynamic_array_recurrent_exc_pre_delay.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_exc_pre_delay[0]), _dynamic_array_recurrent_exc_pre_delay.size()*sizeof(_dynamic_array_recurrent_exc_pre_delay[0]));
		outfile__dynamic_array_recurrent_exc_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_exc_pre_delay." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_exc_w;
	outfile__dynamic_array_recurrent_exc_w.open("results/_dynamic_array_recurrent_exc_w", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_exc_w.is_open())
	{
		outfile__dynamic_array_recurrent_exc_w.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_exc_w[0]), _dynamic_array_recurrent_exc_w.size()*sizeof(_dynamic_array_recurrent_exc_w[0]));
		outfile__dynamic_array_recurrent_exc_w.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_exc_w." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_inh__synaptic_post;
	outfile__dynamic_array_recurrent_inh__synaptic_post.open("results/_dynamic_array_recurrent_inh__synaptic_post", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_inh__synaptic_post.is_open())
	{
		outfile__dynamic_array_recurrent_inh__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_inh__synaptic_post[0]), _dynamic_array_recurrent_inh__synaptic_post.size()*sizeof(_dynamic_array_recurrent_inh__synaptic_post[0]));
		outfile__dynamic_array_recurrent_inh__synaptic_post.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_inh__synaptic_post." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_inh__synaptic_pre;
	outfile__dynamic_array_recurrent_inh__synaptic_pre.open("results/_dynamic_array_recurrent_inh__synaptic_pre", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_inh__synaptic_pre.is_open())
	{
		outfile__dynamic_array_recurrent_inh__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_inh__synaptic_pre[0]), _dynamic_array_recurrent_inh__synaptic_pre.size()*sizeof(_dynamic_array_recurrent_inh__synaptic_pre[0]));
		outfile__dynamic_array_recurrent_inh__synaptic_pre.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_inh__synaptic_pre." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_inh_lastupdate;
	outfile__dynamic_array_recurrent_inh_lastupdate.open("results/_dynamic_array_recurrent_inh_lastupdate", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_inh_lastupdate.is_open())
	{
		outfile__dynamic_array_recurrent_inh_lastupdate.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_inh_lastupdate[0]), _dynamic_array_recurrent_inh_lastupdate.size()*sizeof(_dynamic_array_recurrent_inh_lastupdate[0]));
		outfile__dynamic_array_recurrent_inh_lastupdate.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_inh_lastupdate." << endl;
	}
	ofstream outfile__dynamic_array_recurrent_inh_pre_delay;
	outfile__dynamic_array_recurrent_inh_pre_delay.open("results/_dynamic_array_recurrent_inh_pre_delay", ios::binary | ios::out);
	if(outfile__dynamic_array_recurrent_inh_pre_delay.is_open())
	{
		outfile__dynamic_array_recurrent_inh_pre_delay.write(reinterpret_cast<char*>(&_dynamic_array_recurrent_inh_pre_delay[0]), _dynamic_array_recurrent_inh_pre_delay.size()*sizeof(_dynamic_array_recurrent_inh_pre_delay[0]));
		outfile__dynamic_array_recurrent_inh_pre_delay.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_recurrent_inh_pre_delay." << endl;
	}

}

void _dealloc_arrays()
{
	using namespace brian;

	if(_array_feedforward_N_incoming!=0)
	{
		delete [] _array_feedforward_N_incoming;
		_array_feedforward_N_incoming = 0;
	}
	if(_array_feedforward_N_outgoing!=0)
	{
		delete [] _array_feedforward_N_outgoing;
		_array_feedforward_N_outgoing = 0;
	}
	if(_array_layer23__spikespace!=0)
	{
		delete [] _array_layer23__spikespace;
		_array_layer23__spikespace = 0;
	}
	if(_array_layer23_barrel_idx!=0)
	{
		delete [] _array_layer23_barrel_idx;
		_array_layer23_barrel_idx = 0;
	}
	if(_array_layer23_ge!=0)
	{
		delete [] _array_layer23_ge;
		_array_layer23_ge = 0;
	}
	if(_array_layer23_gi!=0)
	{
		delete [] _array_layer23_gi;
		_array_layer23_gi = 0;
	}
	if(_array_layer23_i!=0)
	{
		delete [] _array_layer23_i;
		_array_layer23_i = 0;
	}
	if(_array_layer23_lastspike!=0)
	{
		delete [] _array_layer23_lastspike;
		_array_layer23_lastspike = 0;
	}
	if(_array_layer23_not_refractory!=0)
	{
		delete [] _array_layer23_not_refractory;
		_array_layer23_not_refractory = 0;
	}
	if(_array_layer23_subgroup_1__sub_idx!=0)
	{
		delete [] _array_layer23_subgroup_1__sub_idx;
		_array_layer23_subgroup_1__sub_idx = 0;
	}
	if(_array_layer23_subgroup__sub_idx!=0)
	{
		delete [] _array_layer23_subgroup__sub_idx;
		_array_layer23_subgroup__sub_idx = 0;
	}
	if(_array_layer23_v!=0)
	{
		delete [] _array_layer23_v;
		_array_layer23_v = 0;
	}
	if(_array_layer23_vt!=0)
	{
		delete [] _array_layer23_vt;
		_array_layer23_vt = 0;
	}
	if(_array_layer23_x!=0)
	{
		delete [] _array_layer23_x;
		_array_layer23_x = 0;
	}
	if(_array_layer23_y!=0)
	{
		delete [] _array_layer23_y;
		_array_layer23_y = 0;
	}
	if(_array_layer4__spikespace!=0)
	{
		delete [] _array_layer4__spikespace;
		_array_layer4__spikespace = 0;
	}
	if(_array_layer4_barrel_x!=0)
	{
		delete [] _array_layer4_barrel_x;
		_array_layer4_barrel_x = 0;
	}
	if(_array_layer4_barrel_y!=0)
	{
		delete [] _array_layer4_barrel_y;
		_array_layer4_barrel_y = 0;
	}
	if(_array_layer4_direction!=0)
	{
		delete [] _array_layer4_direction;
		_array_layer4_direction = 0;
	}
	if(_array_layer4_i!=0)
	{
		delete [] _array_layer4_i;
		_array_layer4_i = 0;
	}
	if(_array_layer4_selectivity!=0)
	{
		delete [] _array_layer4_selectivity;
		_array_layer4_selectivity = 0;
	}
	if(_array_layer4_stim_start_time!=0)
	{
		delete [] _array_layer4_stim_start_time;
		_array_layer4_stim_start_time = 0;
	}
	if(_array_layer4_stim_start_x!=0)
	{
		delete [] _array_layer4_stim_start_x;
		_array_layer4_stim_start_x = 0;
	}
	if(_array_layer4_stim_start_y!=0)
	{
		delete [] _array_layer4_stim_start_y;
		_array_layer4_stim_start_y = 0;
	}
	if(_array_recurrent_exc_N_incoming!=0)
	{
		delete [] _array_recurrent_exc_N_incoming;
		_array_recurrent_exc_N_incoming = 0;
	}
	if(_array_recurrent_exc_N_outgoing!=0)
	{
		delete [] _array_recurrent_exc_N_outgoing;
		_array_recurrent_exc_N_outgoing = 0;
	}
	if(_array_recurrent_inh_N_incoming!=0)
	{
		delete [] _array_recurrent_inh_N_incoming;
		_array_recurrent_inh_N_incoming = 0;
	}
	if(_array_recurrent_inh_N_outgoing!=0)
	{
		delete [] _array_recurrent_inh_N_outgoing;
		_array_recurrent_inh_N_outgoing = 0;
	}

	// static arrays
	if(_static_array__array_layer23_lastspike!=0)
	{
		delete [] _static_array__array_layer23_lastspike;
		_static_array__array_layer23_lastspike = 0;
	}
	if(_static_array__array_layer23_not_refractory!=0)
	{
		delete [] _static_array__array_layer23_not_refractory;
		_static_array__array_layer23_not_refractory = 0;
	}
}

