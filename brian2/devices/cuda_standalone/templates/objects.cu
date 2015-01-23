{% macro cu_file() %}

#include<stdint.h>
#include<vector>
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "network.h"
#include<iostream>
#include<fstream>
#include <thrust/device_vector.h>

//////////////// clocks ///////////////////
{% for clock in clocks | sort(attribute='name') %}
Clock brian::{{clock.name}}({{clock.dt_}});
{% endfor %}

//////////////// networks /////////////////
{% for net in networks | sort(attribute='name') %}
Network brian::{{net.name}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
{{c_data_type(var.dtype)}} * brian::{{varname}};
{{c_data_type(var.dtype)}} * brian::dev{{varname}};
const int brian::_num_{{varname}} = {{var.size}};

{% endif %}
{% endfor %}

//////////////// dynamic arrays 1d /////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
std::vector<{{c_data_type(var.dtype)}}> brian::{{varname}};
thrust::device_vector<{{c_data_type(var.dtype)}}> brian::dev{{varname}};
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
DynamicArray2D<{{c_data_type(var.dtype)}}> brian::{{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not name in array_specs.values() %}
{{dtype_spec}} * brian::{{name}};
{{dtype_spec}} * brian::dev{{name}};
const int brian::_num_{{name}} = {{N}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
Synapses<double> brian::{{S.name}}({{S.source|length}}, {{S.target|length}});
{% for path in S._pathways | sort(attribute='name') %}
SynapticPathway<double> brian::{{path.name}}(
		{{path.source|length}}, {{path.target|length}},
		{{dynamic_array_specs[path.variables['delay']]}},
		{{dynamic_array_specs[path.synapse_sources]}},
		{{path.source.dt_}},
		{{path.source.start}}, {{path.source.stop}});
{% endfor %}
{% endfor %}

unsigned int brian::num_cuda_processors;
unsigned int brian::max_threads_per_block;
unsigned int brian::max_shared_mem_size;


void _init_arrays()
{
	using namespace brian;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	num_cuda_processors = props.multiProcessorCount;
	max_threads_per_block = props.maxThreadsPerBlock;
	max_shared_mem_size = props.sharedMemPerBlock;

    // Arrays initialized to 0
	{% for var in zero_arrays | sort(attribute='name') %}
	{% set varname = array_specs[var] %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;
	cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	if(!dev{{varname}})
	{
		printf("ERROR while allocating device memory with size %ld\n", sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	}
	cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice);

	{% endfor %}

	// Arrays initialized to an "arange"
	{% for var, start in arange_arrays %}
	{% set varname = array_specs[var] %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = {{start}} + i;
	cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	if(!dev{{varname}})
	{
		printf("ERROR while allocating device memory with size %ld\n", sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	}
	cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice);

	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{{name}} = new {{dtype_spec}}[{{N}}];
	cudaMalloc((void**)&dev{{name}}, sizeof({{dtype_spec}})*{{N}});
	if(!dev{{name}})
	{
		printf("ERROR while allocating device memory with size %ld\n", sizeof({{dtype_spec}})*{{N}});
	}

	{% endfor %}
}

void _load_arrays()
{
	using namespace brian;

	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	ifstream f{{name}};
	f{{name}}.open("static_arrays/{{name}}", ios::in | ios::binary);
	if(f{{name}}.is_open())
	{
		f{{name}}.read(reinterpret_cast<char*>({{name}}), {{N}}*sizeof({{dtype_spec}}));
	} else
	{
		std::cout << "Error opening static array {{name}}." << endl;
	}
	{% endfor %}
}	

void _write_arrays()
{
	using namespace brian;

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs) %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}), {{var.size}}*sizeof({{varname}}[0]));
		outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}[0]), {{varname}}.size()*sizeof({{varname}}[0]));
		outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
        for (int n=0; n<{{varname}}.n; n++)
        {
            outfile_{{varname}}.write(reinterpret_cast<char*>(&{{varname}}(n, 0)), {{varname}}.m*sizeof({{varname}}(0, 0)));
        }
        outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endfor %}
}

void _dealloc_arrays()
{
	using namespace brian;


	{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
	dev{{varname}}.clear();
	thrust::device_vector<{{c_data_type(var.dtype)}}>().swap(dev{{varname}});
	{% endfor %}

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not var in dynamic_array_specs %}
	if({{varname}}!=0)
	{
		delete [] {{varname}};
		{{varname}} = 0;
	}
	if(dev{{varname}}!=0)
	{
		cudaFree(dev{{varname}});
		dev{{varname}} = 0;
	}
	{% endif %}
	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	if({{name}}!=0)
	{
		delete [] {{name}};
		{{name}} = 0;
	}
	{% endfor %}
}

{% endmacro %}

/////////////////////////////////////////////////////////////////////////////////////////////////////

{% macro h_file() %}

#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include<vector>
#include<stdint.h>
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "network.h"

#include <thrust/device_vector.h>

namespace brian {

//////////////// clocks ///////////////////
{% for clock in clocks %}
extern Clock {{clock.name}};
{% endfor %}

//////////////// networks /////////////////
extern Network magicnetwork;
{% for net in networks %}
extern Network {{net.name}};
{% endfor %}

//////////////// dynamic arrays ///////////
{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
extern std::vector<{{c_data_type(var.dtype)}}> {{varname}};
extern thrust::device_vector<{{c_data_type(var.dtype)}}> dev{{varname}};
{% endfor %}

//////////////// arrays ///////////////////
{% for var, varname in array_specs | dictsort(by='value') %}
{% if not var in dynamic_array_specs %}
extern {{c_data_type(var.dtype)}} * {{varname}};
extern {{c_data_type(var.dtype)}} * dev{{varname}};
extern const int _num_{{varname}};
{% endif %}
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
extern DynamicArray2D<{{c_data_type(var.dtype)}}> {{varname}};
{% endfor %}

/////////////// static arrays /////////////
{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
{# arrays that are initialized from static data are already declared #}
{% if not name in array_specs.values() %}
extern {{dtype_spec}} *{{name}};
extern {{dtype_spec}} *dev{{name}};
extern const int _num_{{name}};
{% endif %}
{% endfor %}

//////////////// synapses /////////////////
{% for S in synapses | sort(attribute='name') %}
// {{S.name}}
extern Synapses<double> {{S.name}};
{% for path in S._pathways | sort(attribute='name') %}
extern SynapticPathway<double> {{path.name}};
{% endfor %}
{% endfor %}

extern unsigned int num_cuda_processors;
extern unsigned int max_threads_per_block;
extern unsigned int max_shared_mem_size;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
