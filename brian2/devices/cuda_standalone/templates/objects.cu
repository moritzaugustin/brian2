{% macro cu_file() %}

#include<stdint.h>
#include<vector>
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "network.h"
#include<iostream>
#include<fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>

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
thrust::device_vector<{{c_data_type(var.dtype)}}> brian::dev{{varname}};
{% endfor %}

//////////////// dynamic arrays 2d /////////
{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
thrust::device_vector<{{c_data_type(var.dtype)}}*> brian::addresses_monitor_{{varname}};
thrust::device_vector<{{c_data_type(var.dtype)}}>* brian::{{varname}};
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
// {{path.name}}
__device__ unsigned int* brian::{{path.name}}_size_by_pre;
__device__ int32_t** brian::{{path.name}}_synapses_id_by_pre;
__device__ int32_t** brian::{{path.name}}_post_neuron_by_pre;
__device__ unsigned int** brian::{{path.name}}_delay_by_pre;
__device__ SynapticPathway<double> brian::{{path.name}};
{% endfor %}
{% endfor %}

unsigned int brian::num_parallel_blocks;
unsigned int brian::max_threads_per_block;
unsigned int brian::max_shared_mem_size;

{% for S in synapses | sort(attribute='name') %}
{% for path in S._pathways | sort(attribute='name') %}
__global__ void {{path.name}}_init(
				unsigned int Nsource,
				unsigned int Ntarget,
				double* delays,
				int32_t* sources,
				int32_t* targets,
				double dt,
				int32_t start,
				int32_t stop
		)
{
	using namespace brian;

	{{path.name}}.init(Nsource, Ntarget, delays, sources, targets, dt, start, stop);
}
{% endfor %}
{% endfor %}

//////////////random numbers//////////////////
curandGenerator_t brian::random_float_generator;
{% for co in codeobj_with_rand | sort(attribute='name') %}
float* brian::dev_{{co.name}}_random_uniform_floats;
__device__ float* brian::_array_{{co.name}}_rand;
{% endfor %}
{% for co in codeobj_with_randn | sort(attribute='name') %}
float* brian::dev_{{co.name}}_random_normal_floats;
__device__ float* brian::_array_{{co.name}}_randn;
{% endfor %}

void _init_arrays()
{
	using namespace brian;

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	num_parallel_blocks = props.multiProcessorCount;
	max_threads_per_block = props.maxThreadsPerBlock;
	max_shared_mem_size = props.sharedMemPerBlock;
	
	curandCreateGenerator(&random_float_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(random_float_generator, time(0));

	//since random number generation is at the end of each clock_tick, also generate numbers for t = 0
	{% for co in codeobj_with_rand | sort(attribute='name') %}
	cudaMalloc((void**)&dev_{{co.name}}_random_uniform_floats, sizeof(float)*{{co.owner._N}} * {{co.rand_calls}});
	cudaMemcpyToSymbol(_array_{{co.name}}_rand, &dev_{{co.name}}_random_uniform_floats, sizeof(float*));
	curandGenerateUniform(random_float_generator, dev_{{co.name}}_random_uniform_floats, {{co.owner._N}} * {{co.rand_calls}});
	{% endfor %}
	{% for co in codeobj_with_randn | sort(attribute='name') %}
	cudaMalloc((void**)&dev_{{co.name}}_random_normal_floats, sizeof(float)*{{co.owner._N}} * {{co.randn_calls}});
	cudaMemcpyToSymbol(_array_{{co.name}}_randn, &dev_{{co.name}}_random_normal_floats, sizeof(float*));
	curandGenerateNormal(random_float_generator, dev_{{co.name}}_random_normal_floats, {{co.owner._N}} * {{co.randn_calls}}, 0.0, 1.0);
	{% endfor %}

	{% for S in synapses | sort(attribute='name') %}
	{% for path in S._pathways | sort(attribute='name') %}
	{{path.name}}_init<<<1,1>>>(
			{{path.source|length}},
			{{path.target|length}},
			thrust::raw_pointer_cast(&dev{{dynamic_array_specs[path.variables['delay']]}}[0]),
			thrust::raw_pointer_cast(&dev{{dynamic_array_specs[path.synapse_sources]}}[0]),
			thrust::raw_pointer_cast(&dev{{dynamic_array_specs[path.synapse_targets]}}[0]),
			{{path.source.dt_}},
			{{path.source.start}},
			{{path.source.stop}}
			);
	{% endfor %}
	{% endfor %}

    // Arrays initialized to 0
	{% for var in zero_arrays | sort(attribute='name') %}
	{% set varname = array_specs[var] %}
	{{varname}} = new {{c_data_type(var.dtype)}}[{{var.size}}];
	for(int i=0; i<{{var.size}}; i++) {{varname}}[i] = 0;
	cudaMalloc((void**)&dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	if(!dev{{varname}})
	{
		printf("ERROR while allocating {{varname}} on device with size %ld\n", sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
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
		printf("ERROR while allocating {{varname}} on device with size %ld\n", sizeof({{c_data_type(var.dtype)}})*_num_{{varname}});
	}
	cudaMemcpy(dev{{varname}}, {{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyHostToDevice);

	{% endfor %}

	// static arrays
	{% for (name, dtype_spec, N, filename) in static_array_specs | sort %}
	{{name}} = new {{dtype_spec}}[{{N}}];
	cudaMalloc((void**)&dev{{name}}, sizeof({{dtype_spec}})*{{N}});
	if(!dev{{name}})
	{
		printf("ERROR while allocating {{varname}} on device with size %ld\n", sizeof({{dtype_spec}})*{{N}});
	}
	{% endfor %}

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
	{{varname}} = new thrust::device_vector<{{c_data_type(var.dtype)}}>[_num__array_{{var.owner.name}}__indices];
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
	cudaMemcpy(dev{{name}}, {{name}}, sizeof({{dtype_spec}})*{{N}}, cudaMemcpyHostToDevice);
	{% endfor %}
}	

void _write_arrays()
{
	using namespace brian;

	{% for var, varname in array_specs | dictsort(by='value') %}
	{% if not (var in dynamic_array_specs or var in dynamic_array_2d_specs) %}
	cudaMemcpy({{varname}}, dev{{varname}}, sizeof({{c_data_type(var.dtype)}})*_num_{{varname}}, cudaMemcpyDeviceToHost);
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>({{varname}}), {{var.size}}*sizeof({{c_data_type(var.dtype)}}));
		outfile_{{varname}}.close();
	} else
	{
		std::cout << "Error writing output file for {{varname}}." << endl;
	}
	{% endif %}
	{% endfor %}

	{% for var, varname in dynamic_array_specs | dictsort(by='value') %}
	thrust::host_vector<{{c_data_type(var.dtype)}}> temp{{varname}} = dev{{varname}};
	ofstream outfile_{{varname}};
	outfile_{{varname}}.open("results/{{varname}}", ios::binary | ios::out);
	if(outfile_{{varname}}.is_open())
	{
		outfile_{{varname}}.write(reinterpret_cast<char*>(thrust::raw_pointer_cast(&temp{{varname}}[0])), dev{{varname}}.size()*sizeof({{c_data_type(var.dtype)}}));
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
			thrust::host_vector<{{c_data_type(var.dtype)}}>* temp_array{{varname}} = new thrust::host_vector<{{c_data_type(var.dtype)}}>[_num__array_{{var.owner.name}}__indices];
	        for (int n=0; n<_num__array_{{var.owner.name}}__indices; n++)
	        {
	        	temp_array{{varname}}[n] = {{varname}}[n];
	        }
	        for(int j = 0; j < temp_array{{varname}}[0].size(); j++)
	        {
	        	for(int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
	        	{
		        	outfile_{{varname}}.write(reinterpret_cast<char*>(&temp_array{{varname}}[i][j]), sizeof({{c_data_type(var.dtype)}}));
	        	}
	        }
	        outfile_{{varname}}.close();
		} else
		{
			std::cout << "Error writing output file for {{varname}}." << endl;
		}
		{% endfor %}
}

{% for S in synapses | sort(attribute='name') %}
{% for path in S._pathways | sort(attribute='name') %}
__global__ void {{path.name}}_destroy()
{
	using namespace brian;

	{{path.name}}.destroy();
}
{% endfor %}
{% endfor %}

void _dealloc_arrays()
{
	using namespace brian;

	{% for co in codeobj_with_rand | sort(attribute='name') %}
	cudaFree(dev_{{co.name}}_random_uniform_floats);
	{% endfor %}
	{% for co in codeobj_with_randn | sort(attribute='name') %}
	cudaFree(dev_{{co.name}}_random_normal_floats);
	{% endfor %}

	{% for S in synapses | sort(attribute='name') %}
	{% for path in S._pathways | sort(attribute='name') %}
	{{path.name}}_destroy<<<1,1>>>();
	{% endfor %}
	{% endfor %}

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

	{% for var, varname in dynamic_array_2d_specs | dictsort(by='value') %}
	for(int i = 0; i < _num__array_{{var.owner.name}}__indices; i++)
	{
		{{varname}}[i].clear();
		thrust::device_vector<{{c_data_type(var.dtype)}}>().swap({{varname}}[i]);
	}
	addresses_monitor_{{varname}}.clear();
	thrust::device_vector<{{c_data_type(var.dtype)}}*>().swap(addresses_monitor_{{varname}});
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
#include "network.h"

#include <thrust/device_vector.h>
#include <curand.h>

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
extern thrust::device_vector<{{c_data_type(var.dtype)}}*> addresses_monitor_{{varname}};
extern thrust::device_vector<{{c_data_type(var.dtype)}}>* {{varname}};
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
extern __device__ unsigned* {{path.name}}_size_by_pre;
extern __device__ int32_t** {{path.name}}_synapses_id_by_pre;
extern __device__ int32_t** {{path.name}}_post_neuron_by_pre;
extern __device__ unsigned int** {{path.name}}_delay_by_pre;
extern __device__ SynapticPathway<double> {{path.name}};
{% endfor %}
{% endfor %}

//////////////// random numbers /////////////////
extern curandGenerator_t random_float_generator;

{% for co in codeobj_with_rand | sort(attribute='name') %}
extern float* dev_{{co.name}}_random_uniform_floats;
extern __device__ float* _array_{{co.name}}_rand;
{% endfor %}
{% for co in codeobj_with_randn | sort(attribute='name') %}
extern float* dev_{{co.name}}_random_normal_floats;
extern __device__ float* _array_{{co.name}}_randn;
{% endfor %}

//CUDA
extern unsigned int num_parallel_blocks;
extern unsigned int max_threads_per_block;
extern unsigned int max_shared_mem_size;

}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


{% endmacro %}
