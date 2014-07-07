#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include "spikequeue.h"

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget;
	scalar dt;
	CSpikeQueue<scalar>* queue;
	
	__device__ void init(int num_mps, int _Nsource, int _Ntarget,
					scalar _dt, int _spikes_start, int _spikes_stop)
	{
		Nsource = _Nsource;
		Ntarget = _Ntarget;
		dt = _dt;
		this->queue = new CSpikeQueue<scalar>;
		queue->init(num_mps, _spikes_start, _spikes_stop);
	};

	__device__ void destroy()
	{
		queue->destroy();
		delete queue;
	};
};

template <class scalar>
class Synapses
{
public:
    int _N_value;
    inline double _N() { return _N_value;};
	int Nsource;
	int Ntarget;
	std::vector< std::vector<int> > _pre_synaptic;
	std::vector< std::vector<int> > _post_synaptic;

	Synapses(int _Nsource, int _Ntarget)
		: Nsource(_Nsource), Ntarget(_Ntarget)
	{
		for(int i=0; i<Nsource; i++)
			_pre_synaptic.push_back(std::vector<int>());
		for(int i=0; i<Ntarget; i++)
			_post_synaptic.push_back(std::vector<int>());
		_N_value = 0;
	};
};

#endif

