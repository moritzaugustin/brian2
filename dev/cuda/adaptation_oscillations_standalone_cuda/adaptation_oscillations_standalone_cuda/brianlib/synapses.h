#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>
#include "spikequeue.h"

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	CSpikeQueue<scalar> *queue;

	__device__ void init()
	{
		queue = new CSpikeQueue<scalar>;
		if(!queue)
		{
			printf("ERROR while allocating device memory with size %ld in synapses.h/init()\n", sizeof(CSpikeQueue<scalar>));
		}
	}
	
	__device__ void destroy()
	{
		if(queue)
		{
			queue->destroy();
			delete queue;
		}
	}
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

