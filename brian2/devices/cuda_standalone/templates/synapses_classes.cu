{% macro cu_file() %}
{% endmacro %}

{% macro h_file() %}

#ifndef _BRIAN_SYNAPSES_H
#define _BRIAN_SYNAPSES_H

#include<vector>
#include<algorithm>

#include "brianlib/spikequeue.h"

template<class scalar> class Synapses;
template<class scalar> class SynapticPathway;

template <class scalar>
class SynapticPathway
{
public:
	int Nsource, Ntarget;
	std::vector<scalar> &delay;
	std::vector<int> &sources;
	std::vector<int> all_peek;
	scalar dt;
	CSpikeQueue<scalar>* queue;
	SynapticPathway(int _Nsource, int _Ntarget, std::vector<scalar>& _delay, std::vector<int> &_sources,
					scalar _dt, int _spikes_start, int _spikes_stop)
		: Nsource(_Nsource), Ntarget(_Ntarget), delay(_delay), sources(_sources), dt(_dt)
	{
		queue = new CSpikeQueue<scalar>(_spikes_start, _spikes_stop);
    };

	~SynapticPathway()
	{
		delete queue;
	}

	void push(int *spikes, unsigned int nspikes)
    {
    	queue->push(spikes, nspikes);
    }

	void advance()
    {
    	queue->advance();
    }

	DTYPE_int* peek()
    {
		if (_thread == 0)
				all_peek.clear();
		all_peek.insert(all_peek.end(), queue->peek()->begin(), queue->peek()->end());
    	return &all_peek;
    }

    void prepare(scalar *real_delays, unsigned int n_delays,
                 int *sources, unsigned int n_synapses, double _dt)
    {
    	unsigned int length;
    	length = n_synapses;
    	unsigned int padding  = 0;

		if (n_delays > 1)
			queue->prepare(&real_delays[padding], length, &sources[padding], length, _dt);
		else
			queue->prepare(&real_delays[0], 1, &sources[padding], length, _dt);
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

{% endmacro %}
