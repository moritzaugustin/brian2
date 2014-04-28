#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include <stdio.h>

#include <thrust/device_vector.h>

using namespace std;

//TODO: The data type for indices is currently fixed (int), all floating point
//      variables (delays, dt) are assumed to use the same data type
typedef int32_t DTYPE_int;

template <class scalar>
class CSpikeQueue
{
public:
	thrust::device_vector< DTYPE_int >* queue; // queue[(offset+i)%queue.size()] is delay i relative to current time
	scalar dt;
	unsigned int offset;
	unsigned int *delays;
	int max_delay;
	int source_start;
	int source_end;
	vector< vector<int> > synapses;

	CSpikeQueue(int _source_start, int _source_end)
		: source_start(_source_start), source_end(_source_end)
	{
		offset = 0;
		dt = 0.0;
		delays = NULL;
		max_delay = 0;
	};

	int get_maxdelay(unsigned int* delays, int n_delays)
	{
		if(n_delays == 0)
		{
			return 0;
		}
		int max = delays[0];
		for(int i = 1; i < n_delays; i++)
		{
			if(delays[i] > max)
			{
				max = delays[i];
			}
		}
		return max + 1;
	}

    void prepare(scalar *real_delays, int *sources, unsigned int n_synapses,
                 double _dt)
    {
        unsigned int newsize = 0;
        unsigned int oldsize = max_delay;
        thrust::device_vector< DTYPE_int >* queue_copy = queue; // does a real copy

        if (delays)
            delete [] delays;

        if (dt != 0.0 && dt != _dt)
        {
		// dt changed, we have to get the old spikes out of the queue and
		// reinsert them at the correct positions, after creating the new queue
		const double conversion_factor = dt / _dt;

		//get old max_delay
		newsize = (int)(oldsize * conversion_factor) + 1;
	}

        delays = new unsigned int[n_synapses];
        synapses.clear();
        synapses.resize(source_end - source_start);

        for (unsigned int i=0; i<n_synapses; i++)
        {
            delays[i] =  (int)(real_delays[i] / _dt + 0.5); //round to nearest int
            synapses[sources[i] - source_start].push_back(i);
        }

        max_delay = get_maxdelay(delays, n_synapses);
        //Get maximal delay of new and old spikes
        max_delay = (max_delay > newsize) ? max_delay : newsize;

        queue = new thrust::device_vector<DTYPE_int>[max_delay];
        for(int i = 0; i < max_delay; i++)
        {
        	queue[i] = thrust::device_vector<DTYPE_int>(0);
        }

        // if there are old spikes
	if (dt != 0.0 && dt != _dt)
	{
		const double conversion_factor = dt / _dt;
		// insert old spikes
		for (unsigned int i=0; i<oldsize; i++)
		{
			thrust::device_vector<DTYPE_int> spikes = queue_copy[(i + offset) % oldsize];
			queue[(int)(i * conversion_factor + 0.5)] = spikes;
		}
		offset = 0;
		free(queue_copy);
	}

        dt = _dt;
    }

	void push(int *spikes, unsigned int nspikes)
	{
		//TODO: copy_if ?
		for(int i = source_start, j = 0; i < source_end && j < nspikes; i++)
		{
			if(spikes[i] != -1)
			{
				j++;
				const unsigned int idx_neuron = spikes[i] - source_start;
				vector<int> &cur_indices = synapses[idx_neuron];
				for(unsigned int idx_indices=0; idx_indices<cur_indices.size(); idx_indices++)
				{
					const int synaptic_index = cur_indices[idx_indices];
					const unsigned int delay = delays[synaptic_index];
					// insert the index into the correct queue
					queue[(offset+delay)%max_delay].push_back(synaptic_index);
				}
			}
		}
	};

	inline thrust::device_vector<DTYPE_int> peek()
	{
		return queue[offset];
	};

	void advance()
	{
		// empty the current queue, note that for most compilers this shouldn't deallocate the memory,
		// although VC<=7.1 will, so it will be less efficient with that compiler
		queue[offset].clear();
		// and advance to the next offset
		offset = (offset+1)%max_delay;
	};
};

