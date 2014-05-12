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
	CudaVector<DTYPE_int>** queue;
	scalar dt;
	unsigned int offset;
	unsigned int *delays;
	int max_delay;
	int num_parallel;
	int source_start;
	int source_end;
	thrust::device_vector<int> synapses;
	thrust::device_vector<int> synapses_indices;

	CSpikeQueue(int _source_start, int _source_end)
		: source_start(_source_start), source_end(_source_end)
	{
		offset = 0;
		dt = 0.0;
		delays = NULL;
		max_delay = 0;
		num_parallel = source_end - source_start;
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
		synapses_indices.clear();
		synapses_indices.resize(source_end - source_start + 1);

		//syn connect aufbauen
		for(unsigned int i = 0; i < n_synapses + 1; i++)
		{
			delays[i] =  (int)(real_delays[i] / _dt + 0.5);
			int index = synapses_indices[sources[i] - source_start + 1] - 1;
			synapses.insert(index, &i);
			for(int j = index; j < n_synapses; j++)
			{
				synapses_indices[j]++;
			}
		}

		max_delay = get_maxdelay(delays, n_synapses);
		//Get maximal delay of new and old spikes
		max_delay = (max_delay > newsize) ? max_delay : newsize;

		queue = new CudaVector[max_delay][num_parallel];

		//insert old spikes
		//TODO

	}

	__global__ void push_kernel(int* synapses, int* synapses_indices, CudaVector** q, int num_parallel, int* par_spikes, int nspikes)
	{
		int mpid = threadIdx.x;
		
		for(unsigned int idx_spike = mpid; idx_spike < spikespace_size; idx_spike += num_parallel)
//in zusammenhängende loop umschreiben => MP0 hat 0...5, MP1 hat 6...10
		{
			if(idx < spikespace_size)
			{
				const unsigned int idx_neuron = spikes[idx_spike] - source_start;
				int begin_index = synapses_indices[idx_neuron];
				int end_index = synapses_indices[idx_neuron + 1];
				for(unsigned int idx_indices = begin_index; idx_indices < end_index; idx_indices++)
				{
					const int synaptic_index = synapses[idx_indices];
					const unsigned int delay = delays[synaptic_index];
					// insert the index into the correct queue
					//nicht nur synapses index, sondern auch pre/post synaptische id speichern => mehr queues
					queue[mpid][(offset+delay)%max_delay].push(synaptic_index);
				}
			}
		}
	}

	void push(int *spikes, unsigned int nspikes)
	{
		//TODO synapses{_indices} casten
		push_kernel<<<1, num_parallel>>>(synapses, synapses_indices, queue, num_parallel, source_end - source_start, spikes, nspikes);

	};

	inline DTYPE_int* peek()
	{
		//TODO alles von den queues in einen Array kopieren
		return queue[offset];
	};

	__global__ void advance_kernel(CudaVector** q, int offset)
	{
		int tid = threadIdx.x;
		q[tid][offset].reset();
	}

	void advance()
	{
		//clear last queue entries
		advance_kernel<<<1, num_parallel>>(queue, offset);
		// and advance to the next offset
		offset = (offset+1)%max_delay;
	};
};

