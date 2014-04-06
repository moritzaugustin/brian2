#include<iostream>
#include<vector>
#include<algorithm>
#include<inttypes.h>

#include <thrust/device_vector.h>

using namespace std;

//TODO: The data type for indices is currently fixed (int), all floating point
//      variables (delays, dt) are assumed to use the same data type
typedef int32_t DTYPE_int;

template <class scalar, int bsize>
class CSpikeQueue
{
public:
	scalar* dev_q;
	int current_size;
	scalar dt;
	unsigned int offset;
	unsigned int *delays;
	int source_start;
	int source_end;
	thrust::device_vector<int> synapses_data;
	thrust::device_vector<int> synapses_index;

	CSpikeQueue(int _source_start, int _source_end)
		: source_start(_source_start), source_end(_source_end)
	{
		cudaMalloc((void**)&dev_q, sizeof(scalar)*bsize);
		current_size = 1;
		offset = 0;
		dt = 0.0;
		delays = NULL;
		synapses_data.clear();
		synapses_index.clear();
	};

	void clear_resize(int newsize)
	{
		scalar* new_q;
		cudaMalloc((void**)&new_q, newsize*sizeof(scalar)*bsize);

		cudaFree(q);
		dev_q = new_q;

		current_size = newsize;
		offset = 0;
	}

	void expand(unsigned int newsize)
	{
		const unsigned int n = current_size;
		if (newsize<=n)
		    return;

		// rotate offset back to start (leaves the circular structure unchanged)
		scalar* new_q;
		cudaMalloc((void**)&new_q, newsize*sizeof(scalar)*bsize);
		cudaMemcpy(new_q, dev_q + sizeof(scalar)*bsize*offset, (current_size - offset)*sizeof(scalar)*bsize, cudaMemcpyDeviceToDevice);
		cudaMemcpy(new_q + (current_size - offset)*sizeof(scalar)*bsize, dev_q, sizeof(scalar)*bsize*offset, cudaMemcpyDeviceToDevice);

		cudaFree(q);
		dev_q = new_q;

		current_size = newsize;
		offset = 0;
	};

	inline void ensure_delay(unsigned int delay)
	{
		if(delay>=current_size)
		{
			expand(delay+1);
		}
	};

	inline scalar* peek()
	{
		return (dev_q + offset*sizeof(scalar)*bsize);
	};

	void advance()
	{
		offset = (offset+1)%current_size;
	};

    void prepare(scalar *real_delays, int *sources, unsigned int n_synapses,
                 double _dt)
    {
        if (delays)
            delete [] delays;

        if (dt != 0.0 && dt != _dt)
        {
            // dt changed, we have to get the old spikes out of the queue and
            // reinsert them at the correct positions
            scalar* q_copy;
            cudaMalloc(&q_copy, current_size*sizeof(scalar)*bsize);
            cudaMemcpy(q_copy, dev_q, current_size*sizeof(scalar)*bsize, cudaMemcpyDeviceToDevice);
            const double conversion_factor = dt / _dt;
            const unsigned int oldsize = current_size;
            const unsigned int newsize = (int)(oldsize * conversion_factor) + 1;
            clear_resize(newsize);
            cudaMemset(dev_q, -1, newsize*bsize*sizeof(scalar));
            for (unsigned int i=0; i<oldsize; i++)
            {
            	cudaMemcpy(dev_q + ((int)(i*conversion_factor + 0.5)) * bsize * sizeof(scalar), q_copy + ((i + offset)%oldsize)*sizeof(scalar)*bsize, sizeof(scalar)*bsize, cudaMemcpyDeviceToDevice);
            }
            offset = 0;
        }

        delays = new unsigned int[n_synapses];
        synapses_data.clear();
        synapses_index.clear();
        synapses_index.resize(source_end - source_start + 1);

        for (unsigned int i=0; i<n_synapses; i++)
        {
            delays[i] =  (int)(real_delays[i] / _dt + 0.5); //round to nearest int
            synapses_data.insert(synapses_data.begin() + synapses_index[sources[i] - source_start + 1], i);
            for(int j = sources[i] - source_start + 1; j < synapses_index.size(); j++)
            {
            	synapses_index[j]++;
            }
        }

        dt = _dt;
    }

    __global__ void push_kernel()
    {
    	int tid = threadIdx.x;
    }

	void push(int *spikes, unsigned int nspikes)
	{
		const unsigned int start = source_start;
		const unsigned int stop = source_end;
		//TODO: in kernel
		//push_kernel<<<1, stop - start>>>();

		/*
		for(unsigned int idx_spike=start; idx_spike<stop; idx_spike++)
		{
			const unsigned int idx_neuron = spikes[idx_spike] - source_start;
			int start_index = synapses_index[idx_neuron];
			int stop_index = synapses_index[idx_neuron + 1];
			for(unsigned int idx_indices=start_index; idx_indices<stop_index; idx_indices++)
			{
				const int synaptic_index = synapses_data[idx_indices];
				const unsigned int delay = delays[synaptic_index];
				// make sure there is enough space and resize if not
				ensure_delay(delay);
				// insert the index into the correct queue
				dev_q[((offset + delay)%current_size)*sizeof(scalar)*bsize + synaptic_index] = synaptic_index;
			}
		}
		*/
	};

};
