#ifndef _CUDA_VECTOR_H_
#define _CUDA_VECTOR_H_

using namespace std;

/*
 * current memory allocation strategy:
 * only grow larger (by factor 2)
 */
template <class scalar>
class CudaVector
{
private:
		scalar* data;
		int capacity;
		int used;

		__device__ void resize(int new_capacity)
		{
			if(new_capacity > capacity)
			{
				scalar* new_data = (scalar*)malloc(sizeof(scalar) * new_capacity);
				if (new_data)
				{
					memcpy(new_data, data, size());
					free(data);
					data = new_data;
					capacity = new_capacity;
				}
			}
			else if(new_capacity < used)
			{
				capacity = new_capacity;
				used = capacity;
			}
		};

public:
		__device__ CudaVector()
		{
			capacity = 1;
			used = 0;
			data = (scalar*)malloc(sizeof(scalar) * capacity);
		};

		__device__ ~CudaVector()
		{
			free(data);
		};

		__device__ scalar* content()
		{
			return data;
		};

		__device__ scalar get(int index)
		{
			if(index <= used && index >= 0)
			{
				return data[index];
			}
			else
			{
				return 0;
			}
		}

		__device__ void push(scalar elem)
		{
		    if(capacity == used)
		    {
		    	resize(capacity * 2);
		    }
		    data[used++] = elem;
		};

		__device__ void reset()
		{
			used = 0;
			//kleiner reallocen?
		};

		__device__ int size()
		{
			return used;
		};
};

#endif

