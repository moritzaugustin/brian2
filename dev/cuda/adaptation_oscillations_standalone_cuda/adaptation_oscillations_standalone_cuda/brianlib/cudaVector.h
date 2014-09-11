#ifndef _CUDA_VECTOR_H_
#define _CUDA_VECTOR_H_

#include <cstdio>

/*
 * current memory allocation strategy:
 * only grow larger (by factor 2)
 */

#define INITIAL_SIZE 1

template <class scalar>
class cudaVector
{
private:
	scalar* data;
	int size_allocated;
	int size_used;

	__device__ void resize(int new_capacity)
	{
		if(new_capacity > size_allocated)
		{
			scalar* new_data = (scalar*)malloc(sizeof(scalar) * new_capacity);
			if (new_data)
			{
				memcpy(new_data, data, sizeof(scalar)*size());
				free(data);
				data = new_data;
				size_allocated = new_capacity;
			}
		}
		else if(new_capacity < size_used)
		{
			//kleiner reallocen?
			size_used = new_capacity;
		};
	};

public:
	__device__ cudaVector()
	{
		size_used = 0;
		data = (scalar*)malloc(sizeof(scalar) * INITIAL_SIZE);
		if(data)
		{
			size_allocated = INITIAL_SIZE;
		}
	};

	__device__ ~cudaVector()
	{
		free(data);
	};

	__device__ scalar* getDataPointer()
	{
		return data;
	};

	__device__ scalar getDataByIndex(int index)
	{
		if(index <= size_used && index >= 0)
		{
			return data[index];
		}
		else
		{
			return -1;
		}
	}

	__device__ void push(scalar elem)
	{
		if(size_allocated == size_used)
		{
			resize(size_allocated*2 + 1);
		}
		else{}
		if(size() + 1 <= size_allocated)
		{
			data[size_used] = elem;
			size_used++;
		}
	};

	__device__ void reset()
	{
		size_used = 0;
	};

	__device__ int size()
	{
		return size_used;
	};
};

#endif

