#include "CudaVector.h"

template<typename scalar>
__device__ void CudaVector<scalar>::resize(int new_capacity)
{
	if(new_capacity > capacity)
	{
		scalar* new_data = (scalar*)malloc(sizeof(scalar) * new_capacity);
		if (new_data)
		{
			for(int i = 0; i < size(); i++)
			{
				new_data[i] = data[i];
			}
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

template<typename scalar>
__device__ CudaVector<scalar>::CudaVector()
{
	capacity = 1;
	used = 0;
	data = (scalar*)malloc(sizeof(scalar) * capacity);
};

template<typename scalar>
__device__ CudaVector<scalar>::~CudaVector()
{
	free(data);
};

template<typename scalar>
__device__ scalar* CudaVector<scalar>::content()
{
	return data;
};

template<typename scalar>
__device__ void CudaVector<scalar>::push(scalar elem)
{
    if(capacity == used)
    {
    	resize(capacity * 2);
    }
    data[used++] = elem;
};

template<typename scalar>
__device__ void CudaVector<scalar>::reset()
{
	used = 0;
	//kleiner reallocen?
};

template<typename scalar>
__device__ int CudaVector<scalar>::size()
{
	return used;
};
