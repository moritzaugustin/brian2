using namespace std;

/*
 * current memory allocation strategy:
 * only grow larger (by factor 2)
 */
template <class scalar>
class CudaVector
{
private:
		int capacity;
		int used;
		scalar* data;
		__device__ void resize(int new_capacity);
public:
		__device__ CudaVector();
		__device__ ~CudaVector();
		__device__ scalar* content();
		__device__ void push(scalar elem);
		__device__ void reset();
		__device__ int size();
};
