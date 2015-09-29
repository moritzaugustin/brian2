////////////////////////////////////////////////////////////////////////////
//// MAIN CODE /////////////////////////////////////////////////////////////

{# USES_VARIABLES { Cm, dt, v, N,
                  ab_star0, ab_star1, ab_star2, b_plus,
                  ab_plus0, ab_plus1, ab_plus2, b_minus,
                  ab_minus0, ab_minus1, ab_minus2, v_star, u_plus, u_minus,
                  gtot_all, I0_all,
                  c1, c2, c3,
                  _P, _P_diag, _P_parent, _P_children,
                  _B, _morph_i, _morph_parent_i, _starts, _ends,
                  _morph_children, _morph_children_num, _morph_idxchild,
                  _invr0, _invrn} #}
{% extends 'common_group.cu' %}


{% block kernel %}
__global__ void kernel_{{codeobj_name}}_integration(
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%


	if (tid == 0)
		printf("(integration kernel) bid=%d\n", bid);

	{% block num_thread_check %}
	if(_idx >= N)
	{
		return;
	}
	{% endblock %}

	//// MAIN CODE ////////////
	{{scalar_code|autoindent}}
	{{vector_code|autoindent}}
	{{gtot_all}}[_idx] = _gtot;
	{{I0_all}}[_idx] = _I0;
}



__global__ void kernel_{{codeobj_name}}_system1(
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid; // bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%


	if (tid == 0)
		printf("(system1 kernel) bid=%d\n", bid);

	double ai,bi,_m;

	// system 2a: solve for v_star
	const int _i = _idx;
	// first and last index of the i-th branch
	const int i_start = {{_starts}}[_i];
	const int i_end = {{_ends}}[_i];

	// upper triangularization of tridiagonal system for v_star
	for(int i=i_start;i<i_end+1;i++)
	{
		{{v_star}}[i]=-({{Cm}}[i]/{{dt}}*{{v}}[i])-{{I0_all}}[i]; // RHS -> v_star (solution)
		bi={{ab_star1}}[i]-{{gtot_all}}[i]; // main diagonal
		if (i<N-1)
			{{c1}}[i]={{ab_star0}}[i+1]; // superdiagonal
		if (i>0)
		{
			ai={{ab_star2}}[i-1]; // subdiagonal
			_m=1.0/(bi-ai*{{c1}}[i-1]);
			{{c1}}[i]={{c1}}[i]*_m;
			{{v_star}}[i]=({{v_star}}[i] - ai*{{v_star}}[i-1])*_m;
		} else
		{
			{{c1}}[0]={{c1}}[0]/bi;
			{{v_star}}[0]={{v_star}}[0]/bi;
		}
	}
	// backwards substituation of the upper triangularized system for v_star
	for(int i=i_end-1;i>=i_start;i--)
		{{v_star}}[i]={{v_star}}[i] - {{c1}}[i]*{{v_star}}[i+1];
}


__global__ void kernel_{{codeobj_name}}_system2(
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid; //bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%


	if (tid == 0)
		printf("(system2 kernel) bid=%d\n", bid);

	double ai,bi,_m;

	// system 2b: solve for u_plus
	const int _i = _idx;
	// first and last index of the i-th branch
	const int i_start = {{_starts}}[_i];
	const int i_end = {{_ends}}[_i];

	// upper triangularization of tridiagonal system for u_plus
	for(int i=i_start;i<i_end+1;i++)
	{
		{{u_plus}}[i]={{b_plus}}[i]; // RHS -> u_plus (solution)
		bi={{ab_plus1}}[i]-{{gtot_all}}[i]; // main diagonal
		if (i<N-1)
			{{c2}}[i]={{ab_plus0}}[i+1]; // superdiagonal
		if (i>0)
		{
			ai={{ab_plus2}}[i-1]; // subdiagonal
			_m=1.0/(bi-ai*{{c2}}[i-1]);
			{{c2}}[i]={{c2}}[i]*_m;
			{{u_plus}}[i]=({{u_plus}}[i] - ai*{{u_plus}}[i-1])*_m;
		} else
		{
			{{c2}}[0]={{c2}}[0]/bi;
			{{u_plus}}[0]={{u_plus}}[0]/bi;
		}
	}
	// backwards substituation of the upper triangularized system for u_plus
	for(int i=i_end-1;i>=i_start;i--)
		{{u_plus}}[i]={{u_plus}}[i] - {{c2}}[i]*{{u_plus}}[i+1];
}


__global__ void kernel_{{codeobj_name}}_system3(
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid; //bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%

	if (tid == 0)
			printf("(system3 kernel) bid=%d\n", bid);

	double ai,bi,_m;

	// system 2c: solve for u_minus
	const int _i = _idx;
	// first and last index of the i-th branch
	const int i_start = {{_starts}}[_i];
	const int i_end = {{_ends}}[_i];

	// upper triangularization of tridiagonal system for u_minus
	for(int i=i_start;i<i_end+1;i++)
	{
		{{u_minus}}[i]={{b_minus}}[i]; // RHS -> u_minus (solution)
		bi={{ab_minus1}}[i]-{{gtot_all}}[i]; // main diagonal
		if (i<N-1)
			{{c3}}[i]={{ab_minus0}}[i+1]; // superdiagonal
		if (i>0)
		{
			ai={{ab_minus2}}[i-1]; // subdiagonal
			_m=1.0/(bi-ai*{{c3}}[i-1]);
			{{c3}}[i]={{c3}}[i]*_m;
			{{u_minus}}[i]=({{u_minus}}[i] - ai*{{u_minus}}[i-1])*_m;
		} else
		{
			{{c3}}[0]={{c3}}[0]/bi;
			{{u_minus}}[0]={{u_minus}}[0]/bi;
		}
	}
	// backwards substituation of the upper triangularized system for u_minus
	for(int i=i_end-1;i>=i_start;i--)
		{{u_minus}}[i]={{u_minus}}[i] - {{c3}}[i]*{{u_minus}}[i+1];
}


__global__ void kernel_{{codeobj_name}}_coupling(
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%

	bool verbose_output = false; // print information for debugging purposes (later to be removed)
	// before the efficient sparse solver is final we provide three kinds of solver
	enum linearsolver { EFFICIENT_SPARSE, EFFICIENT_DENSE, INEFFICIENT_DENSE};
	linearsolver linsolv = EFFICIENT_SPARSE;
//	linearsolver linsolv = EFFICIENT_DENSE;
//	linearsolver linsolv = INEFFICIENT_DENSE; // old version from Marcel and Romain


	if (tid == 0)
		printf("(coupling kernel) bid=%d\n", bid);

	// integration step 3: solve the coupling system (no parallelism)

	// indexing for _P_children, the elements above the diagonal of the sparse version coupling matrix _P
	const int children_rowlength = _num_morph_children/_num_morph_children_num;
	#define IDX_C(idx_row,idx_col) children_rowlength * idx_row + idx_col

//	if (verbose_output) {
//		cout << "=== construct matrix P (dense and sparse version)";
//		cout << "children_rowlength=" << children_rowlength << endl;
//	}


	// indexing for the coupling matrix _P
	#define IDX_B(idx_row,idx_col) _num_B * idx_row + idx_col

	// step 3a: construct the coupling matrix _P
	for (int _j=0; _j<_num_B - 1; _j++)
	{
		const int _i = {{_morph_i}}[_j];
		const int _i_parent = {{_morph_parent_i}}[_j];
		const int _i_childind = {{_morph_idxchild}}[_j];
		const int _first = {{_starts}}[_j];
		const int _last = {{_ends}}[_j];
		const double _invr0 = {{_invr0}}[_j];
		const double _invrn = {{_invrn}}[_j];

//		if (verbose_output)
//			cout << "elments(s) for branch _i=" << _i << ", _i_parent=" << _i_parent << ", _i_childind=" << _i_childind << ", _first=" << _first << ", _last=" << _last << endl;

		// Towards parent
		if (_i == 1) // first branch, sealed end
		{
			// dense matrix version
			{{_P}}[IDX_B(0,0)] = {{u_minus}}[_first] - 1;
			{{_P}}[IDX_B(0,1)] = {{u_plus}}[_first];

			// sparse matrix version:
			{{_P_diag}}[0] = {{u_minus}}[_first] - 1;
			{{_P_children}}[IDX_C(0,0)] = {{u_plus}}[_first];

			// RHS
			{{_B}}[0] = -{{v_star}}[_first];
		}
		else
		{
			// dense matrix version
			{{_P}}[IDX_B(_i_parent,_i_parent)] += (1 - {{u_minus}}[_first]) * _invr0;
			{{_P}}[IDX_B(_i_parent,_i)] = -{{u_plus}}[_first] * _invr0;

			// sparse matrix version:
			{{_P_diag}}[_i_parent] += (1 - {{u_minus}}[_first]) * _invr0;
			{{_P_children}}[IDX_C(_i_parent, _i_childind)] = -{{u_plus}}[_first] * _invr0;

			// RHS
			{{_B}}[_i_parent] += {{v_star}}[_first] * _invr0;
		}

		// Towards children

		// dense matrix version
		{{_P}}[IDX_B(_i,_i)] = (1 - {{u_plus}}[_last]) * _invrn;
		{{_P}}[IDX_B(_i,_i_parent)] = -{{u_minus}}[_last] * _invrn;

		// sparse matrix version
		{{_P_diag}}[_i] = (1 - {{u_plus}}[_last]) * _invrn;
		{{_P_parent}}[_i-1] = -{{u_minus}}[_last] * _invrn;

		// RHS
		{{_B}}[_i] = {{v_star}}[_last] * _invrn;
	}


	// compare the dense and sparse version of matrix P
//	if (verbose_output) {
//		cout << "=== matrix comparison dense vs. sparse " << endl;
//		for (int _j=0; _j<_num_B-1; _j++) // iterate over all branches
//		{
//			const int _i = {{_morph_i}}[_j];
//			const int _i_parent = {{_morph_parent_i}}[_j];
//			const int _i_childind = {{_morph_idxchild}}[_j];
//
//			if (_j==0) {
//				cout << "comp zero line:_P[0,0]=" << {{_P}}[IDX_B(0,0)] << ", _P_sparse[0,0] = _P_diag[0] = " << {{_P_diag}}[0] << endl;
//				cout << "comp zero line:_P[0,1]=" << {{_P}}[IDX_B(0,1)] << ", _P_sparse[0,1] = _P_children[IDX_C(0,0)] = " << {{_P_children}}[IDX_C(0,0)] << endl;
//			}
//
//			cout << "comp (_j=" << _j << ", _i=" << _i << "):_P[_i,_i]=" << {{_P}}[IDX_B(_i,_i)] << ", _P_sparse[_i,_i] = _P_diag[_i] = " << {{_P_diag}}[_i] << endl;
//			cout << "comp (_i_parent=" << _i_parent << "):_P[_i,_i_parent]=" << {{_P}}[IDX_B(_i,_i_parent)] << ", _P_sparse[_i,_i_parent] = _P_parent[_i-1] = " << {{_P_parent}}[_i-1] << endl;
//			for (int k=0; k<{{_morph_children_num}}[_i]; k++) {
//				int _i_child_k = {{_morph_children}}[IDX_C(_i,k)];
//				int _i_childind_k = {{_morph_idxchild}}[_i_child_k - 1];
//				cout << "comp (k=" << k << ", _i_child_k=" << _i_child_k << ", _i_childind_k=" << _i_childind_k << "): _P[_i,_i_child_k]=" << {{_P}}[IDX_B(_i,_i_child_k)]
//					 << ", _P_sparse[_i,_i_child_k] = _P_children[IDX_C(_i,k)] = " << {{_P_children}}[IDX_C(_i,k)]
//					 << " OR equivalently: _P_sparse[_i,_i_child_k] = _P_children[IDX_C(_i,_i_childind_k)] = " << {{_P_children}}[IDX_C(_i,_i_childind_k)] << endl;
//			}
//		}
//		cout << "print P:" << endl;
//		for (int i=0; i<_num_B; i++) {
//			for (int j=0; j<_num_B; j++)
//				cout << {{_P}}[i*_num_B+j] << " ";
//			cout << endl;
//		}
//	}


	// step 3b: solve the linear system (the result will be in _B in the end)

	if (linsolv == EFFICIENT_SPARSE) {
		// use efficient O(n) solution of the linear system (structure-specific Gaussian elemination)
		// exploiting the sparse structural symmetric matrix from branch tree  but still in dense format

//		if (verbose_output)
//			cout << "=== solving the linear system: efficient and sparse version..." << endl;

		// part 1: lower triangularization
		for (int i=_num_B-1; i>=0; i--) {
			const int num_children = {{_morph_children_num}}[i];
//			if (verbose_output)
//				cout << "eliminate above the diagonal in row i=" << i << ", num_children=" << num_children << endl;

			// for every child eliminate the corresponding matrix element of row i
			for (int k=0; k<num_children; k++) {
				int j = {{_morph_children}}[i*children_rowlength+k]; // child index
				// subtracting subfac times the j-th from the i-th row
				//double subfac = {{_P}}[IDX_B(i,j)] / {{_P}}[IDX_B(j,j)]; // P[i,j] appears only here
				double subfac = {{_P_children}}[IDX_C(i,k)] / {{_P_diag}}[j]; // P[i,j] appears only here

				//{{_P}}[IDX_B(i,j)] = 0; //{{_P}}[IDX_B(i,j)]  - subfac * {{_P}}[IDX_B(j,j)]; // element is not used in the following anymore, just for clarity
				{{_P_children}}[IDX_C(i,k)] = {{_P_children}}[IDX_C(i,k)]  - subfac * {{_P_diag}}[j]; // = 0; //{{_P}}[IDX_B(i,j)]  - subfac * {{_P}}[IDX_B(j,j)]; // element is not used in the following anymore, just for clarity

				//{{_P}}[IDX_B(i,i)] = {{_P}}[IDX_B(i,i)]  - subfac * {{_P}}[IDX_B(j,i)]; // note: element j,i is ONLY used here; maybe omit it if a sparse matrix format is introduced
				{{_P_diag}}[i] = {{_P_diag}}[i]  - subfac * {{_P_parent}}[j-1]; // note: element j,i is ONLY used here; maybe omit it if a sparse matrix format is introduced
				{{_B}}[i] = {{_B}}[i] - subfac * {{_B}}[j];
//				if (verbose_output)
//					cout << "eliminated children entry, i=" << i << ", k=" << k << ", j=" << j << ", _morph_idxchild[j-1]=" << {{_morph_idxchild}}[j-1] << endl;

			}
		}

		// part 2: forwards substitution
		//{{_B}}[0] = {{_B}}[0] / {{_P}}[IDX_B(0,0)]; // the first branch does not have a parent
		{{_B}}[0] = {{_B}}[0] / {{_P_diag}}[0]; // the first branch does not have a parent
		for (int i=1; i<_num_B; i++) {
			const int j = {{_morph_parent_i}}[i-1]; // parent index
			//{{_B}}[i] = {{_B}}[i] - {{_P}}[IDX_B(i,j)] * {{_B}}[j];
			{{_B}}[i] = {{_B}}[i] - {{_P_parent}}[i-1] * {{_B}}[j];
			{{_B}}[i] = {{_B}}[i] / {{_P_diag}}[i];
//			if (verbose_output)
//				cout << "solve by substitution in row i=" << i << ", j=" << j << endl;
		}
	}
	else if (linsolv == EFFICIENT_DENSE) {
		// use efficient O(n) solution of the linear system (structure-specific Gaussian elemination)
		// exploiting the sparse structural symmetric matrix from branch tree  but still in dense format
//		if (verbose_output)
//			cout << "=== solving the linear system: efficient but dense version..." << endl;


		// part 1: lower triangularization
		for (int i=_num_B-1; i>=0; i--) {
			const int num_children = {{_morph_children_num}}[i];
//			if (verbose_output)
//				cout << "eliminate above the diagonal in row i=" << i << ", num_children=" << num_children << endl;

			// for every child eliminate the corresponding matrix element of row i
			for (int k=0; k<num_children; k++) {
				int j = {{_morph_children}}[i*children_rowlength+k]; // child index
				// subtracting subfac times the j-th from the i-th row
				double subfac = {{_P}}[IDX_B(i,j)] / {{_P}}[IDX_B(j,j)]; // P[i,j] appears only here
				{{_P}}[IDX_B(i,j)] = 0; // = {{_P}}[IDX_B(i,j)]  - subfac * {{_P}}[IDX_B(j,j)]; // element is not used in the following anymore, zeroed (by definition) just for clarity
				{{_P}}[IDX_B(i,i)] = {{_P}}[IDX_B(i,i)]  - subfac * {{_P}}[IDX_B(j,i)]; // note: element j,i is ONLY used here; maybe omit it if a sparse matrix format is introduced
				{{_B}}[i] = {{_B}}[i] - subfac * {{_B}}[j];
//				if (verbose_output)
//					cout << "eliminated children entry, i=" << i << ", k=" << k << ", j=" << j << ", _morph_idxchild[j-1]=" << {{_morph_idxchild}}[j-1] << endl;
			}
		}

//		if (verbose_output) {
//			cout << "print P after (lower) triang:" << endl;
//				for (int i=0; i<_num_B; i++) {
//					for (int j=0; j<_num_B; j++)
//						cout << {{_P}}[i*_num_B+j] << " ";
//					cout << endl;
//				}
//		}

		// part 2: forwards substitution
		{{_B}}[0] = {{_B}}[0] / {{_P}}[IDX_B(0,0)]; // the first branch does not have a parent
		for (int i=1; i<_num_B; i++) {
			const int j = {{_morph_parent_i}}[i-1]; // parent index
			{{_B}}[i] = {{_B}}[i] - {{_P}}[IDX_B(i,j)] * {{_B}}[j];
			{{_B}}[i] = {{_B}}[i] / {{_P}}[IDX_B(i,i)];
//			if (verbose_output)
//				cout << "solve by substitution in row i=" << i << ", j=" << j << endl;
		}
	}
	else if (linsolv == INEFFICIENT_DENSE) {
		// use inefficient O(n^3) solution (dense Gaussian elemination)

//		if (verbose_output)
//			cout << "=== solving the linear system: old, inefficient version..." << endl;
		for (int i=0; i<_num_B; i++)
		{
			// find pivot element
			int i_pivot = i;
			double pivot_magnitude = fabs({{_P}}[i*_num_B + i]);
			for (int j=i+1; j<_num_B; j++)
			   if (fabs({{_P}}[j*_num_B + i]) > pivot_magnitude)
			   {
				   i_pivot = j;
				   pivot_magnitude = fabs({{_P}}[j*_num_B + i]);
			   }

//			if (pivot_magnitude == 0)
//			{
//				std::cerr << "Singular!" << std::endl;
//			}

			// swap rows
			if (i != i_pivot)
			{
				for (int col=i; col<_num_B; col++)
				{
					const double tmp = {{_P}}[i*_num_B + col];
					{{_P}}[i*_num_B + col] = {{_P}}[i_pivot*_num_B + col];
					{{_P}}[i_pivot*_num_B + col] = tmp;
				}
			}

			// Deal with rows below
			for (int j=i+1; j<_num_B; j++)
			{
				const double pivot_element = {{_P}}[j*_num_B + i];
				if (pivot_element == 0.0)
					continue;
				const double pivot_factor = pivot_element/{{_P}}[i*_num_B + i];
				for (int k=i+1; k<_num_B; k++)
				{
					{{_P}}[j*_num_B + k] -= {{_P}}[i*_num_B + k]*pivot_factor;
				}
				{{_B}}[j] -= {{_B}}[i]*pivot_factor;
				{{_P}}[j*_num_B + i] = 0;
			}

		}

//		if (verbose_output) {
//			cout << "print P after (upper) triang:" << endl;
//				for (int i=0; i<_num_B; i++) {
//					for (int j=0; j<_num_B; j++)
//						cout << {{_P}}[i*_num_B+j] << " ";
//					cout << endl;
//				}
//		}

		// Back substitution
		for (int i=_num_B-1; i>=0; i--)
		{
			// substitute all the known values
			for (int j=_num_B-1; j>i; j--)
			{
				{{_B}}[i] -= {{_P}}[i*_num_B + j]*{{_B}}[j];
				{{_P}}[i*_num_B + j] = 0;
			}
			// divide by the diagonal element
			{{_B}}[i] /= {{_P}}[i*_num_B + i];
			{{_P}}[i*_num_B + i] = 1;
		}

	}
//	else {
//		cout << "solver not supported" << endl;
//		exit(1);
//	}
}

__global__ void kernel_{{codeobj_name}}_combine(
	unsigned int THREADS_PER_BLOCK,
	%DEVICE_PARAMETERS%
	)
{
	using namespace brian;

	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
//	unsigned int _idx = bid * THREADS_PER_BLOCK + tid;
//	unsigned int _vectorisation_idx = _idx;
	%KERNEL_VARIABLES%

	// TODO: add num_thread_check again!

	// for each branch compute the final solution by linear combination
	// of the general solution (independent: branches & compartments)


	const int _j = bid;

	// load indices into shared memory
	// TODO: correct data type for the four integers? 64 vs 32bit...
	__shared__ int _i;
	__shared__ int _i_parent;
	__shared__ int _first;
	__shared__ int _last;

	if (tid == 0) {
		_i = {{_morph_i}}[_j];
		_i_parent = {{_morph_parent_i}}[_j];
		_first = {{_starts}}[_j];
		_last = {{_ends}}[_j];
		printf("(combine kernel) bid=%d, _i=%d\n", bid, _i);
	}
// TODO: add B_i and B_i_parent to shared memory as well
	__syncthreads();


//	const unsigned int _i = {{_morph_i}}[_j];
//	const unsigned int _i_parent = {{_morph_parent_i}}[_j];
//	const unsigned int _first = {{_starts}}[_j];
//	const unsigned int _last = {{_ends}}[_j];
	printf("(combine kernel) bid=%d\n", bid);

	const unsigned int _k = _first + tid; // total index of compartment

	// only continue if thread belongs to a compartment
	if (_k > _last) // TODO: check if the condition  "|| _k >= _numv" is required, too
		return;


	// TODO: add outer loop if too few threads are there...
	//

	{{v}}[_k] = {{v_star}}[_k] + {{_B}}[_i_parent] * {{u_minus}}[_k]
							   + {{_B}}[_i] * {{u_plus}}[_k];
}

{% endblock %}

{% block kernel_call %}

//////////////////////////////////////////////////////////////////////////
// integration step 1: compute g_total and I_0 (independent: compartments)
// ERROR: this kernel is never run...
kernel_{{codeobj_name}}_integration<<<num_blocks(N),num_threads(N)>>>(
		num_threads(N),
		%HOST_PARAMETERS%
	);


//////////////////////////////////////////////////////////////////////////
// integration step 2: for each branch: solve three tridiagonal systems (independent: branches & nested: the 3 linear systems)

// create three streams to allow parallel solution of tridiagonal systems
// TODO: put this into spatialneuron_prepare
//       i.e., create/destroy streams only once per simulation not per timestep
cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

// integrate the tridiagonal system1 for each branch: solve for v_star
kernel_{{codeobj_name}}_system1<<<_num_morph_i,1,0,stream1>>>(
		1,
		%HOST_PARAMETERS%
	);
// integrate the tridiagonal system2 for each branch: solve for u_plus
kernel_{{codeobj_name}}_system2<<<_num_morph_i,1,0,stream2>>>(
		1,
		%HOST_PARAMETERS%
	);
// integrate the tridiagonal system3 for each branch: solve for u_minus
kernel_{{codeobj_name}}_system3<<<_num_morph_i,1,0,stream3>>>(
		1,
		%HOST_PARAMETERS%
	);

// continue only after the three streams have finished
cudaDeviceSynchronize(); // check if  cudaStreamSynchronize(streamX) for X=1,2,3 is faster

// destroy the streams
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaStreamDestroy(stream3);


//////////////////////////////////////////////////////////////////////////
// integration step 3: solve the coupling system (no parallelism)
kernel_{{codeobj_name}}_coupling<<<1,1>>>(
		1,
		%HOST_PARAMETERS%
	);
// TODO: use shared memory in _coupling kernel


//////////////////////////////////////////////////////////////////////////
// integration step 4: for each branch compute the final solution by
// linear combination of the general solution (independent: branches & compartments)
const int blocks_combine = _num_morph_i;
const int threads_combine = max_threads_per_block; // brian::max_threads_per_block
cout << "blocks_combine=" << blocks_combine << ", threads_combine=" << threads_combine << endl;
// ERROR: this kernel is never run....
kernel_{{codeobj_name}}_combine<<<blocks_combine,threads_combine>>>(
		threads_combine,
		%HOST_PARAMETERS%
	);
// TODO: make outer loop in _combine kernel as we might have less threads than compartments within a branch
// TODO: instead of max_threads_per_block we should use min(max_threads_per_block, max. no of compartments in a branch)
// TODO: use shared memory in _combine kernel

{% endblock %}
