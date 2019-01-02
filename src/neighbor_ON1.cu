/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/




#define USE_THRUST



#include "neighbor_ON1.cuh"
#include "atom.cuh"
#include "error.cuh"
#include "parameters.cuh"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>


#define BLOCK_SIZE 128
#ifdef USE_DP
    #define HALF  0.5
#else
    #define HALF  0.5f
#endif




template <int pbc_x, int pbc_y, int pbc_z>
static __device__ void dev_apply_mic
(real lx, real ly, real lz, real *x12, real *y12, real *z12)
{
    if      (pbc_x == 1 && *x12 < - lx * HALF) {*x12 += lx;}
    else if (pbc_x == 1 && *x12 > + lx * HALF) {*x12 -= lx;}
    if      (pbc_y == 1 && *y12 < - ly * HALF) {*y12 += ly;}
    else if (pbc_y == 1 && *y12 > + ly * HALF) {*y12 -= ly;}
    if      (pbc_z == 1 && *z12 < - lz * HALF) {*z12 += lz;}
    else if (pbc_z == 1 && *z12 > + lz * HALF) {*z12 -= lz;}
}




// find the cell id for an atom
static __device__ void find_cell_id
(
    real x, real y, real z, real cell_size, 
    int cell_n_x, int cell_n_y, int cell_n_z, int* cell_id
)
{
    int cell_id_x = floor(x / cell_size);
    int cell_id_y = floor(y / cell_size);
    int cell_id_z = floor(z / cell_size);
    while (cell_id_x < 0)         cell_id_x += cell_n_x;
    while (cell_id_x >= cell_n_x) cell_id_x -= cell_n_x;
    while (cell_id_y < 0)         cell_id_y += cell_n_y;
    while (cell_id_y >= cell_n_y) cell_id_y -= cell_n_y;
    while (cell_id_z < 0)         cell_id_z += cell_n_z;
    while (cell_id_z >= cell_n_z) cell_id_z -= cell_n_z;
    *cell_id =  cell_id_x + cell_n_x*cell_id_y + cell_n_x*cell_n_y*cell_id_z;
}



// find the cell id for an atom
static __device__ void find_cell_id
(
    real x, real y, real z, real cell_size, 
    int cell_n_x, int cell_n_y, int cell_n_z, 
    int *cell_id_x, int *cell_id_y, int *cell_id_z, int *cell_id
)
{
    *cell_id_x = floor(x / cell_size);
    *cell_id_y = floor(y / cell_size);
    *cell_id_z = floor(z / cell_size);
    while (*cell_id_x < 0)         *cell_id_x += cell_n_x;
    while (*cell_id_x >= cell_n_x) *cell_id_x -= cell_n_x;
    while (*cell_id_y < 0)         *cell_id_y += cell_n_y;
    while (*cell_id_y >= cell_n_y) *cell_id_y -= cell_n_y;
    while (*cell_id_z < 0)         *cell_id_z += cell_n_z;
    while (*cell_id_z >= cell_n_z) *cell_id_z -= cell_n_z;
    *cell_id = (*cell_id_x) + cell_n_x * (*cell_id_y) 
             + cell_n_x * cell_n_y * (*cell_id_z);
}




// cell_count[i] = number of atoms in the i-th cell
static __global__ void find_cell_counts
(
    int N, int* cell_count, real* x, real* y,real* z, 
    int cell_n_x, int cell_n_y, int cell_n_z, real cell_size
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int cell_id;
        find_cell_id
        (
            x[n1], y[n1], z[n1], cell_size, 
            cell_n_x, cell_n_y, cell_n_z, &cell_id
        );
        atomicAdd(&cell_count[cell_id], 1);
    }
}




// cell_contents[some index] = an atom index
static __global__ void find_cell_contents
(
    int N, int* cell_count, int* cell_count_sum, int* cell_contents, 
    real* x, real* y, real* z,
    int cell_n_x, int cell_n_y, int cell_n_z, real cell_size
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int cell_id;
        find_cell_id
        (
            x[n1], y[n1], z[n1], cell_size, 
            cell_n_x, cell_n_y, cell_n_z, &cell_id
        );
        int ind = atomicAdd(&cell_count[cell_id], 1);
        cell_contents[cell_count_sum[cell_id] + ind] = n1;
    }
}




// a simple (but 100% correct) version of prefix sum (used for testing)
#ifndef USE_THRUST
static __global__ void prefix_sum
(int N_cells, int* cell_count, int* cell_count_sum)
{
    //<<< 1,1 >>>
    cell_count_sum[0] = 0;
    for (int i=1; i<N_cells; ++i) 
    cell_count_sum[i] = cell_count_sum[i-1] + cell_count[i-1];
}
#endif




// new version (faster)
template<int pbc_x, int pbc_y, int pbc_z>
static __global__ void gpu_find_neighbor_ON1
(
    int N, int* cell_counts, int* cell_count_sum, int* cell_contents, 
    int* NN, int* NL,
    real* x, real* y, real* z, int cell_n_x, int cell_n_y, int cell_n_z, 
    real *box, real cutoff, real cutoff_square
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;

    int count = 0;
    if (n1 < N)
    {
        real lx = box[0];
        real ly = box[1];
        real lz = box[2];

        real x1 = x[n1];
        real y1 = y[n1];
        real z1 = z[n1];
        
        int cell_id;
        int cell_id_x;
        int cell_id_y;
        int cell_id_z;
        find_cell_id
        (
            x1, y1, z1, cutoff, cell_n_x, cell_n_y, cell_n_z, 
            &cell_id_x, &cell_id_y, &cell_id_z, &cell_id
        );

        int klim = pbc_z ? 1 : 0;
        int jlim = pbc_y ? 1 : 0;
        int ilim = pbc_x ? 1 : 0;
       
        // loop over the neighbor cells of the central cell
        for (int k=-klim; k<klim+1; ++k)
        {
            for (int j=-jlim; j<jlim+1; ++j)
            {
                for (int i=-ilim; i<ilim+1; ++i)
                {
                    int neighbour=cell_id+k*cell_n_x*cell_n_y+j*cell_n_x+i;
                    if (cell_id_x + i < 0)
                        neighbour += cell_n_x;
                    if (cell_id_x + i >= cell_n_x) 
                        neighbour -= cell_n_x;
                    if (cell_id_y + j < 0)
                        neighbour += cell_n_y*cell_n_x;
                    if (cell_id_y + j >= cell_n_y) 
                        neighbour -= cell_n_y*cell_n_x;
                    if (cell_id_z + k < 0) 
                        neighbour += cell_n_z*cell_n_y*cell_n_x;
                    if (cell_id_z + k >= cell_n_z) 
                        neighbour -= cell_n_z*cell_n_y*cell_n_x;
                        
                    // loop over the atoms in a neighbor cell
                    for (int k = 0; k < cell_counts[neighbour]; ++k)
                    {
                        int n2 = cell_contents[cell_count_sum[neighbour] + k];
                        if (n1 == n2) continue;

                        real x12 = x[n2]-x1;
                        real y12 = y[n2]-y1;
                        real z12 = z[n2]-z1;

                        dev_apply_mic<pbc_x, pbc_y, pbc_z>
                        (lx, ly, lz, &x12, &y12, &z12);

                        real d2 = x12*x12 + y12*y12 + z12*z12;
                        if (d2 < cutoff_square)
                        {        
                            NL[count * N + n1] = n2;
                            count++;
                        }
                    }
                }
            }
        }
        NN[n1] = count;
    }
}




// a driver function
void find_neighbor_ON1
(Parameters *para, Atom *atom, int cell_n_x, int cell_n_y, int cell_n_z)
{                           
    int N = atom->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1; 
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    real rc = para->neighbor.rc;
    real rc2 = rc * rc; 
    int *NN = atom->NN;
    int *NL = atom->NL;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *box = atom->box_length;

    int N_cells = cell_n_x * cell_n_y * cell_n_z;

    // some local data
    int* cell_count;
    int* cell_count_sum;
    int* cell_contents;
    CHECK(cudaMalloc((void**)&cell_count, sizeof(int)*N_cells));
    CHECK(cudaMemset(cell_count, 0, sizeof(int)*N_cells));
    CHECK(cudaMalloc((void**)&cell_count_sum, sizeof(int)*N_cells));
    CHECK(cudaMemset(cell_count_sum, 0, sizeof(int)*N_cells));
    CHECK(cudaMalloc((void**)&cell_contents, sizeof(int)*N));
    CHECK(cudaMemset(cell_contents, 0, sizeof(int)*N));

    // Find the number of particles in each cell
    find_cell_counts<<<grid_size, BLOCK_SIZE>>>
    (N, cell_count, x, y, z, cell_n_x, cell_n_y, cell_n_z, rc);
    #ifdef DEBUG
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    #endif

#ifndef USE_THRUST
    // Simple (but 100% correct) version of prefix sum
    prefix_sum<<<1, 1>>>(N_cells, cell_count, cell_count_sum);
#else
    // use thrust to calculate the prefix sum
    thrust::exclusive_scan
    (thrust::device, cell_count, cell_count + N_cells, cell_count_sum);
#endif
    // reset to zero
    CHECK(cudaMemset(cell_count, 0, sizeof(int)*N_cells));
	
    // Create particle list for each cell
    find_cell_contents<<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, 
        x, y, z, cell_n_x, cell_n_y, cell_n_z, rc
    );
    #ifdef DEBUG
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    #endif

    // new version (faster)

    if (pbc_x && pbc_y && pbc_z)
    gpu_find_neighbor_ON1<1, 1, 1><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (pbc_x && pbc_y && !pbc_z)
    gpu_find_neighbor_ON1<1, 1, 0><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (pbc_x && !pbc_y && pbc_z)
    gpu_find_neighbor_ON1<1, 0, 1><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (!pbc_x && pbc_y && pbc_z)
    gpu_find_neighbor_ON1<0, 1, 1><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (pbc_x && !pbc_y && !pbc_z)
    gpu_find_neighbor_ON1<1, 0, 0><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (!pbc_x && pbc_y && !pbc_z)
    gpu_find_neighbor_ON1<0, 1, 0><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (!pbc_x && !pbc_y && pbc_z)
    gpu_find_neighbor_ON1<0, 0, 1><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );

    if (!pbc_x && !pbc_y && !pbc_z)
    gpu_find_neighbor_ON1<0, 0, 0><<<grid_size, BLOCK_SIZE>>>
    (
        N, cell_count, cell_count_sum, cell_contents, NN, NL, x, y, z, 
        cell_n_x, cell_n_y, cell_n_z, box, rc, rc2
    );


    #ifdef DEBUG
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    #endif

    CHECK(cudaFree(cell_count));
    CHECK(cudaFree(cell_count_sum));
    CHECK(cudaFree(cell_contents));
}



