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


/*----------------------------------------------------------------------------80
Construct the neighbor list using the O(N) method.
Written by Ville Vierimaa and optimized by Zheyong Fan.
------------------------------------------------------------------------------*/


#include "atom.cuh"
#include "utilities/error.cuh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#define USE_THRUST


// find the cell id for an atom
static __device__ void find_cell_id
(
    double x, double y, double z, double cell_size, 
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
    double x, double y, double z, double cell_size, 
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
    int N, int* cell_count, double* x, double* y,double* z, 
    int cell_n_x, int cell_n_y, int cell_n_z, double cell_size
)
{
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
    double* x, double* y, double* z,
    int cell_n_x, int cell_n_y, int cell_n_z, double cell_size
)
{
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


// construct the Verlet neighbor list from the cell list
static __global__ void gpu_find_neighbor_ON1
(
    const Box box,
    const int N,
    const int* __restrict__ cell_counts,
    const int* __restrict__ cell_count_sum,
    const int* __restrict__ cell_contents,
    int* NN,
    int* NL,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const int cell_n_x,
    const int cell_n_y,
    const int cell_n_z,
    const double cutoff,
    const double cutoff_square
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    if (n1 < N)
    {
        double x1 = x[n1];
        double y1 = y[n1];
        double z1 = z[n1];
        int cell_id;
        int cell_id_x;
        int cell_id_y;
        int cell_id_z;
        find_cell_id
        (
            x1, y1, z1, cutoff, cell_n_x, cell_n_y, cell_n_z, 
            &cell_id_x, &cell_id_y, &cell_id_z, &cell_id
        );
        int klim = box.pbc_z ? 1 : 0;
        int jlim = box.pbc_y ? 1 : 0;
        int ilim = box.pbc_x ? 1 : 0;

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

                    // number of atoms in the preceding cells
                    int offset = cell_count_sum[neighbour];
                    // number of atoms in the current cell
                    int M = cell_counts[neighbour];

                    // loop over the atoms in a neighbor cell
                    for (int m = 0; m < M; ++m)
                    {
                        int n2 = cell_contents[offset + m];
                        double x12 = x[n2] - x1;
                        double y12 = y[n2] - y1;
                        double z12 = z[n2] - z1;
                        apply_mic(box, x12, y12, z12);
                        double d2 = x12*x12 + y12*y12 + z12*z12;

                        if (n1 != n2 && d2 < cutoff_square)
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


// a wrapper of the above kernels
void Neighbor::find_neighbor_ON1
(
    int cell_n_x,
    int cell_n_y,
    int cell_n_z,
    const Box& box,
    double* x,
    double* y,
    double* z
)
{
    const int N = NN.size();
    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;
 
    double rc2 = rc * rc; 
    int N_cells = cell_n_x * cell_n_y * cell_n_z;

    CHECK(cudaMemset(cell_count.data(), 0, sizeof(int)*N_cells));
    CHECK(cudaMemset(cell_count_sum.data(), 0, sizeof(int)*N_cells));
    CHECK(cudaMemset(cell_contents.data(), 0, sizeof(int)*N));

    find_cell_counts<<<grid_size, block_size>>>
    (N, cell_count.data(), x, y, z, cell_n_x, cell_n_y, cell_n_z, rc);
    CUDA_CHECK_KERNEL

#ifndef USE_THRUST
    prefix_sum<<<1, 1>>>(N_cells, cell_count.data(), cell_count_sum.data());
    CUDA_CHECK_KERNEL
#else
    thrust::exclusive_scan
    (
        thrust::device,
        cell_count.data(),
        cell_count.data() + N_cells,
        cell_count_sum.data()
    );
#endif

    CHECK(cudaMemset(cell_count.data(), 0, sizeof(int)*N_cells));

    find_cell_contents<<<grid_size, block_size>>>
    (
        N, cell_count.data(), cell_count_sum.data(), cell_contents.data(),
        x, y, z, cell_n_x, cell_n_y, cell_n_z, rc
    );
    CUDA_CHECK_KERNEL

    gpu_find_neighbor_ON1<<<grid_size, block_size>>>
    (
        box,
        N,
        cell_count.data(),
        cell_count_sum.data(),
        cell_contents.data(),
        NN.data(),
        NL.data(),
        x,
        y,
        z,
        cell_n_x,
        cell_n_y,
        cell_n_z,
        rc,
        rc2
    );
    CUDA_CHECK_KERNEL
}


