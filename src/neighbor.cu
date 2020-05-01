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
Construct the neighbor list, choosing the O(N) or O(N^2) method automatically
------------------------------------------------------------------------------*/


#include "atom.cuh"
#include "error.cuh"
#include "gpu_vector.cuh"

const int NUM_OF_CELLS = 50; // use the O(N^2) method when #cells < this number


// determining whether a new neighbor list should be built
static __global__ void gpu_check_atom_distance
(
    int N, double d2, double *x_old, double *y_old, double *z_old,
    double *x_new, double *y_new, double *z_new, int *g_sum
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    __shared__ int s_sum[1024];
    s_sum[tid] = 0;
    if (n < N)
    {
        double dx = x_new[n] - x_old[n];
        double dy = y_new[n] - y_old[n];
        double dz = z_new[n] - z_old[n];
        if ((dx * dx + dy * dy + dz * dz) > d2) { s_sum[tid] = 1; }
    }
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_sum[tid] += s_sum[tid + offset]; }
        __syncthreads();
    }

    if (tid == 0) { atomicAdd(g_sum, s_sum[0]); }
}


// If the returned value > 0, the neighbor list will be updated.
int Atom::check_atom_distance(void)
{
    int M = (N - 1) / 1024 + 1;
    double d2 = neighbor.skin * neighbor.skin * 0.25;
    GPU_Vector<int> s2(1);
    int cpu_s2[1] = {0};
    s2.copy_from_host(cpu_s2);
    gpu_check_atom_distance<<<M, 1024>>>(N, d2, x0, y0, z0, x, y, z, s2.data());
    CUDA_CHECK_KERNEL
    s2.copy_to_host(cpu_s2);
    return cpu_s2[0];
}


// pull the atoms back to the box after updating the neighbor list
static __global__ void gpu_apply_pbc
(
    int N, Box box, double *g_x, double *g_y, double *g_z
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        if (box.triclinic == 0)
        {
            double lx = box.cpu_h[0];
            double ly = box.cpu_h[1];
            double lz = box.cpu_h[2];
            if (box.pbc_x == 1)
            {
                if      (g_x[n] < 0)  {g_x[n] += lx;}
                else if (g_x[n] > lx) {g_x[n] -= lx;}
            }
            if (box.pbc_y == 1)
            {
                if      (g_y[n] < 0)  {g_y[n] += ly;}
                else if (g_y[n] > ly) {g_y[n] -= ly;}
            }
            if (box.pbc_z == 1)
            {
                if      (g_z[n] < 0)  {g_z[n] += lz;}
                else if (g_z[n] > lz) {g_z[n] -= lz;}
            }
        }
        else
        {
            double x = g_x[n];
            double y = g_y[n];
            double z = g_z[n];
            double sx = box.cpu_h[9]  * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
            double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
            double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
            if (box.pbc_x == 1) sx -= nearbyint(sx);
            if (box.pbc_y == 1) sy -= nearbyint(sy);
            if (box.pbc_z == 1) sz -= nearbyint(sz);
            g_x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
            g_y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
            g_z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
        }
    }
}


// update the reference positions:
static __global__ void gpu_update_xyz0
(int N, double *x, double *y, double *z, double *x0, double *y0, double *z0)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) { x0[n] = x[n]; y0[n] = y[n]; z0[n] = z[n]; }
}


// check the bound of the neighbor list
void Atom::check_bound(void)
{
    std::vector<int> cpu_NN(N);
    CHECK(cudaMemcpy(cpu_NN.data(), NN, sizeof(int)*N, cudaMemcpyDeviceToHost));
    int flag = 0;
    for (int n = 0; n < N; ++n)
    {
        if (cpu_NN[n] > neighbor.MN)
        {
            printf("Error: NN[%d] = %d > %d\n", n, cpu_NN[n], neighbor.MN);
            flag = 1;
        }
    }
    if (flag == 1) { exit(1); }
}


// simple version for sorting the neighbor indicies of each atom
#ifdef DEBUG
static __global__ void gpu_sort_neighbor_list
(const int N, const int *NN, int *NL)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int neighbor_number = NN[bid];
    int atom_index;
    extern __shared__ int atom_index_copy[];

    if (tid < neighbor_number) 
    {
        atom_index = LDG(NL, bid + tid * N);
        atom_index_copy[tid] = atom_index;
    }
    int count = 0;
    __syncthreads();

    for (int j = 0; j < neighbor_number; ++j)
    {
        if (atom_index > atom_index_copy[j]) { count++; }
    }

    if (tid < neighbor_number) { NL[bid + count * N] = atom_index; }
}
#endif


void Atom::find_neighbor(void)
{
    bool use_ON2 = false;
    int cell_n_x = 0;
    int cell_n_y = 0;
    int cell_n_z = 0;

    if (box.triclinic)
    {
        use_ON2 = true;
    }
    else
    {
        if (box.pbc_x)
        {
            cell_n_x = floor(box.cpu_h[0] / neighbor.rc);
            if (cell_n_x < 3) {use_ON2 = true;}
        }
        else {cell_n_x = 1;}

        if (box.pbc_y)
        {
            cell_n_y = floor(box.cpu_h[1] / neighbor.rc);
            if (cell_n_y < 3) {use_ON2 = true;}
        }
        else {cell_n_y = 1;}

        if (box.pbc_z)
        {
            cell_n_z = floor(box.cpu_h[2] / neighbor.rc);
            if (cell_n_z < 3) {use_ON2 = true;}
        }
        else {cell_n_z = 1;}

        if (cell_n_x * cell_n_y * cell_n_z < NUM_OF_CELLS) {use_ON2 = true;}
    }
	
    int num_bins = cell_n_x * cell_n_y * cell_n_z;
    if (num_bins > N)
    {
        PRINT_INPUT_ERROR("Number of bins is larger than number of atoms.\n");
    }

    if (use_ON2)
    {
        find_neighbor_ON2();
    }
    else
    {
        find_neighbor_ON1(cell_n_x, cell_n_y, cell_n_z);
#ifdef DEBUG
        const int smem = neighbor.MN * sizeof(int);
        gpu_sort_neighbor_list<<<N, neighbor.MN, smem>>>(N, NN, NL);
#endif
    }
}


// the driver function to be called outside this file
void Atom::find_neighbor(int is_first)
{
    const int block_size = 256;
    const int grid_size = (N - 1) / block_size + 1;

    if (is_first == 1)
    {
        find_neighbor();
        check_bound();

        gpu_update_xyz0<<<grid_size, block_size>>>(N, x, y, z, x0, y0, z0);
        CUDA_CHECK_KERNEL
    }
    else
    {
        int update = check_atom_distance();

        if (update)
        {
            neighbor.number_of_updates++;

            find_neighbor();
            check_bound();

            gpu_apply_pbc<<<grid_size, block_size>>>(N, box, x, y, z);
            CUDA_CHECK_KERNEL

            gpu_update_xyz0<<<grid_size, block_size>>>(N, x, y, z, x0, y0, z0);
            CUDA_CHECK_KERNEL
        }
    }
}


