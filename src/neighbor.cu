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
#define BLOCK_SIZE 128
#define DIM 3
#define NUM_OF_CELLS 50


static __device__ void warp_reduce(volatile int *s, int t)
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}


// the first step for determining whether a new neighbor list should be built
static __global__ void check_atom_distance_1
(
    int N, real d2, real *x_old, real *y_old, real *z_old,
    real *x_new, real *y_new, real *z_new, int *g_sum
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    __shared__ int s_sum[1024];
    s_sum[tid] = 0;
    if (n < N)
    {
        real dx = x_new[n] - x_old[n];
        real dy = y_new[n] - y_old[n];
        real dz = z_new[n] - z_old[n];
        if ((dx * dx + dy * dy + dz * dz) > d2) { s_sum[tid] = 1; }
    }
    __syncthreads();
    if (tid < 512) s_sum[tid] += s_sum[tid + 512]; __syncthreads();
    if (tid < 256) s_sum[tid] += s_sum[tid + 256]; __syncthreads();
    if (tid < 128) s_sum[tid] += s_sum[tid + 128]; __syncthreads();
    if (tid <  64) s_sum[tid] += s_sum[tid + 64];  __syncthreads();
    if (tid <  32) warp_reduce(s_sum, tid);
    if (tid ==  0) { g_sum[bid] = s_sum[0]; }
}


// the second step for determining whether a new neighbor list should be built
static __global__ void check_atom_distance_2(int M, int *g_sum_i, int *g_sum_o)
{
    int tid = threadIdx.x;
    int number_of_patches = (M - 1) / 1024 + 1;
    __shared__ int s_sum[1024];
    s_sum[tid] = 0;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024;
        if (n < M) { s_sum[tid] += g_sum_i[n]; }
    }
    __syncthreads();
    if (tid < 512) s_sum[tid] += s_sum[tid + 512]; __syncthreads();
    if (tid < 256) s_sum[tid] += s_sum[tid + 256]; __syncthreads();
    if (tid < 128) s_sum[tid] += s_sum[tid + 128]; __syncthreads();
    if (tid <  64) s_sum[tid] += s_sum[tid + 64];  __syncthreads();
    if (tid <  32) warp_reduce(s_sum, tid);
    if (tid ==  0) { g_sum_o[0] = s_sum[0]; }
}


// If the returned value > 0, the neighbor list will be updated.
int Atom::check_atom_distance(void)
{
    int M = (N - 1) / 1024 + 1;
    real d2 = neighbor.skin * neighbor.skin * 0.25;
    int *s1; CHECK(cudaMalloc((void**)&s1, sizeof(int) * M));
    int *s2; CHECK(cudaMalloc((void**)&s2, sizeof(int)));
    check_atom_distance_1<<<M, 1024>>>(N, d2, x0, y0, z0, x, y, z, s1);
    CUDA_CHECK_KERNEL
    check_atom_distance_2<<<1, 1024>>>(M, s1, s2);
    CUDA_CHECK_KERNEL
    int *cpu_s2; MY_MALLOC(cpu_s2, int, 1);
    CHECK(cudaMemcpy(cpu_s2, s2, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(s1));
    CHECK(cudaFree(s2));
    int update = cpu_s2[0];
    MY_FREE(cpu_s2);
    return update;
}


// pull the atoms back to the box after updating the neighbor list
static __global__ void gpu_apply_pbc
(
    int N, int triclinic, int pbc_x, int pbc_y, int pbc_z, 
    const real* __restrict__ h, real *g_x, real *g_y, real *g_z
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        if (triclinic == 0)
        {
            real lx = LDG(h, 0);
            real ly = LDG(h, 1);
            real lz = LDG(h, 2);
            if (pbc_x == 1)
            {
                if      (g_x[n] < 0)  {g_x[n] += lx;}
                else if (g_x[n] > lx) {g_x[n] -= lx;}
            }
            if (pbc_y == 1)
            {
                if      (g_y[n] < 0)  {g_y[n] += ly;}
                else if (g_y[n] > ly) {g_y[n] -= ly;}
            }
            if (pbc_z == 1)
            {
                if      (g_z[n] < 0)  {g_z[n] += lz;}
                else if (g_z[n] > lz) {g_z[n] -= lz;}
            }
        }
        else
        {
            real x = g_x[n];
            real y = g_y[n];
            real z = g_z[n];
            real sx = LDG(h,9)  * x + LDG(h,10) * y + LDG(h,11) * z;
            real sy = LDG(h,12) * x + LDG(h,13) * y + LDG(h,14) * z;
            real sz = LDG(h,15) * x + LDG(h,16) * y + LDG(h,17) * z;
            if (pbc_x == 1) sx -= nearbyint(sx);
            if (pbc_y == 1) sy -= nearbyint(sy);
            if (pbc_z == 1) sz -= nearbyint(sz);
            g_x[n] = LDG(h,0) * sx + LDG(h,1) * sy + LDG(h,2) * sz;
            g_y[n] = LDG(h,3) * sx + LDG(h,4) * sy + LDG(h,5) * sz;
            g_z[n] = LDG(h,6) * sx + LDG(h,7) * sy + LDG(h,8) * sz;
        }
    }
}


// update the reference positions:
static __global__ void gpu_update_xyz0
(int N, real *x, real *y, real *z, real *x0, real *y0, real *z0)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) { x0[n] = x[n]; y0[n] = y[n]; z0[n] = z[n]; }
}


// check the bound of the neighbor list
void Atom::check_bound(void)
{
    int *cpu_NN; MY_MALLOC(cpu_NN, int, N);
    CHECK(cudaMemcpy(cpu_NN, NN, sizeof(int)*N, cudaMemcpyDeviceToHost));
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
    MY_FREE(cpu_NN);
}


// simple version for sorting the neighbor indicies of each atom
#ifdef DEBUG
static __global__ void gpu_sort_neighbor_list(int N, int* NN, int* NL)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int neighbor_number = NN[bid];
    int atom_index;
    __shared__ int atom_index_copy[BLOCK_SIZE];
    if (tid < neighbor_number) 
    {
        atom_index = NL[bid + tid * N];
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
    int cell_n_x = 0; int cell_n_y = 0; int cell_n_z = 0;
    int use_ON2 = 0;
    if (box.pbc_x)
    {
        cell_n_x = floor(box.cpu_h[0] / neighbor.rc);
        if (cell_n_x < 3) {use_ON2 = 1;}
    }
    else {cell_n_x = 1;}
    if (box.pbc_y)
    {
        cell_n_y = floor(box.cpu_h[1] / neighbor.rc);
        if (cell_n_y < 3) {use_ON2 = 1;}
    }
    else {cell_n_y = 1;}
    if (box.pbc_z)
    {
        cell_n_z = floor(box.cpu_h[2] / neighbor.rc);
        if (cell_n_z < 3) {use_ON2 = 1;}
    }
    else {cell_n_z = 1;}
    if (cell_n_x * cell_n_y * cell_n_z < NUM_OF_CELLS) {use_ON2 = 1;}
    if (use_ON2) { find_neighbor_ON2(); }
    else
    {
        find_neighbor_ON1(cell_n_x, cell_n_y, cell_n_z);
#ifdef DEBUG
        gpu_sort_neighbor_list<<<N, BLOCK_SIZE>>>(N, NN, NL);
#endif
    }
}


// the driver function to be called outside this file
void Atom::find_neighbor(int is_first)
{
    if (is_first == 1)
    {
        find_neighbor();
        check_bound();
        gpu_update_xyz0<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
        (N, x, y, z, x0, y0, z0);
        CUDA_CHECK_KERNEL
    }
    else
    {
        int update = check_atom_distance();
        if (update != 0)
        {
            neighbor.number_of_updates++;
            find_neighbor();
            check_bound();
            gpu_apply_pbc<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
            (N, box.triclinic, box.pbc_x, box.pbc_y, box.pbc_z, box.h, x, y, z);
            CUDA_CHECK_KERNEL
            gpu_update_xyz0<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
            (N, x, y, z, x0, y0, z0);
            CUDA_CHECK_KERNEL
        }
    }
}


