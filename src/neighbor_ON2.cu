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




#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128
#ifdef USE_DP
    #define HALF  0.5
#else
    #define HALF  0.5f
#endif




static __device__ void dev_apply_mic
(
    int pbc_x, int pbc_y, int pbc_z, real &x12, real &y12, real &z12, 
    real lx, real ly, real lz
)
{
    if      (pbc_x == 1 && x12 < - lx * HALF) {x12 += lx;}
    else if (pbc_x == 1 && x12 > + lx * HALF) {x12 -= lx;}
    if      (pbc_y == 1 && y12 < - ly * HALF) {y12 += ly;}
    else if (pbc_y == 1 && y12 > + ly * HALF) {y12 -= ly;}
    if      (pbc_z == 1 && z12 < - lz * HALF) {z12 += lz;}
    else if (pbc_z == 1 && z12 > + lz * HALF) {z12 -= lz;}
}




// a simple O(N^2) version of neighbor list construction
static __global__ void gpu_find_neighbor_ON2
(
    int pbc_x, int pbc_y, int pbc_z,
    int N, real cutoff_square, 
    real *box,
    int *NN, int *NL, real *x, real *y, real *z
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    if (n1 < N)
    {  
        real x1 = x[n1];   
        real y1 = y[n1];
        real z1 = z[n1];  
        for (int n2 = 0; n2 < N; ++n2)
        { 
            if (n2 == n1) { continue; }
            real x12  = x[n2] - x1;  
            real y12  = y[n2] - y1;
            real z12  = z[n2] - z1;

            dev_apply_mic
            (pbc_x, pbc_y, pbc_z, x12, y12, z12, box[0], box[1], box[2]);

            real distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < cutoff_square)
            {        
                NL[count * N + n1] = n2;
                ++count;
            }
        }
        NN[n1] = count;
    }
}




// a driver function
void Atom::find_neighbor_ON2(void)
{
    int grid_size = (N - 1) / BLOCK_SIZE + 1; 
    real rc = neighbor.rc;
    real rc2 = rc * rc; 
    real *box = box_length;

    // Find neighbours
    gpu_find_neighbor_ON2<<<grid_size, BLOCK_SIZE>>>
    (pbc_x, pbc_y, pbc_z, N, rc2, box, NN, NL, x, y, z);
    CUDA_CHECK_KERNEL
}



