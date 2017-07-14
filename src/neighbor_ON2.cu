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




#include "common.h"
#include "neighbor_ON2.h"




// apply the minimum image convention 
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




// a simple O(N^2) version of neighbor list construction
template <int pbc_x, int pbc_y, int pbc_z>
static __global__ void gpu_find_neighbor_ON2
(
    int N, real cutoff_square, real *box_length, 
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
            dev_apply_mic<pbc_x, pbc_y, pbc_z>
            (box_length[0], box_length[1], box_length[2], &x12, &y12, &z12);
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
void find_neighbor_ON2(Parameters *para, GPU_Data *gpu_data)
{                           
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1; 
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    real rc = para->neighbor.rc;
    real rc2 = rc * rc; 
    int *NN = gpu_data->NN;
    int *NL = gpu_data->NL;
    real *x = gpu_data->x;
    real *y = gpu_data->y;
    real *z = gpu_data->z;
    real *box = gpu_data->box_length;
    
    // Find neighbours
    if (pbc_x && pbc_y && pbc_z)
    gpu_find_neighbor_ON2<1,1,1><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    if (pbc_x && pbc_y && !pbc_z)
    gpu_find_neighbor_ON2<1,1,0><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    if (pbc_x && !pbc_y && pbc_z)
    gpu_find_neighbor_ON2<1,0,1><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    if (!pbc_x && pbc_y && pbc_z)
    gpu_find_neighbor_ON2<0,1,1><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    if (pbc_x && !pbc_y && !pbc_z)
    gpu_find_neighbor_ON2<1,0,0><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    if (!pbc_x && pbc_y && !pbc_z)
    gpu_find_neighbor_ON2<0,1,0><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);
        
    if (!pbc_x && !pbc_y && pbc_z)
    gpu_find_neighbor_ON2<0,0,1><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    if (!pbc_x && !pbc_y && !pbc_z)
    gpu_find_neighbor_ON2<0,0,0><<<grid_size, BLOCK_SIZE>>>
    (N, rc2, box, NN, NL, x, y, z);

    #ifdef DEBUG
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    #endif
}



