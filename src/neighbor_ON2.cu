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
Construct the neighbor list using the O(N^2) method.
------------------------------------------------------------------------------*/


#include "atom.cuh"
#include "error.cuh"
#include "mic.cuh"
#define BLOCK_SIZE 128


// a simple O(N^2) version of neighbor list construction
static __global__ void gpu_find_neighbor_ON2
(
    int triclinic, int pbc_x, int pbc_y, int pbc_z, int N, real cutoff_square, 
    const real* __restrict__ box, int *NN, int *NL, real *x, real *y, real *z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        real x1 = x[n1];  real y1 = y[n1];  real z1 = z[n1];
        int count = 0;
        for (int n2 = 0; n2 < N; ++n2)
        { 
            if (n2 == n1) { continue; }
            real x12 = x[n2]-x1; real y12 = y[n2]-y1; real z12 = z[n2]-z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, box, x12, y12, z12);
            real distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < cutoff_square) { NL[count++ * N + n1] = n2; }
        }
        NN[n1] = count;
    }
}


// a wrapper function of the above kernel
void Atom::find_neighbor_ON2(void)
{ 
    real rc2 = neighbor.rc * neighbor.rc; 
    gpu_find_neighbor_ON2<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        box.triclinic, box.pbc_x, box.pbc_y, box.pbc_z, N, rc2, box.h,
        NN, NL, x, y, z
    );
    CUDA_CHECK_KERNEL
}


