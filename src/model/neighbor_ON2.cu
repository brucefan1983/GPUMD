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


#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "mic.cuh"


// a simple O(N^2) version of neighbor list construction
static __global__ void gpu_find_neighbor_ON2
(
    Box box, int N, double cutoff_square, 
    int *NN, int *NL, double *x, double *y, double *z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;

    if (n1 < N)
    {
        double x1 = x[n1];
        double y1 = y[n1];
        double z1 = z[n1];
        int count = 0;

        for (int n2 = 0; n2 < N; ++n2)
        { 
            double x12 = x[n2] - x1;
            double y12 = y[n2] - y1;
            double z12 = z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d2 = x12 * x12 + y12 * y12 + z12 * z12;

            if (n1 != n2 && d2 < cutoff_square)
            {
                NL[count++ * N + n1] = n2;
            }
        }
        NN[n1] = count;
    }
}


// a wrapper function of the above kernel
void Neighbor::find_neighbor_ON2
(
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

    gpu_find_neighbor_ON2<<<grid_size, block_size>>>
    (
        box,
        N,
        rc2,
        NN.data(),
        NL.data(),
        x,
        y,
        z
    );
    CUDA_CHECK_KERNEL
}


