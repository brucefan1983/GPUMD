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
Use finite difference to validate the analytical force calculations.
------------------------------------------------------------------------------*/


#include "validate.cuh"
#include "force.cuh"
#include "atom.cuh"
#include "gpu_vector.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128

// This choice gives optimal accuracy for finite-difference calculations
#define DX1 1.0e-7
#define DX2 2.0e-7


// move one atom left or right
static __global__ void shift_atom
(
    int d, int n, int direction, 
    double *x0, double *y0, double *z0, double *x, double *y, double *z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 == n)
    {
        if (d == 0)
        {
            if (direction == 1)
            {
                x[n] = x0[n] - DX1;
            }
            else
            {
                x[n] = x0[n] + DX1;
            }
        }
        else if (d == 1)
        {
            if (direction == 1)
            {
                y[n] = y0[n] - DX1;
            }
            else
            {
                y[n] = y0[n] + DX1;
            }
        }
        else
        {
            if (direction == 1)
            {
                z[n] = z0[n] - DX1;
            }
            else
            {
                z[n] = z0[n] + DX1;
            }
        } 
    }
}


// get the total potential form the per-atom potentials
static __global__ void sum_potential(int N, int m, double *p, double *p_sum)
{
    int tid = threadIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1; 
    
    __shared__ double s_sum[1024];
    s_sum[tid] = 0;
    
    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int n = tid + patch * 1024;
        if (n < N)
        {        
            s_sum[tid] += p[n];
        }
    }
    
    __syncthreads();
    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_sum[tid] += s_sum[tid + offset]; }
        __syncthreads();
    } 

    if (tid ==  0) 
    {
        p_sum[m] = s_sum[0]; 
    }
}


// get the forces from the potential energies using finite difference
static __global__ void find_force_from_potential
(int N, double *p1, double *p2, double *fx, double *fy, double *fz)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    int m;
    if (n1 < N)
    {
        m = n1; fx[n1] = (p1[m] - p2[m]) / DX2;
        m += N; fy[n1] = (p1[m] - p2[m]) / DX2;
        m += N; fz[n1] = (p1[m] - p2[m]) / DX2; 
    }
}


void validate_force(Force *force, Atom *atom)
{
    int N = atom->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1; 
    int M = sizeof(double) * N;
    double *x = atom->x;
    double *y = atom->y;
    double *z = atom->z;
    std::vector<double> fx(N);
    std::vector<double> fy(N);
    std::vector<double> fz(N);

    // first calculate the forces directly:
    force->compute(atom);

    // make a copy of the positions
    GPU_Vector<double> x0(N), y0(N), z0(N);
    x0.copy_from_device(x);
    y0.copy_from_device(y);
    z0.copy_from_device(z);

    // get the potentials
    GPU_Vector<double> p1(N * 3), p2(N * 3);
    for (int d = 0; d < 3; ++d)
    {
        for (int n = 0; n < N; ++n)
        {
            int m = d * N + n;

            // shift one atom to the left by a small amount
            shift_atom<<<grid_size, BLOCK_SIZE>>>
            (d, n, 1, x0.data(), y0.data(), z0.data(), x, y, z);
            CUDA_CHECK_KERNEL

            // get the potential energy
            force->compute(atom);

            // sum up the potential energy
            sum_potential<<<1, 1024>>>(N, m, atom->potential_per_atom, p1.data());
            CUDA_CHECK_KERNEL

            // shift one atom to the right by a small amount
            shift_atom<<<grid_size, BLOCK_SIZE>>>
            (d, n, 2, x0.data(), y0.data(), z0.data(), x, y, z);
            CUDA_CHECK_KERNEL

            // get the potential energy
            force->compute(atom);

            // sum up the potential energy
            sum_potential<<<1, 1024>>>(N, m, atom->potential_per_atom, p2.data());
            CUDA_CHECK_KERNEL
        }
    }

    // copy the positions back (as if nothing happens)
    x0.copy_to_device(x);
    y0.copy_to_device(y);
    z0.copy_to_device(z);

    // get the forces from the potential energies using finite difference
    GPU_Vector<double> fx_compare(N), fy_compare(N), fz_compare(N);
    find_force_from_potential<<<grid_size, BLOCK_SIZE>>>
    (N, p1.data(), p2.data(), fx_compare.data(), fy_compare.data(), fz_compare.data());
    CUDA_CHECK_KERNEL

    // open file
    FILE *fid = my_fopen("f_compare.out", "w");
    
    // output the forces from direct calculations
    CHECK(cudaMemcpy(fx.data(), atom->fx, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fy.data(), atom->fy, M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fz.data(), atom->fz, M, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%25.15e%25.15e%25.15e\n", fx[n], fy[n], fz[n]);
    }
 
    // output the forces from finite difference
    CHECK(cudaMemcpy(fx.data(), fx_compare.data(), M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fy.data(), fy_compare.data(), M, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(fz.data(), fz_compare.data(), M, cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%25.15e%25.15e%25.15e\n", fx[n], fy[n], fz[n]);
    }
    
    // close file
    fflush(fid);
    fclose(fid); 
}


