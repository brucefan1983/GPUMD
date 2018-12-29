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




#include "common.cuh"
#include "validate.cuh"
#include "force.cuh"




// This choice gives optimal accuracy for finite-difference calculations
#define DX1 1.0e-7
#define DX2 2.0e-7




static FILE *my_fopen(const char *filename, const char *mode)
{
    FILE *fid = fopen(filename, mode);
    if (fid == NULL) 
    {
        printf ("Failed to open %s!\n", filename);
        printf ("%s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return fid;
}




// copy from xyz to xyz0
static __global__ void copy_positions
(int N, real *xi, real *yi, real *zi, real *xo, real *yo, real *zo)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {  
        xo[n] = xi[n];
        yo[n] = yi[n];
        zo[n] = zi[n];  
    }
}


// move one atom left or right
static __global__ void shift_atom
(
    int d, int n, int direction, 
    real *x0, real *y0, real *z0, real *x, real *y, real *z
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


static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}


// get the total potential form the per-atom potentials
static __global__ void sum_potential(int N, int m, real *p, real *p_sum)
{
    int tid = threadIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1; 
    
    __shared__ real s_sum[1024];
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
    if (tid < 512) s_sum[tid] += s_sum[tid + 512]; __syncthreads();
    if (tid < 256) s_sum[tid] += s_sum[tid + 256]; __syncthreads();
    if (tid < 128) s_sum[tid] += s_sum[tid + 128]; __syncthreads();
    if (tid <  64) s_sum[tid] += s_sum[tid + 64];  __syncthreads();
    if (tid <  32) warp_reduce(s_sum, tid); 
    
    if (tid ==  0) 
    {
        p_sum[m] = s_sum[0]; 
    }       		
}


// get the forces from the potential energies using finite difference
static __global__ void find_force_from_potential
(int N, real *p1, real *p2, real *fx, real *fy, real *fz)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    int m;
    if (n1 < N)
    {
        m = n1;
        fx[n1] = (p1[m] - p2[m]) / DX2;

        m += N;
        fy[n1] = (p1[m] - p2[m]) / DX2;

        m += N;
        fz[n1] = (p1[m] - p2[m]) / DX2; 
    }
}


void validate_force
(Force *force, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1; 
    int M = sizeof(real) * N;
    real *x = gpu_data->x;
    real *y = gpu_data->y;
    real *z = gpu_data->z;
    real *fx = cpu_data->fx;
    real *fy = cpu_data->fy;
    real *fz = cpu_data->fz;

    // first calculate the forces directly:
    force->compute(para, gpu_data);

    // make a copy of the positions
    real *x0, *y0, *z0;
    cudaMalloc((void**)&x0, M);
    cudaMalloc((void**)&y0, M);
    cudaMalloc((void**)&z0, M);
    copy_positions<<<grid_size, BLOCK_SIZE>>>(N, x, y, z, x0, y0, z0);
    
    // get the potentials
    real *p1, *p2;
    cudaMalloc((void**)&p1, M * 3);
    cudaMalloc((void**)&p2, M * 3);
    for (int d = 0; d < 3; ++d)
    {
        for (int n = 0; n < N; ++n)
        {
            int m = d * N + n;

            // shift one atom to the left by a small amount
            shift_atom<<<grid_size, BLOCK_SIZE>>>
            (d, n, 1, x0, y0, z0, x, y, z);

            // get the potential energy
            force->compute(para, gpu_data);

            // sum up the potential energy
            sum_potential<<<1, 1024>>>(N, m, gpu_data->potential_per_atom, p1); 

            // shift one atom to the right by a small amount
            shift_atom<<<grid_size, BLOCK_SIZE>>>
            (d, n, 2, x0, y0, z0, x, y, z);

            // get the potential energy
            force->compute(para, gpu_data);

            // sum up the potential energy
            sum_potential<<<1, 1024>>>(N, m, gpu_data->potential_per_atom, p2);     
        }
    }

    // copy the positions back (as if nothing happens)
    copy_positions<<<grid_size, BLOCK_SIZE>>>(N, x0, y0, z0, x, y, z);

    // get the forces from the potential energies using finite difference
    real *fx_compare, *fy_compare, *fz_compare;
    cudaMalloc((void**)&fx_compare, M);
    cudaMalloc((void**)&fy_compare, M);
    cudaMalloc((void**)&fz_compare, M);
    find_force_from_potential<<<grid_size, BLOCK_SIZE>>>
    (N, p1, p2, fx_compare, fy_compare, fz_compare);

    // open file
    FILE *fid = my_fopen("f_compare.out", "w");
    
    // output the forces from direct calculations
    CHECK(cudaMemcpy(cpu_data->fx, gpu_data->fx, M, cudaMemcpyDeviceToHost)); 
    CHECK(cudaMemcpy(cpu_data->fy, gpu_data->fy, M, cudaMemcpyDeviceToHost)); 
    CHECK(cudaMemcpy(cpu_data->fz, gpu_data->fz, M, cudaMemcpyDeviceToHost));    
    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%25.15e%25.15e%25.15e\n", fx[n], fy[n], fz[n]);
    }
 
    // output the forces from finite difference
    CHECK(cudaMemcpy(cpu_data->fx, fx_compare, M, cudaMemcpyDeviceToHost)); 
    CHECK(cudaMemcpy(cpu_data->fy, fy_compare, M, cudaMemcpyDeviceToHost)); 
    CHECK(cudaMemcpy(cpu_data->fz, fz_compare, M, cudaMemcpyDeviceToHost));   
    for (int n = 0; n < N; n++)
    {
        fprintf(fid, "%25.15e%25.15e%25.15e\n", fx[n], fy[n], fz[n]);
    }
    
    // close file
    fflush(fid);
    fclose(fid); 
    
    // free memory
    cudaFree(x0);  
    cudaFree(y0); 
    cudaFree(z0); 
    cudaFree(p1); 
    cudaFree(p2); 
    cudaFree(fx_compare); 
    cudaFree(fy_compare); 
    cudaFree(fz_compare);  
}


