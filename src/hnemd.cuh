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




/*------------------------------------------------------------------------------
    This file will be directly included in some other files
------------------------------------------------------------------------------*/




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}




// get the total force
static __global__ void gpu_sum_force
(int N, real *g_fx, real *g_fy, real *g_fz, real *g_f)
{
    //<<<3, MAX_THREAD>>>

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int patch, n;
    int number_of_patches = (N - 1) / 1024 + 1; 

    switch (bid)
    {
        case 0:
            __shared__ real s_fx[1024];
            s_fx[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N) s_fx[tid] += g_fx[n]; 
            }
            __syncthreads();
            if (tid < 512) s_fx[tid] += s_fx[tid + 512]; __syncthreads();
            if (tid < 256) s_fx[tid] += s_fx[tid + 256]; __syncthreads();
            if (tid < 128) s_fx[tid] += s_fx[tid + 128]; __syncthreads();
            if (tid <  64) s_fx[tid] += s_fx[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_fx, tid); 
            if (tid ==  0) { g_f[0] = s_fx[0]; }                  
            break;
        case 1:
            __shared__ real s_fy[1024];
            s_fy[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N) s_fy[tid] += g_fy[n]; 
            }
            __syncthreads();
            if (tid < 512) s_fy[tid] += s_fy[tid + 512]; __syncthreads();
            if (tid < 256) s_fy[tid] += s_fy[tid + 256]; __syncthreads();
            if (tid < 128) s_fy[tid] += s_fy[tid + 128]; __syncthreads();
            if (tid <  64) s_fy[tid] += s_fy[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_fy, tid); 
            if (tid ==  0) { g_f[1] = s_fy[0]; }                  
            break;
        case 2:
            __shared__ real s_fz[1024];
            s_fz[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N) s_fz[tid] += g_fz[n]; 
            }
            __syncthreads();
            if (tid < 512) s_fz[tid] += s_fz[tid + 512]; __syncthreads();
            if (tid < 256) s_fz[tid] += s_fz[tid + 256]; __syncthreads();
            if (tid < 128) s_fz[tid] += s_fz[tid + 128]; __syncthreads();
            if (tid <  64) s_fz[tid] += s_fz[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_fz, tid); 
            if (tid ==  0) { g_f[2] = s_fz[0]; }                  
            break;
    }
}




// correct the total force
static __global__ void gpu_correct_force
(int N, real *g_fx, real *g_fy, real *g_fz, real *g_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {  
        g_fx[i] -= g_f[0] / N;
        g_fy[i] -= g_f[1] / N;
        g_fz[i] -= g_f[2] / N;
    }
}




