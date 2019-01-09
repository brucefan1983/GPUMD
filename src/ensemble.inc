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




#define DIM 3
#ifndef USE_SP
    #define ZERO  0.0
    #define HALF  0.5
    #define ONE   1.0
    #define K_B   8.617343e-5
#else
    #define ZERO  0.0f
    #define HALF  0.5f
    #define ONE   1.0f
    #define K_B   8.617343e-5
#endif




// This file will be included into ensemble_nve.cu, ensemble_ber.cu, and
// ensemble_nhc.cu.




// The first step of velocity-Verlet
static __global__ void gpu_velocity_verlet_1
(
    int number_of_particles,
    int fixed_group,
    int *group_id, 
    real g_time_step,
    real* g_mass,
    real* g_x,  real* g_y,  real* g_z, 
    real* g_vx, real* g_vy, real* g_vz,
    real* g_fx, real* g_fy, real* g_fz
)
{
    //<<<(number_of_particles - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_particles)
    {
        real time_step = g_time_step;
        real time_step_half = time_step * HALF;
        real x  = g_x[i];  real y  = g_y[i];  real z  = g_z[i]; 
        real vx = g_vx[i]; real vy = g_vy[i]; real vz = g_vz[i];
        real mass_inv = ONE / g_mass[i];
        real ax = g_fx[i] * mass_inv;
        real ay = g_fy[i] * mass_inv;
        real az = g_fz[i] * mass_inv;
        if (group_id[i] == fixed_group) 
        {
            vx = ZERO;
#ifdef ZHEN_LI // special version for Zhen Li
            vy += ay * time_step_half;
            vz += az * time_step_half;
#else
            vy = ZERO;
            vz = ZERO;
#endif 
        }
        else
        {
            vx += ax * time_step_half;
            vy += ay * time_step_half;
            vz += az * time_step_half;
        }
        x += vx * time_step; y += vy * time_step;z += vz * time_step; 
        g_x[i]  = x;  g_y[i]  = y;  g_z[i]  = z;
        g_vx[i] = vx; g_vy[i] = vy; g_vz[i] = vz; 
    }
}




// The second step of velocity-Verlet
static __global__ void gpu_velocity_verlet_2
(
    int number_of_particles, 
    int fixed_group,
    int *group_id,
    real g_time_step,
    real* g_mass,
    real* g_vx, real* g_vy, real* g_vz,
    real* g_fx, real* g_fy, real* g_fz
)
{
    //<<<(number_of_particles - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < number_of_particles)
    {
        real time_step_half = g_time_step * HALF;  
        real vx = g_vx[i]; real vy = g_vy[i]; real vz = g_vz[i];
        real mass_inv = ONE / g_mass[i];
        real ax = g_fx[i] * mass_inv;
        real ay = g_fy[i] * mass_inv;
        real az = g_fz[i] * mass_inv;
        if (group_id[i] == fixed_group)
        {
            vx = ZERO;
#ifdef ZHEN_LI // special version for Zhen Li
            vy += ay * time_step_half;
            vz += az * time_step_half;
#else
            vy = ZERO;
            vz = ZERO;
#endif
        }
        else
        {
            vx += ax * time_step_half; 
            vy += ay * time_step_half; 
            vz += az * time_step_half;
        }
        g_vx[i] = vx; g_vy[i] = vy; g_vz[i] = vz;
    }
}




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}




// Find some thermodynamic properties:
// g_thermo[0-4] = T, U, p_x, p_y, p_z
static __global__ void gpu_find_thermo
(
    int N, 
    int N_fixed,
    int fixed_group,
    int *group_id,
    real T,
    real *g_box_length,
    real *g_mass, real *g_z, real *g_potential,
    real *g_vx, real *g_vy, real *g_vz, 
    real *g_sx, real *g_sy, real *g_sz,
    real *g_thermo
)
{
    //<<<5, MAX_THREAD>>>

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int patch, n;
    int number_of_patches = (N - 1) / 1024 + 1; 
    real mass, vx, vy, vz;

    switch (bid)
    {
        case 0:
            __shared__ real s_ke[1024];
            s_ke[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N && group_id[n] != fixed_group)
                {        
                    mass = g_mass[n];
                    vx = g_vx[n]; vy = g_vy[n]; vz = g_vz[n]; 
                    s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
                }
            }
            __syncthreads();
            if (tid < 512) s_ke[tid] += s_ke[tid + 512]; __syncthreads();
            if (tid < 256) s_ke[tid] += s_ke[tid + 256]; __syncthreads();
            if (tid < 128) s_ke[tid] += s_ke[tid + 128]; __syncthreads();
            if (tid <  64) s_ke[tid] += s_ke[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_ke, tid); 
            if (tid ==  0) 
            {
#ifdef ZHEN_LI // special version for Zhen Li
                    g_thermo[0] = s_ke[0] / ((DIM * N - N_fixed) * K_B);
#else
                    g_thermo[0] = s_ke[0] / (DIM * (N - N_fixed) * K_B);
#endif  
            }                  
            break;
        case 1:
            __shared__ real s_pe[1024];
            s_pe[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N && group_id[n] != fixed_group)
                {          
                    s_pe[tid] += g_potential[n];
                }
            }
            __syncthreads();
            if (tid < 512) s_pe[tid] += s_pe[tid + 512]; __syncthreads();
            if (tid < 256) s_pe[tid] += s_pe[tid + 256]; __syncthreads();
            if (tid < 128) s_pe[tid] += s_pe[tid + 128]; __syncthreads();
            if (tid <  64) s_pe[tid] += s_pe[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_pe, tid); 
            if (tid ==  0) g_thermo[1] = s_pe[0];
            break;
        case 2:
            __shared__ real s_sx[1024];
            s_sx[tid] = ZERO; 
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N && group_id[n] != fixed_group)
                {        
                    s_sx[tid] += g_sx[n]; 
                }
            }
            __syncthreads();
            if (tid < 512) s_sx[tid] += s_sx[tid + 512]; __syncthreads();
            if (tid < 256) s_sx[tid] += s_sx[tid + 256]; __syncthreads();
            if (tid < 128) s_sx[tid] += s_sx[tid + 128]; __syncthreads();
            if (tid <  64) s_sx[tid] += s_sx[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_sx, tid);  
            if (tid == 0) 
            { 
                real volume_inv 
                    = ONE / (g_box_length[0]*g_box_length[1]*g_box_length[2]);
                g_thermo[2] = (s_sx[0] + (N - N_fixed) * K_B * T) * volume_inv;
            }
            break;
        case 3:
            __shared__ real s_sy[1024];
            s_sy[tid] = ZERO; 
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N && group_id[n] != fixed_group)
                {        
                    s_sy[tid] += g_sy[n]; 
                }
            }
            __syncthreads();
            if (tid < 512) s_sy[tid] += s_sy[tid + 512]; __syncthreads();
            if (tid < 256) s_sy[tid] += s_sy[tid + 256]; __syncthreads();
            if (tid < 128) s_sy[tid] += s_sy[tid + 128]; __syncthreads();
            if (tid <  64) s_sy[tid] += s_sy[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_sy, tid);  
            if (tid == 0) 
            { 
                real volume_inv 
                    = ONE / (g_box_length[0]*g_box_length[1]*g_box_length[2]);
#ifdef ZHEN_LI // special version for Zhen Li
                g_thermo[3] = (s_sy[0] + N * K_B * T) * volume_inv;
#else
                g_thermo[3] = (s_sy[0] + (N - N_fixed) * K_B * T) * volume_inv;
#endif
            }
            break;
        case 4:
            __shared__ real s_sz[1024];
            s_sz[tid] = ZERO; 
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N && group_id[n] != fixed_group)
                {        
                    s_sz[tid] += g_sz[n]; 
                }
            }
            __syncthreads();
            if (tid < 512) s_sz[tid] += s_sz[tid + 512]; __syncthreads();
            if (tid < 256) s_sz[tid] += s_sz[tid + 256]; __syncthreads();
            if (tid < 128) s_sz[tid] += s_sz[tid + 128]; __syncthreads();
            if (tid <  64) s_sz[tid] += s_sz[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_sz, tid);  
            if (tid == 0) 
            { 
                real volume_inv 
                    = ONE / (g_box_length[0]*g_box_length[1]*g_box_length[2]);
#ifdef ZHEN_LI // special version for Zhen Li
                g_thermo[4] = (s_sz[0] + N * K_B * T) * volume_inv;
#else
                g_thermo[4] = (s_sz[0] + (N - N_fixed) * K_B * T) * volume_inv;
#endif
            }
            break;
    }
}




