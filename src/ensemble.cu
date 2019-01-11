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




#include "ensemble.cuh"

#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128
#define DIM 3




Ensemble::Ensemble(void)
{
    // nothing now
}



Ensemble::~Ensemble(void)
{
    // nothing now
}




// The first step of velocity-Verlet
static __global__ void gpu_velocity_verlet_1
(
    int number_of_particles, int fixed_group, int *group_id, real g_time_step,
    real* g_mass, real* g_x, real* g_y, real* g_z, real* g_vx, real* g_vy,
    real* g_vz, real* g_fx, real* g_fy, real* g_fz
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




// wrapper of the above kernel
void Ensemble::velocity_verlet_1(Atom* atom)
{
    gpu_velocity_verlet_1<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        atom->N, atom->fixed_group, atom->label, atom->time_step, atom->mass,
        atom->x, atom->y, atom->z, atom->vx, atom->vy, atom->vz,
        atom->fx, atom->fy, atom->fz
    );
    CUDA_CHECK_KERNEL
}




// The second step of velocity-Verlet
static __global__ void gpu_velocity_verlet_2
(
    int number_of_particles, int fixed_group, int *group_id, real g_time_step,
    real* g_mass, real* g_vx, real* g_vy, real* g_vz,
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




// wrapper of the above kernel
void Ensemble::velocity_verlet_2(Atom* atom)
{
    gpu_velocity_verlet_2<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        atom->N, atom->fixed_group, atom->label, atom->time_step, atom->mass,
        atom->vx, atom->vy, atom->vz, atom->fx, atom->fy, atom->fz
    );
    CUDA_CHECK_KERNEL
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
    int N, int N_fixed, int fixed_group, int *group_id, real T,
    real *g_box_length, real *g_mass, real *g_potential, real *g_vx,
    real *g_vy, real *g_vz, real *g_sx, real *g_sy, real *g_sz, real *g_thermo
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




// wrapper of the above kernel
void Ensemble::find_thermo(Atom* atom)
{
    int N_fixed = (atom->fixed_group == -1) ? 0 :
        atom->cpu_group_size[atom->fixed_group];
    gpu_find_thermo<<<5, 1024>>>
    (
        atom->N, N_fixed, atom->fixed_group, atom->label, temperature,
        atom->box_length, atom->mass, atom->potential_per_atom,
        atom->vx, atom->vy, atom->vz, atom->virial_per_atom_x,
        atom->virial_per_atom_y, atom->virial_per_atom_z, atom->thermo
    );
    CUDA_CHECK_KERNEL
}




// Scale the velocity of every particle in the systems by a factor
static void __global__ gpu_scale_velocity
(int N, real *g_vx, real *g_vy, real *g_vz, real factor)
{
    //<<<(number_of_particles - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        g_vx[i] *= factor;
        g_vy[i] *= factor;
        g_vz[i] *= factor;
    }
}




// wrapper of the above kernel
void Ensemble::scale_velocity_global(Atom* atom, real factor)
{
    gpu_scale_velocity<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (atom->N, atom->vx, atom->vy, atom->vz, factor);
    CUDA_CHECK_KERNEL
}




static __global__ void gpu_find_vc_and_ke
(
    int* g_group_size, int* g_group_size_sum, int* g_group_contents, 
    real* g_mass, real *g_vx, real *g_vy, real *g_vz, 
    real *g_vcx, real *g_vcy, real *g_vcz, real *g_ke
)
{
    //<<<number_of_groups, 512>>>

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int group_size = g_group_size[bid];
    int offset = g_group_size_sum[bid];
    int number_of_patches = (group_size - 1) / 512 + 1; 

    __shared__ real s_mc[512]; // center of mass
    __shared__ real s_vx[512]; // center of mass velocity
    __shared__ real s_vy[512];
    __shared__ real s_vz[512];
    __shared__ real s_ke[512]; // relative kinetic energy

    s_mc[tid] = ZERO;
    s_vx[tid] = ZERO;
    s_vy[tid] = ZERO;
    s_vz[tid] = ZERO;
    s_ke[tid] = ZERO;
    
    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int n = tid + patch * 512;
        if (n < group_size)
        {  
            int index = g_group_contents[offset + n];     
            real mass = g_mass[index];
            real vx = g_vx[index];
            real vy = g_vy[index];
            real vz = g_vz[index];

            s_mc[tid] += mass;
            s_vx[tid] += mass * vx;
            s_vy[tid] += mass * vy;
            s_vz[tid] += mass * vz;
            s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
    }
    __syncthreads();

    if (tid < 256) 
    { 
        s_mc[tid] += s_mc[tid + 256]; 
        s_vx[tid] += s_vx[tid + 256];
        s_vy[tid] += s_vy[tid + 256];
        s_vz[tid] += s_vz[tid + 256];
        s_ke[tid] += s_ke[tid + 256];
    } 
    __syncthreads();

    if (tid < 128) 
    { 
        s_mc[tid] += s_mc[tid + 128]; 
        s_vx[tid] += s_vx[tid + 128];
        s_vy[tid] += s_vy[tid + 128];
        s_vz[tid] += s_vz[tid + 128];
        s_ke[tid] += s_ke[tid + 128];
    } 
    __syncthreads();

    if (tid <  64) 
    { 
        s_mc[tid] += s_mc[tid + 64]; 
        s_vx[tid] += s_vx[tid + 64];
        s_vy[tid] += s_vy[tid + 64];
        s_vz[tid] += s_vz[tid + 64];
        s_ke[tid] += s_ke[tid + 64];
    } 
    __syncthreads();

    if (tid <  32) 
    { 
        warp_reduce(s_mc, tid);  
        warp_reduce(s_vx, tid); 
        warp_reduce(s_vy, tid); 
        warp_reduce(s_vz, tid);    
        warp_reduce(s_ke, tid);       
    }  

    if (tid == 0) 
    { 
        real mc = s_mc[0];
        real vx = s_vx[0] / mc;
        real vy = s_vy[0] / mc;
        real vz = s_vz[0] / mc;
        g_vcx[bid] = vx; // center of mass velocity
        g_vcy[bid] = vy;
        g_vcz[bid] = vz;

        // relative kinetic energy times 2
        g_ke[bid] = (s_ke[0] - mc * (vx * vx + vy * vy + vz * vz));  
    }
}




// wrapper of the above kernel
void Ensemble::find_vc_and_ke
(Atom* atom, real* vcx, real* vcy, real* vcz, real* ke)
{
    gpu_find_vc_and_ke<<<atom->number_of_groups, 512>>>
    (
        atom->group_size, atom->group_size_sum, atom->group_contents, 
        atom->mass, atom->vx, atom->vy, atom->vz, vcx, vcy, vcz, ke
    );
    CUDA_CHECK_KERNEL
}




static __global__ void gpu_scale_velocity
(
    int number_of_particles, int label_1, int label_2, int *g_atom_label, 
    real factor_1, real factor_2, real *g_vcx, real *g_vcy, real *g_vcz,
    real *g_ke, real *g_vx, real *g_vy, real *g_vz
)
{
    // <<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number_of_particles)
    {
        int atom_label = g_atom_label[n];     
                 
        if (atom_label == label_1) 
        {
            // center of mass velocity for the source
            real vcx = g_vcx[atom_label]; 
            real vcy = g_vcy[atom_label];
            real vcz = g_vcz[atom_label];  

            // momentum is conserved
            g_vx[n] = vcx + factor_1 * (g_vx[n] - vcx);
            g_vy[n] = vcy + factor_1 * (g_vy[n] - vcy);
            g_vz[n] = vcz + factor_1 * (g_vz[n] - vcz);
        }
        if (atom_label == label_2)
        {
            // center of mass velocity for the sink
            real vcx = g_vcx[atom_label]; 
            real vcy = g_vcy[atom_label];
            real vcz = g_vcz[atom_label];  

            // momentum is conserved
            g_vx[n] = vcx + factor_2 * (g_vx[n] - vcx);
            g_vy[n] = vcy + factor_2 * (g_vy[n] - vcy);
            g_vz[n] = vcz + factor_2 * (g_vz[n] - vcz);
        }
    }
}




// wrapper of the above kernel
void Ensemble::scale_velocity_local
(
    Atom* atom, real factor_1, real factor_2,
    real* vcx, real* vcy, real* vcz, real* ke
)
{
    gpu_scale_velocity<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        atom->N, source, sink, atom->label, factor_1, factor_2, 
        vcx, vcy, vcz, ke, atom->vx, atom->vy, atom->vz
    );
    CUDA_CHECK_KERNEL
}




