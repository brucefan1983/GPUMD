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
#include "ensemble_lan.cuh"
#include "ensemble.inc"
#include "force.cuh"
#include <curand_kernel.h>
#include "memory.cuh"

#define BLOCK_SIZE 128




#ifdef USE_DP
    #define CURAND_NORMAL(a) curand_normal_double(a)
#else
    #define CURAND_NORMAL(a) curand_normal(a)
#endif




// initialize curand states
static __global__ void initialize_curand_states(curandState *state, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    // We can use a fixed seed here.
    if (n < N) { curand_init(12345678, n, 0, &state[n]); }
}




Ensemble_LAN::Ensemble_LAN(int t, int N, real T, real Tc)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
    c1 = exp(-HALF/temperature_coupling);
    c2 = sqrt((1 - c1 * c1) * K_B * T);
    cudaMalloc((void**)&curand_states, sizeof(curandState) * N);
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    initialize_curand_states<<<grid_size, BLOCK_SIZE>>>(curand_states, N);
}




Ensemble_LAN::Ensemble_LAN
(
    int t, int source_input, int sink_input, int source_size, int sink_size, 
    int source_offset, int sink_offset, real T, real Tc, real dT
)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
    delta_temperature = dT;
    source = source_input;
    sink   = sink_input;
    N_source = source_size;
    N_sink = sink_size;
    offset_source = source_offset; 
    offset_sink = sink_offset;
    c1 = exp(-HALF/temperature_coupling);
    c2_source = sqrt((1 - c1 * c1) * K_B * (T + dT));
    c2_sink   = sqrt((1 - c1 * c1) * K_B * (T - dT));

    cudaMalloc((void**)&curand_states_source, sizeof(curandState) * N_source);
    cudaMalloc((void**)&curand_states_sink,   sizeof(curandState) * N_sink);

    int grid_size_source = (N_source - 1) / BLOCK_SIZE + 1;
    int grid_size_sink   = (N_sink - 1)   / BLOCK_SIZE + 1;
    initialize_curand_states<<<grid_size_source, BLOCK_SIZE>>>
    (curand_states_source, N_source);
    initialize_curand_states<<<grid_size_sink, BLOCK_SIZE>>>
    (curand_states_sink,   N_sink);

    energy_transferred[0] = 0.0;
    energy_transferred[1] = 0.0;
}




Ensemble_LAN::~Ensemble_LAN(void)
{
    if (type == 5)
    {
        cudaFree(curand_states);
    }
    else
    {
        cudaFree(curand_states_source);
        cudaFree(curand_states_sink);
    }
}




// global Langevin thermostatting
static __global__ void gpu_langevin
(
    curandState *g_state, int N, real c1, real c2, real *g_mass, 
    real *g_vx, real *g_vy, real *g_vz
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        // get the curand state
        curandState state = g_state[n];

        real c2m = c2 * sqrt(ONE / g_mass[n]);
        g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
        g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
        g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);

        // save the curand state
        g_state[n] = state;
    }
}




// local Langevin thermostatting 
static __global__ void gpu_langevin
(
    curandState *g_state, int N, int offset, int *g_group_contents,
    real c1, real c2, real *g_mass, real *g_vx, real *g_vy, real *g_vz
)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < N)
    {
        // get the curand state
        curandState state = g_state[m];

        int n = g_group_contents[offset + m];
        real c2m = c2 * sqrt(ONE / g_mass[n]);
        g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
        g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
        g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);

        // save the curand state
        g_state[m] = state;
    }
}




// group kinetic energy
static __global__ void find_ke
(
    int  *g_group_size,
    int  *g_group_size_sum,
    int  *g_group_contents,
    real *g_mass,
    real *g_vx, 
    real *g_vy, 
    real *g_vz,
    real *g_ke
)
{
    //<<<number_of_groups, 512>>>

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int group_size = g_group_size[bid];
    int offset = g_group_size_sum[bid];
    int number_of_patches = (group_size - 1) / 512 + 1; 
    __shared__ real s_ke[512]; // relative kinetic energy
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
            s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
    }
    __syncthreads();

    if (tid < 256) {s_ke[tid] += s_ke[tid + 256];} __syncthreads();
    if (tid < 128) {s_ke[tid] += s_ke[tid + 128];} __syncthreads();
    if (tid <  64) {s_ke[tid] += s_ke[tid + 64];}  __syncthreads();
    if (tid <  32) {warp_reduce(s_ke, tid);}  
    if (tid == 0)  {g_ke[bid] = s_ke[0];} // kinetic energy times 2
}




void Ensemble_LAN::integrate_nvt_lan
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force, Measure* measure)
{
    int  N           = para->N;
    int  grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group  = para->fixed_group;
    int *label       = gpu_data->label;
    real time_step   = para->time_step;
    real *mass = gpu_data->mass;
    real *x    = gpu_data->x;
    real *y    = gpu_data->y;
    real *z    = gpu_data->z;
    real *vx   = gpu_data->vx;
    real *vy   = gpu_data->vy;
    real *vz   = gpu_data->vz;
    real *fx   = gpu_data->fx;
    real *fy   = gpu_data->fy;
    real *fz   = gpu_data->fz;
    real *potential_per_atom = gpu_data->potential_per_atom;
    real *virial_per_atom_x  = gpu_data->virial_per_atom_x; 
    real *virial_per_atom_y  = gpu_data->virial_per_atom_y;
    real *virial_per_atom_z  = gpu_data->virial_per_atom_z;
    real *thermo             = gpu_data->thermo;
    real *box_length         = gpu_data->box_length;

    // the first half of Langevin, before velocity-Verlet
    gpu_langevin<<<grid_size, BLOCK_SIZE>>>
    (curand_states, N, c1, c2, mass, vx, vy, vz);

    // the standard velocity-Verlet
    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);
    force->compute(para, gpu_data, measure);
    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // the second half of Langevin, after velocity-Verlet
    gpu_langevin<<<grid_size, BLOCK_SIZE>>>
    (curand_states, N, c1, c2, mass, vx, vy, vz);

    // thermo
    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];
    gpu_find_thermo<<<5, 1024>>>
    (
        N, N_fixed, fixed_group, label, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    );
}




// integrate by one step, with heating and cooling
void Ensemble_LAN::integrate_heat_lan
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force, Measure* measure)
{
    int N                = para->N;
    int grid_size        = (N - 1) / BLOCK_SIZE + 1;
    int grid_size_source = (N_source - 1) / BLOCK_SIZE + 1;
    int grid_size_sink   = (N_sink - 1)   / BLOCK_SIZE + 1;
    int fixed_group      = para->fixed_group;
    int *label           = gpu_data->label;
    int *group_size      = gpu_data->group_size;
    int *group_size_sum  = gpu_data->group_size_sum;
    int *group_contents  = gpu_data->group_contents;
    real time_step       = para->time_step;
    real *mass = gpu_data->mass;
    real *x    = gpu_data->x;
    real *y    = gpu_data->y;
    real *z    = gpu_data->z;
    real *vx   = gpu_data->vx;
    real *vy   = gpu_data->vy;
    real *vz   = gpu_data->vz;
    real *fx   = gpu_data->fx;
    real *fy   = gpu_data->fy;
    real *fz   = gpu_data->fz;

    int label_1 = source;
    int label_2 = sink;
    int Ng = para->number_of_groups;

    // allocate some memory
    real *ek2;
    MY_MALLOC(ek2, real, sizeof(real) * Ng);
    real *ke;
    cudaMalloc((void**)&ke, sizeof(real) * Ng);

    // the first half of Langevin, before velocity-Verlet
    find_ke<<<Ng, 512>>>
    (group_size, group_size_sum, group_contents, mass, vx, vy, vz, ke);
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);
    energy_transferred[0] += ek2[label_1] * 0.5;
    energy_transferred[1] += ek2[label_2] * 0.5;

    gpu_langevin<<<grid_size_source, BLOCK_SIZE>>>
    (
        curand_states_source, N_source, offset_source, group_contents, 
        c1, c2_source, mass, vx, vy, vz
    );
    gpu_langevin<<<grid_size_sink, BLOCK_SIZE>>>
    (
        curand_states_sink, N_sink, offset_sink, group_contents, 
        c1, c2_sink, mass, vx, vy, vz
    );

    find_ke<<<Ng, 512>>>
    (group_size, group_size_sum, group_contents, mass, vx, vy, vz, ke);
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);
    energy_transferred[0] -= ek2[label_1] * 0.5;
    energy_transferred[1] -= ek2[label_2] * 0.5;

    // the standard veloicty-Verlet
    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);
    force->compute(para, gpu_data, measure);
    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // the second half of Langevin, after velocity-Verlet
    find_ke<<<Ng, 512>>>
    (group_size, group_size_sum, group_contents, mass, vx, vy, vz, ke);
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);
    energy_transferred[0] += ek2[label_1] * 0.5;
    energy_transferred[1] += ek2[label_2] * 0.5;

    gpu_langevin<<<grid_size_source, BLOCK_SIZE>>>
    (
        curand_states_source, N_source, offset_source, group_contents, 
        c1, c2_source, mass, vx, vy, vz
    );
    gpu_langevin<<<grid_size_sink, BLOCK_SIZE>>>
    (
        curand_states_sink, N_sink, offset_sink, group_contents, 
        c1, c2_sink, mass, vx, vy, vz
    );

    find_ke<<<Ng, 512>>>
    (group_size, group_size_sum, group_contents, mass, vx, vy, vz, ke);
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);
    energy_transferred[0] -= ek2[label_1] * 0.5;
    energy_transferred[1] -= ek2[label_2] * 0.5;

    // clean up
    MY_FREE(ek2); cudaFree(ke);
}



 
void Ensemble_LAN::compute
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force, Measure* measure)
{
    if (type == 3)
    {
        integrate_nvt_lan(para, cpu_data, gpu_data, force, measure);
    }
    else
    {
        integrate_heat_lan(para, cpu_data, gpu_data, force, measure);
    }
}




