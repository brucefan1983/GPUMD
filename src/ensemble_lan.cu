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
The Bussi-Parrinello integrator of the Langevin thermostat:
[1] G. Bussi and M. Parrinello, Phys. Rev. E 75, 056707 (2007).
------------------------------------------------------------------------------*/


#include "ensemble_lan.cuh"
#include "force.cuh"
#include <curand_kernel.h>
#include "atom.cuh"
#include "error.cuh"
#include <vector>

#define BLOCK_SIZE 128
#define CURAND_NORMAL(a) curand_normal_double(a)


// initialize curand states
static __global__ void initialize_curand_states(curandState *state, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    // We can use a fixed seed here.
    if (n < N) { curand_init(123456, n, 0, &state[n]); }
}


Ensemble_LAN::Ensemble_LAN(int t, int fg, int N, double T, double Tc)
{
    type = t;
    fixed_group = fg;
    temperature = T;
    temperature_coupling = Tc;
    c1 = exp(-HALF/temperature_coupling);
    c2 = sqrt((1 - c1 * c1) * K_B * T);
    CHECK(cudaMalloc((void**)&curand_states, sizeof(curandState) * N));
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    initialize_curand_states<<<grid_size, BLOCK_SIZE>>>(curand_states, N);
    CUDA_CHECK_KERNEL
}


Ensemble_LAN::Ensemble_LAN
(
    int t, int fg, int source_input, int sink_input, int source_size, 
    int sink_size, int source_offset, int sink_offset, double T, double Tc, double dT
)
{
    type = t;
    fixed_group = fg;
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
    CHECK(cudaMalloc((void**)&curand_states_source,
        sizeof(curandState) * N_source));
    CHECK(cudaMalloc((void**)&curand_states_sink,
        sizeof(curandState) * N_sink));
    int grid_size_source = (N_source - 1) / BLOCK_SIZE + 1;
    int grid_size_sink   = (N_sink - 1)   / BLOCK_SIZE + 1;
    initialize_curand_states<<<grid_size_source, BLOCK_SIZE>>>
    (curand_states_source, N_source);
    CUDA_CHECK_KERNEL
    initialize_curand_states<<<grid_size_sink, BLOCK_SIZE>>>
    (curand_states_sink,   N_sink);
    CUDA_CHECK_KERNEL
    energy_transferred[0] = 0.0;
    energy_transferred[1] = 0.0;
}


Ensemble_LAN::~Ensemble_LAN(void)
{
    if (type == 3)
    {
        CHECK(cudaFree(curand_states));
    }
    else
    {
        CHECK(cudaFree(curand_states_source));
        CHECK(cudaFree(curand_states_sink));
    }
}


// global Langevin thermostatting
static __global__ void gpu_langevin
(
    curandState *g_state, int N, double c1, double c2, double *g_mass, 
    double *g_vx, double *g_vy, double *g_vz
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        curandState state = g_state[n];
        double c2m = c2 * sqrt(ONE / g_mass[n]);
        g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
        g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
        g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);
        g_state[n] = state;
    }
}


// wrapper of the above kernel
void Ensemble_LAN::integrate_nvt_lan_half(Atom *atom)
{
    // the first half of Langevin, before velocity-Verlet
    gpu_langevin<<<(atom->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (curand_states, atom->N, c1, c2, atom->mass, atom->vx, atom->vy, atom->vz);
    CUDA_CHECK_KERNEL
}


// local Langevin thermostatting 
static __global__ void gpu_langevin
(
    curandState *g_state, int N, int offset, int *g_group_contents,
    double c1, double c2, double *g_mass, double *g_vx, double *g_vy, double *g_vz
)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < N)
    {
        curandState state = g_state[m];
        int n = g_group_contents[offset + m];
        double c2m = c2 * sqrt(ONE / g_mass[n]);
        g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
        g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
        g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);
        g_state[m] = state;
    }
}


// group kinetic energy
static __global__ void find_ke
(
    int  *g_group_size, int  *g_group_size_sum, int  *g_group_contents,
    double *g_mass, double *g_vx, double *g_vy, double *g_vz, double *g_ke
)
{
    //<<<number_of_groups, 512>>>
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int group_size = g_group_size[bid];
    int offset = g_group_size_sum[bid];
    int number_of_patches = (group_size - 1) / 512 + 1; 
    __shared__ double s_ke[512]; // relative kinetic energy
    s_ke[tid] = ZERO;
    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int n = tid + patch * 512;
        if (n < group_size)
        {  
            int index = g_group_contents[offset + n];     
            double mass = g_mass[index];
            double vx = g_vx[index];
            double vy = g_vy[index];
            double vz = g_vz[index];
            s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_ke[tid] += s_ke[tid + offset]; }
        __syncthreads();
    }

    if (tid == 0)  {g_ke[bid] = s_ke[0];} // kinetic energy times 2
}


// wrapper of the above two kernels
void Ensemble_LAN::integrate_heat_lan_half(Atom *atom)
{
    int *group_size      = atom->group[0].size;
    int *group_size_sum  = atom->group[0].size_sum;
    int *group_contents  = atom->group[0].contents;
    double *mass = atom->mass;
    double *vx = atom->vx; double *vy = atom->vy; double *vz = atom->vz;
    int Ng = atom->group[0].number;

    std::vector<double> ek2(Ng);
    double *ke; CHECK(cudaMalloc((void**)&ke, sizeof(double) * Ng));
    find_ke<<<Ng, 512>>>
    (group_size, group_size_sum, group_contents, mass, vx, vy, vz, ke);
    CUDA_CHECK_KERNEL
    CHECK(cudaMemcpy(ek2.data(), ke, sizeof(double) * Ng, cudaMemcpyDeviceToHost));
    energy_transferred[0] += ek2[source] * 0.5;
    energy_transferred[1] += ek2[sink] * 0.5;
    gpu_langevin<<<(N_source - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        curand_states_source, N_source, offset_source, group_contents, 
        c1, c2_source, mass, vx, vy, vz
    );
    CUDA_CHECK_KERNEL
    gpu_langevin<<<(N_sink - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        curand_states_sink, N_sink, offset_sink, group_contents, 
        c1, c2_sink, mass, vx, vy, vz
    );
    CUDA_CHECK_KERNEL
    find_ke<<<Ng, 512>>>
    (group_size, group_size_sum, group_contents, mass, vx, vy, vz, ke);
    CUDA_CHECK_KERNEL
    CHECK(cudaMemcpy(ek2.data(), ke, sizeof(double) * Ng, cudaMemcpyDeviceToHost));
    energy_transferred[0] -= ek2[source] * 0.5;
    energy_transferred[1] -= ek2[sink] * 0.5;
    CHECK(cudaFree(ke));
}


void Ensemble_LAN::compute(Atom *atom, Force *force, Measure* measure)
{
    if (type == 3)
    {
        integrate_nvt_lan_half(atom);
        velocity_verlet(atom, force, measure);
        integrate_nvt_lan_half(atom);
        find_thermo(atom);
    }
    else
    {
        integrate_heat_lan_half(atom);
        velocity_verlet(atom, force, measure);
        integrate_heat_lan_half(atom);
    }
}


