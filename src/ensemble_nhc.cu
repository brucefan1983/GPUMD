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
#include "ensemble_nhc.cuh"
#include "ensemble.inc"
#include "force.cuh"
#include "memory.cuh"

#define BLOCK_SIZE 128




Ensemble_NHC::Ensemble_NHC(int t, int N, real T, real Tc, real dt)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
    // position and momentum variables for one NHC
    pos_nhc1[0] = pos_nhc1[1] = pos_nhc1[2] = pos_nhc1[3] = ZERO;
    vel_nhc1[0] = vel_nhc1[2] =  ONE;
    vel_nhc1[1] = vel_nhc1[3] = -ONE;

    real tau = dt * temperature_coupling; 
    real kT = K_B * temperature;
    real dN = DIM * N;
    for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; i++)
    {
        mas_nhc1[i] = kT * tau * tau;
    }
    mas_nhc1[0] *= dN;
}




Ensemble_NHC::Ensemble_NHC
(
    int t, int source_input, int sink_input, int N1, int N2, 
    real T, real Tc, real dT, real time_step
)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
    delta_temperature = dT;
    source = source_input;
    sink = sink_input;

    // position and momentum variables for NHC
    pos_nhc1[0] = pos_nhc1[1] = pos_nhc1[2] = pos_nhc1[3] =  ZERO;
    pos_nhc2[0] = pos_nhc2[1] = pos_nhc2[2] = pos_nhc2[3] =  ZERO;
    vel_nhc1[0] = vel_nhc1[2] = vel_nhc2[0] = vel_nhc2[2] =  ONE;
    vel_nhc1[1] = vel_nhc1[3] = vel_nhc2[1] = vel_nhc2[3] = -ONE;

    real tau = time_step * temperature_coupling;
    real kT1 = K_B * (temperature + delta_temperature);
    real kT2 = K_B * (temperature - delta_temperature);
    real dN1 = DIM * N1;
    real dN2 = DIM * N2;
    for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; i++)
    {
        mas_nhc1[i] = kT1 * tau * tau;
        mas_nhc2[i] = kT2 * tau * tau;
    }
    mas_nhc1[0] *= dN1;
    mas_nhc2[0] *= dN2;

    // initialize the energies transferred from the system to the baths
    energy_transferred[0] = 0.0;
    energy_transferred[1] = 0.0;
}




Ensemble_NHC::~Ensemble_NHC(void)
{
    // nothing now
}




//The Nose-Hover thermostat integrator
//Run it on the CPU, which requires copying the kinetic energy 
//from the GPU to the CPU
static real nhc
(
    int M, real* pos_eta, real *vel_eta, real *mas_eta,
    real Ek2, real kT, real dN, real dt2_particle
)
{
    // These constants are taken from Tuckerman's book
    int n_sy = 7;
    int n_respa = 4;
    const real w[7] = {
                             0.784513610477560,
                             0.235573213359357,
                             -1.17767998417887,
                              1.31518632068391,
                             -1.17767998417887,
                             0.235573213359357,
                             0.784513610477560
                        };
                            
    real factor = 1.0; // to be accumulated

    for (int n1 = 0; n1 < n_sy; n1++)
    {
        real dt2 = dt2_particle * w[n1] / n_respa;
        real dt4 = dt2 * 0.5;
        real dt8 = dt4 * 0.5;
        for (int n2 = 0; n2 < n_respa; n2++)
        {
        
            // update velocity of the last (M - 1) thermostat:
            real G = vel_eta[M - 2] * vel_eta[M - 2] / mas_eta[M - 2] - kT;
            vel_eta[M - 1] += dt4 * G;

            // update thermostat velocities from M - 2 to 0:
            for (int m = M - 2; m >= 0; m--)
            { 
                real tmp = exp(-dt8 * vel_eta[m + 1] / mas_eta[m + 1]);
                G = vel_eta[m - 1] * vel_eta[m - 1] / mas_eta[m - 1] - kT;
                if (m == 0) { G = Ek2 - dN  * kT; }
                vel_eta[m] = tmp * (tmp * vel_eta[m] + dt4 * G);   
            }

            // update thermostat positions from M - 1 to 0:
            for (int m = M - 1; m >= 0; m--)
            { 
                pos_eta[m] += dt2 * vel_eta[m] / mas_eta[m];  
            } 

            // compute the scale factor 
            real factor_local = exp(-dt2 * vel_eta[0] / mas_eta[0]); 
            Ek2 *= factor_local * factor_local;
            factor *= factor_local;

            // update thermostat velocities from 0 to M - 2:
            for (int m = 0; m < M - 1; m++)
            { 
                real tmp = exp(-dt8 * vel_eta[m + 1] / mas_eta[m + 1]);
                G = vel_eta[m - 1] * vel_eta[m - 1] / mas_eta[m - 1] - kT;
                if (m == 0) {G = Ek2 - dN * kT;}
                vel_eta[m] = tmp * (tmp * vel_eta[m] + dt4 * G);   
            }

            // update velocity of the last (M - 1) thermostat:
            G = vel_eta[M - 2] * vel_eta[M - 2] / mas_eta[M - 2] - kT;
            vel_eta[M - 1] += dt4 * G;
        }
    }
    return factor;
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




void Ensemble_NHC::integrate_nvt_nhc
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    int  N           = para->N;
    int  grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
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

    real kT = K_B * temperature;
    real dN = (real) DIM * N; 
    real dt2 = time_step * HALF;

    const int M = NOSE_HOOVER_CHAIN_LENGTH;

    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];
    gpu_find_thermo<<<5, 1024>>>
    (
        N, N_fixed, fixed_group, label, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    );

    real *ek2;
    MY_MALLOC(ek2, real, sizeof(real) * 1);
    cudaMemcpy(ek2, thermo, sizeof(real) * 1, cudaMemcpyDeviceToHost);
    ek2[0] *= DIM * N * K_B;

    real factor = nhc(M, pos_nhc1, vel_nhc1, mas_nhc1, ek2[0], kT, dN, dt2);

    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>(N, vx, vy, vz, factor);

    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);

    force->compute(para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    gpu_find_thermo<<<5, 1024>>>
    (
        N, N_fixed, fixed_group, label, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    );

    cudaMemcpy(ek2, thermo, sizeof(real) * 1, cudaMemcpyDeviceToHost);
    ek2[0] *= DIM * N * K_B;

    factor = nhc(M, pos_nhc1, vel_nhc1, mas_nhc1, ek2[0], kT, dN, dt2);

    MY_FREE(ek2);

    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>(N, vx, vy, vz, factor);

}




static __global__ void find_vc_and_ke
(
    int  *g_group_size,
    int  *g_group_size_sum,
    int  *g_group_contents,
    real *g_mass, 
    real *g_vx, 
    real *g_vy, 
    real *g_vz, 
    real *g_vcx,
    real *g_vcy,
    real *g_vcz,
    real *g_ke
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




static __global__ void gpu_scale_velocity
(
    int number_of_particles, 
    int label_1,
    int label_2,
    int *g_atom_label, 
    real factor_1,
    real factor_2,
    real *g_vcx, 
    real *g_vcy,
    real *g_vcz,
    real *g_ke,
    real *g_vx, 
    real *g_vy, 
    real *g_vz
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




// integrate by one step, with heating and cooling, 
// using Nose-Hoover chain method
void Ensemble_NHC::integrate_heat_nhc
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    int N         = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
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
    int *group_size = gpu_data->group_size;
    int *group_size_sum = gpu_data->group_size_sum;
    int *group_contents = gpu_data->group_contents;

    int label_1 = source;
    int label_2 = sink;

    int Ng = para->number_of_groups;

    real kT1 = K_B * (temperature + delta_temperature); 
    real kT2 = K_B * (temperature - delta_temperature); 
    real dN1 = (real) DIM * cpu_data->group_size[source];
    real dN2 = (real) DIM * cpu_data->group_size[sink];
    real dt2 = time_step * HALF;

    // allocate some memory (to be improved)
    real *ek2;
    MY_MALLOC(ek2, real, sizeof(real) * Ng);
    real *vcx, *vcy, *vcz, *ke;
    cudaMalloc((void**)&vcx, sizeof(real) * Ng);
    cudaMalloc((void**)&vcy, sizeof(real) * Ng);
    cudaMalloc((void**)&vcz, sizeof(real) * Ng);
    cudaMalloc((void**)&ke, sizeof(real) * Ng);

    // NHC first
    find_vc_and_ke<<<Ng, 512>>>
    (
        group_size, group_size_sum, group_contents, 
        mass, vx, vy, vz, vcx, vcy, vcz, ke
    );
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);

    real factor_1 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_nhc1, vel_nhc1, mas_nhc1, ek2[label_1], kT1, dN1, dt2);
    real factor_2 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_nhc2, vel_nhc2, mas_nhc2, ek2[label_2], kT2, dN2, dt2);

    // accumulate the energies transferred from the system to the baths
    energy_transferred[0] += ek2[label_1] * 0.5 * (1.0 - factor_1 * factor_1);
    energy_transferred[1] += ek2[label_2] * 0.5 * (1.0 - factor_2 * factor_2);
    
    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>
    (
        N, label_1, label_2, gpu_data->label, factor_1, factor_2, 
        vcx, vcy, vcz, ke, vx, vy, vz
    );

    // veloicty-Verlet
    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);

    force->compute(para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // NHC second
    find_vc_and_ke<<<Ng, 512>>>
    (
        group_size, group_size_sum, group_contents, 
        mass, vx, vy, vz, vcx, vcy, vcz, ke
    );
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);
    factor_1 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_nhc1, vel_nhc1, mas_nhc1, ek2[label_1], kT1, dN1, dt2);
    factor_2 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_nhc2, vel_nhc2, mas_nhc2, ek2[label_2], kT2, dN2, dt2);

    // accumulate the energies transferred from the system to the baths
    energy_transferred[0] += ek2[label_1] * 0.5 * (1.0 - factor_1 * factor_1);
    energy_transferred[1] += ek2[label_2] * 0.5 * (1.0 - factor_2 * factor_2);

    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>
    (
        N, label_1, label_2, gpu_data->label, factor_1, factor_2, 
        vcx, vcy, vcz, ke, vx, vy, vz
    );

    // clean up
    MY_FREE(ek2); cudaFree(vcx); cudaFree(vcy); cudaFree(vcz); cudaFree(ke);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

}



 
void Ensemble_NHC::compute
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    if (type == 2)
    {
        integrate_nvt_nhc(para, cpu_data, gpu_data, force);
    }
    else
    {
        integrate_heat_nhc(para, cpu_data, gpu_data, force);
    }
}




