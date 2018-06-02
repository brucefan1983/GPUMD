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
#include "force.cuh"



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




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}

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
        if (group_id[i] == fixed_group) { vx = ZERO; vy = ZERO; vz = ZERO; }
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
        if (group_id[i] == fixed_group) { vx = ZERO; vy = ZERO; vz = ZERO; }
        else
        {
            vx += ax * time_step_half; 
            vy += ay * time_step_half; 
            vz += az * time_step_half;
        }
        g_vx[i] = vx; g_vy[i] = vy; g_vz[i] = vz;
    }
}

// Find some thermodynamic properties:
// g_thermo[0-5] = T, U, p_x, p_y, p_z, something for myself
static __global__ void gpu_find_thermo
(
    int N, 
    int N_fixed,
    real T,
    real *g_box_length,
    real *g_mass, real *g_z, real *g_potential,
    real *g_vx, real *g_vy, real *g_vz, 
    real *g_sx, real *g_sy, real *g_sz,
    real *g_thermo
)
{
    //<<<6, MAX_THREAD>>>

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
                if (n >= N_fixed && n < N)
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
                #ifdef USE_2D
                    g_thermo[0] = s_ke[0] / (TWO * (N - N_fixed) * K_B);
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
                if (n >= N_fixed && n < N)
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
                if (n >= N_fixed && n < N)
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
                if (n >= N_fixed && n < N)
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
                g_thermo[3] = (s_sy[0] + (N - N_fixed) * K_B * T) * volume_inv;
            }
            break;
        case 4:
            __shared__ real s_sz[1024];
            s_sz[tid] = ZERO; 
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n >= N_fixed && n < N)
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
                g_thermo[4] = (s_sz[0] + (N - N_fixed) * K_B * T) * volume_inv;
            }
            break;
        case 5:
            __shared__ real s_h[1024];
            s_h[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n >= N_fixed && n < N)
                {        
                    s_h[tid] += g_z[n] * g_z[n];
                }
            }
            __syncthreads();
            if (tid < 512) s_h[tid] += s_h[tid + 512]; __syncthreads();
            if (tid < 256) s_h[tid] += s_h[tid + 256]; __syncthreads();
            if (tid < 128) s_h[tid] += s_h[tid + 128]; __syncthreads();
            if (tid <  64) s_h[tid] += s_h[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_h, tid);           
            if (tid ==  0) g_thermo[5] = s_h[0];
        break;
    }
}


static __global__ void gpu_berendsen_temperature
(
    int N,
    real temperature, 
    real coupling,
    real *g_prop, 
    real *g_vx, 
    real *g_vy, 
    real *g_vz
)
{
    //<<<(number_of_particles - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {  
        real factor = sqrt(ONE + coupling * (temperature / g_prop[0] - ONE)); 
        g_vx[i] *= factor; 
        g_vy[i] *= factor; 
        g_vz[i] *= factor;
    }
}



static __global__ void gpu_berendsen_pressure
(
    Strain strain,
    int number_of_particles,
    int pbc_x,
    int pbc_y,
    int pbc_z,
    real p0x,
    real p0y,
    real p0z,
    real p_coupling, 
    real *g_prop,
    real *g_box_length, 
    real *g_x,
    real *g_y,
    real *g_z
)
{

    //<<<(number_of_particles - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < number_of_particles)
    {  
        if (strain.compute)
        {
            real scale_factor = (g_box_length[0] + strain.rate) 
                              / g_box_length[0];
            g_x[i] *= scale_factor;
            if (i == 0) { g_box_length[0] *= scale_factor; }
        }
        else if (pbc_x == 1)
        {
            real scale_factor = ONE - p_coupling * (p0x - g_prop[2]);
            g_x[i] *= scale_factor;
            if (i == 0) { g_box_length[0] *= scale_factor; }
        }

        if (pbc_y == 1)
        {
            real scale_factor = ONE - p_coupling * (p0y - g_prop[3]);
            g_y[i] *= scale_factor;
            if (i == 1) { g_box_length[1] *= scale_factor; }
        }

        if (pbc_z == 1)
        {
            real scale_factor = ONE - p_coupling * (p0z - g_prop[4]);
            g_z[i] *= scale_factor;
            if (i == 2) { g_box_length[2] *= scale_factor; }
        }
    }

}


//integrate by one step in the NVE ensemble
static void gpu_integrate_nve
(
    Force_Model *force_model, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    int    N           = para->N;
    int    grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    real time_step   = para->time_step;
    real temperature = para->temperature;
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

    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);


    gpu_find_force(force_model, para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);


    // for the time being:
    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];

    gpu_find_thermo<<<6, 1024>>>
    (
        N, N_fixed, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    ); 
}




// integrate by one step in the NVT ensemble using the Berendsen method
static void gpu_integrate_nvt_berendsen
(
    Force_Model *force_model, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    int    N           = para->N;
    int    grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    real time_step   = para->time_step;
    real temperature = para->temperature;
    real t_coupling  = para->temperature_coupling;
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

    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);

    gpu_find_force(force_model, para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // for the time being:
    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];

    gpu_find_thermo<<<6, 1024>>>
    (
        N, N_fixed, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    ); 

    // control temperature
    gpu_berendsen_temperature<<<grid_size, BLOCK_SIZE>>>
    (N, temperature, t_coupling, thermo, vx, vy, vz);

}



// integrate by one step in the NPT ensemble using the Berendsen method
static void gpu_integrate_npt_berendsen
(
    Force_Model *force_model, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    int    N           = para->N;
    int    grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    int    pbc_x       = para->pbc_x;
    int    pbc_y       = para->pbc_y;
    int    pbc_z       = para->pbc_z;
    real time_step   = para->time_step;
    real temperature = para->temperature;
    real p0x         = para->pressure_x;
    real p0y         = para->pressure_y;
    real p0z         = para->pressure_z;
    real p_coupling  = para->pressure_coupling;
    real t_coupling  = para->temperature_coupling;
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

    // for the time being:
    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];

    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);

    gpu_find_force(force_model, para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    gpu_find_thermo<<<6, 1024>>>
    (
        N, N_fixed, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    ); 

    // control temperature
    gpu_berendsen_temperature<<<grid_size, BLOCK_SIZE>>>
    (N, temperature, t_coupling, thermo, vx, vy, vz);

    // control pressure 
    gpu_berendsen_pressure<<<grid_size, BLOCK_SIZE>>>
    (
        para->strain, N, pbc_x, pbc_y, pbc_z, p0x, p0y, p0z, p_coupling, 
        thermo, box_length, x, y, z
    );

}



// integrate by one step in the NVT ensemble using the NHC method
static void gpu_integrate_nvt_nhc
(
    Force_Model *force_model, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    int  N           = para->N;
    int  grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    real time_step   = para->time_step;
    real temperature = para->temperature;
    real *pos_eta = para->pos_nhc1;
    real *vel_eta = para->vel_nhc1;
    real *mas_eta = para->mas_nhc1;
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
    // for the time being:
    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];

    gpu_find_thermo<<<6, 1024>>>
    (
        N, N_fixed, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    );

    real *ek2;
    MY_MALLOC(ek2, real, sizeof(real) * 1);
    cudaMemcpy(ek2, thermo, sizeof(real) * 1, cudaMemcpyDeviceToHost);
    ek2[0] *= DIM * N * K_B;

    real factor = nhc(M, pos_eta, vel_eta, mas_eta, ek2[0], kT, dN, dt2);

    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>(N, vx, vy, vz, factor);

    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);

    gpu_find_force(force_model, para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    gpu_find_thermo<<<6, 1024>>>
    (
        N, N_fixed, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    );

    cudaMemcpy(ek2, thermo, sizeof(real) * 1, cudaMemcpyDeviceToHost);
    ek2[0] *= DIM * N * K_B;

    factor = nhc(M, pos_eta, vel_eta, mas_eta, ek2[0], kT, dN, dt2);

    MY_FREE(ek2);

    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>(N, vx, vy, vz, factor);

}



static __global__ void find_vc_and_ke
(
    int  *g_group_size,
    int  *g_group_size_sum,
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
            int index = offset + n;     
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
static void gpu_integrate_heat_nhc
(
    Force_Model *force_model, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    int N         = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    real time_step   = para->time_step;
    real temperature = para->temperature;
    real *pos_eta1 = para->pos_nhc1;
    real *vel_eta1 = para->vel_nhc1;
    real *pos_eta2 = para->pos_nhc2;
    real *vel_eta2 = para->vel_nhc2;
    real *mas_eta1 = para->mas_nhc1;
    real *mas_eta2 = para->mas_nhc2;
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

    int label_1 = para->heat.source;
    int label_2 = para->heat.sink;

    int Ng = para->number_of_groups;

    real kT1 = K_B * (temperature + para->heat.delta_temperature); 
    real kT2 = K_B * (temperature - para->heat.delta_temperature); 
    real dN1 = (real) DIM * cpu_data->group_size[para->heat.source];
    real dN2 = (real) DIM * cpu_data->group_size[para->heat.sink];
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
    (group_size, group_size_sum, mass, vx, vy, vz, vcx, vcy, vcz, ke);
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);

    real factor_1 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_eta1, vel_eta1, mas_eta1, ek2[label_1], kT1, dN1, dt2);
    real factor_2 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_eta2, vel_eta2, mas_eta2, ek2[label_2], kT2, dN2, dt2);
    
    gpu_scale_velocity<<<grid_size, BLOCK_SIZE>>>
    (
        N, label_1, label_2, gpu_data->label, factor_1, factor_2, 
        vcx, vcy, vcz, ke, vx, vy, vz
    );

    // veloicty-Verlet
    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);
    gpu_find_force(force_model, para, gpu_data);
    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    // NHC second
    find_vc_and_ke<<<Ng, 512>>>
    (group_size, group_size_sum, mass, vx, vy, vz, vcx, vcy, vcz, ke);
    cudaMemcpy(ek2, ke, sizeof(real) * Ng, cudaMemcpyDeviceToHost);
    factor_1 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_eta1, vel_eta1, mas_eta1, ek2[label_1], kT1, dN1, dt2);
    factor_2 = nhc(NOSE_HOOVER_CHAIN_LENGTH, 
        pos_eta2, vel_eta2, mas_eta2, ek2[label_2], kT2, dN2, dt2);
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



// integrate by one step 
void gpu_integrate
(
    Force_Model *force_model, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    switch (para->ensemble)
    {
        case 0: 
            gpu_integrate_nve(force_model, para, cpu_data, gpu_data);
            break;
        case 1: 
            gpu_integrate_nvt_berendsen(force_model, para, cpu_data, gpu_data);
            break;
        case 2: 
            gpu_integrate_npt_berendsen(force_model, para, cpu_data, gpu_data);
            break;
        case 3: 
            gpu_integrate_nvt_nhc(force_model, para, cpu_data, gpu_data);
            break;
        case 4: 
            gpu_integrate_heat_nhc(force_model, para, cpu_data, gpu_data);
            break;
        default: 
            printf("Illegal integrator!\n");
            break;
    }
}




