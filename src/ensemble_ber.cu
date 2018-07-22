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
#include "ensemble_ber.cuh"
#include "ensemble.inc"
#include "force.cuh"



Ensemble_BER::Ensemble_BER(int t, real T, real Tc)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
}



Ensemble_BER::Ensemble_BER
(int t, real T, real Tc, real px, real py, real pz, real pc)
{
    type = t;
    temperature = T;
    temperature_coupling = Tc;
    pressure_x = px;
    pressure_y = py;
    pressure_z = pz;
    pressure_coupling = pc;
}



Ensemble_BER::~Ensemble_BER(void)
{
    // nothing now
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




void Ensemble_BER::compute
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data, Force *force)
{
    int N           = para->N;
    int grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = gpu_data->label;
    int  pbc_x       = para->pbc_x;
    int  pbc_y       = para->pbc_y;
    int  pbc_z       = para->pbc_z;
    real time_step   = para->time_step;
    real p0x         = pressure_x;
    real p0y         = pressure_y;
    real p0z         = pressure_z;
    real p_coupling  = pressure_coupling;
    real t_coupling  = temperature_coupling;
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

    force->compute(para, gpu_data);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    int N_fixed = (fixed_group == -1) ? 0 : cpu_data->group_size[fixed_group];
    gpu_find_thermo<<<5, 1024>>>
    (
        N, N_fixed, fixed_group, label, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    ); 

    // control temperature
    gpu_berendsen_temperature<<<grid_size, BLOCK_SIZE>>>
    (N, temperature, t_coupling, thermo, vx, vy, vz);

    // control pressure 
    if (type == 2)
    {
        gpu_berendsen_pressure<<<grid_size, BLOCK_SIZE>>>
        (
            para->strain, N, pbc_x, pbc_y, pbc_z, p0x, p0y, p0z, p_coupling, 
            thermo, box_length, x, y, z
        );
    }
}




