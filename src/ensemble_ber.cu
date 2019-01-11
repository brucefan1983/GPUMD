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




#include "ensemble_ber.cuh"

#include "force.cuh"
#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128




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
    int deform_x, int deform_y, int deform_z, real deform_rate,
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
        if (deform_x)
        {
            real scale_factor = g_box_length[0];
            scale_factor = (scale_factor + deform_rate) / scale_factor;
            g_x[i] *= scale_factor;
            if (i == 0) { g_box_length[0] *= scale_factor; }
        }
        else if (pbc_x == 1)
        {
            real scale_factor = ONE - p_coupling * (p0x - g_prop[2]);
            g_x[i] *= scale_factor;
            if (i == 0) { g_box_length[0] *= scale_factor; }
        }

        if (deform_y)
        {
            real scale_factor = g_box_length[1];
            scale_factor = (scale_factor + deform_rate) / scale_factor;
            g_y[i] *= scale_factor;
            if (i == 1) { g_box_length[1] *= scale_factor; }
        }
        else if (pbc_y == 1)
        {
            real scale_factor = ONE - p_coupling * (p0y - g_prop[3]);
            g_y[i] *= scale_factor;
            if (i == 1) { g_box_length[1] *= scale_factor; }
        }

        if (deform_z)
        {
            real scale_factor = g_box_length[2];
            scale_factor = (scale_factor + deform_rate) / scale_factor;
            g_z[i] *= scale_factor;
            if (i == 2) { g_box_length[2] *= scale_factor; }
        }
        else if (pbc_z == 1)
        {
            real scale_factor = ONE - p_coupling * (p0z - g_prop[4]);
            g_z[i] *= scale_factor;
            if (i == 2) { g_box_length[2] *= scale_factor; }
        }
    }
}




void Ensemble_BER::compute
(Atom *atom, Force *force, Measure* measure)
{
    int N           = atom->N;
    int grid_size   = (N - 1) / BLOCK_SIZE + 1;


    int  pbc_x       = atom->pbc_x;
    int  pbc_y       = atom->pbc_y;
    int  pbc_z       = atom->pbc_z;

    real p0x         = pressure_x;
    real p0y         = pressure_y;
    real p0z         = pressure_z;
    real p_coupling  = pressure_coupling;
    real t_coupling  = temperature_coupling;
    real *x    = atom->x;
    real *y    = atom->y;
    real *z    = atom->z;
    real *vx   = atom->vx;
    real *vy   = atom->vy;
    real *vz   = atom->vz;
    real *thermo             = atom->thermo;
    real *box_length         = atom->box_length;

    velocity_verlet_1(atom);
    force->compute(atom, measure);
    velocity_verlet_2(atom);
    find_thermo(atom);

    // control temperature
    gpu_berendsen_temperature<<<grid_size, BLOCK_SIZE>>>
    (N, temperature, t_coupling, thermo, vx, vy, vz);
    CUDA_CHECK_KERNEL

    // control pressure 
    if (type == 11)
    {
        gpu_berendsen_pressure<<<grid_size, BLOCK_SIZE>>>
        (
            atom->deform_x, atom->deform_y, atom->deform_z, atom->deform_rate,
            N, pbc_x, pbc_y, pbc_z, p0x, p0y, p0z, p_coupling, 
            thermo, box_length, x, y, z
        );
        CUDA_CHECK_KERNEL
    }
}




