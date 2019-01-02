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
#include "ensemble_nve.cuh"
#include "ensemble.inc"
#include "force.cuh"
#include "atom.cuh"
#include "parameters.cuh"

#define BLOCK_SIZE 128




Ensemble_NVE::Ensemble_NVE(int t)
{
    type = t;
}



Ensemble_NVE::~Ensemble_NVE(void)
{
    // nothing now
}




void Ensemble_NVE::compute
(Parameters *para, Atom *atom, Force *force, Measure* measure)
{
    int    N           = para->N;
    int    grid_size   = (N - 1) / BLOCK_SIZE + 1;
    int fixed_group = para->fixed_group;
    int *label = atom->label;
    real time_step   = para->time_step;
    real *mass = atom->mass;
    real *x    = atom->x;
    real *y    = atom->y;
    real *z    = atom->z;
    real *vx   = atom->vx;
    real *vy   = atom->vy;
    real *vz   = atom->vz;
    real *fx   = atom->fx;
    real *fy   = atom->fy;
    real *fz   = atom->fz;
    real *potential_per_atom = atom->potential_per_atom;
    real *virial_per_atom_x  = atom->virial_per_atom_x; 
    real *virial_per_atom_y  = atom->virial_per_atom_y;
    real *virial_per_atom_z  = atom->virial_per_atom_z;
    real *thermo             = atom->thermo;
    real *box_length         = atom->box_length;

    gpu_velocity_verlet_1<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, x,  y,  z, vx, vy, vz, fx, fy, fz);

    force->compute(para, atom, measure);

    gpu_velocity_verlet_2<<<grid_size, BLOCK_SIZE>>>
    (N, fixed_group, label, time_step, mass, vx, vy, vz, fx, fy, fz);

    int N_fixed = (fixed_group == -1) ? 0 : atom->cpu_group_size[fixed_group];
    gpu_find_thermo<<<5, 1024>>>
    (
        N, N_fixed, fixed_group, label, temperature, box_length, 
        mass, z, potential_per_atom, vx, vy, vz, 
        virial_per_atom_x, virial_per_atom_y, virial_per_atom_z, thermo
    ); 
}




