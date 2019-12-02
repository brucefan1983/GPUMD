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
The abstract base class (ABC) for the potential classes.
------------------------------------------------------------------------------*/


#include "potential.cuh"
#include "mic.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE_FORCE 64


Potential::Potential(void)
{
    rc = 0.0;
}


Potential::~Potential(void)
{
    // nothing
}


static __global__ void gpu_find_force_many_body
(
    int number_of_particles, int N1, int N2,
    Box box,
    int *g_neighbor_number, int *g_neighbor_list,
    const real* __restrict__ g_f12x,
    const real* __restrict__ g_f12y,
    const real* __restrict__ g_f12z,
    const real* __restrict__ g_x,
    const real* __restrict__ g_y,
    const real* __restrict__ g_z,
    const real* __restrict__ g_vx,
    const real* __restrict__ g_vy,
    const real* __restrict__ g_vz,
    real *g_fx, real *g_fy, real *g_fz,
    real *g_virial
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    real s_fx = ZERO; // force_x
    real s_fy = ZERO; // force_y
    real s_fz = ZERO; // force_z
    real s_sxx = ZERO; // virial_stress_xx
    real s_sxy = ZERO; // virial_stress_xy
    real s_sxz = ZERO; // virial_stress_xz
    real s_syx = ZERO; // virial_stress_yx
    real s_syy = ZERO; // virial_stress_yy
    real s_syz = ZERO; // virial_stress_yz
    real s_szx = ZERO; // virial_stress_zx
    real s_szy = ZERO; // virial_stress_zy
    real s_szz = ZERO; // virial_stress_zz

    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        real x1 = LDG(g_x, n1); real y1 = LDG(g_y, n1); real z1 = LDG(g_z, n1);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int neighbor_number_2 = g_neighbor_number[n2];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);

            real f12x = LDG(g_f12x, index);
            real f12y = LDG(g_f12y, index);
            real f12z = LDG(g_f12z, index);
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_neighbor_list[n2 + number_of_particles * k])
                { offset = k; break; }
            }
            index = offset * number_of_particles + n2;
            real f21x = LDG(g_f12x, index);
            real f21y = LDG(g_f12y, index);
            real f21z = LDG(g_f12z, index);

            // per atom force
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z; 

            // per-atom virial
            s_sxx += x12 * f21x;
            s_sxy += x12 * f21y;
            s_sxz += x12 * f21z;
            s_syx += y12 * f21x;
            s_syy += y12 * f21y;
            s_syz += y12 * f21z;
            s_szx += z12 * f21x;
            s_szy += z12 * f21y;
            s_szz += z12 * f21z;
        }

        // save force
        g_fx[n1] += s_fx;
        g_fy[n1] += s_fy;
        g_fz[n1] += s_fz;

        // save virial
        // xx xy xz    0 3 4
        // yx yy yz    6 1 5
        // zx zy zz    7 8 2
        g_virial[n1 + 0 * number_of_particles] += s_sxx;
        g_virial[n1 + 1 * number_of_particles] += s_syy;
        g_virial[n1 + 2 * number_of_particles] += s_szz;
        g_virial[n1 + 3 * number_of_particles] += s_sxy;
        g_virial[n1 + 4 * number_of_particles] += s_sxz;
        g_virial[n1 + 5 * number_of_particles] += s_syz;
        g_virial[n1 + 6 * number_of_particles] += s_syx;
        g_virial[n1 + 7 * number_of_particles] += s_szx;
        g_virial[n1 + 8 * number_of_particles] += s_szy;
    }
}


// Wrapper of the above kernel
// used in tersoff.cu, sw.cu, rebo_mos2.cu and vashishta.cu
void Potential::find_properties_many_body
(
    Atom *atom, Measure *measure, int* NN, int* NL,
    real* f12x, real* f12y, real* f12z
)
{
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    gpu_find_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        atom->N, N1, N2, atom->box, NN,
        NL, f12x, f12y, f12z, atom->x, atom->y, atom->z, atom->vx,
        atom->vy, atom->vz, atom->fx, atom->fy, atom->fz,
        atom->virial_per_atom
    );
    CUDA_CHECK_KERNEL
}


