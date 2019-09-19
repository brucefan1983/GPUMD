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
#define BLOCK_GK_RESET 128


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
    int calculate_shc, int calculate_hnemd, int calculate_gkma,
    real fe_x, real fe_y, real fe_z,
    int N, int N1, int N2,
    int triclinic, int pbc_x, int pbc_y, int pbc_z,
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
    const real* __restrict__ g_box,
    real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz,
    real *g_h, int *g_label, int *g_fv_index, real *g_fv,
    int *g_a_map, int *g_b_map, int g_count_b,
    const real* __restrict__ g_mass,
    const real* __restrict__ g_eig,
    const real* __restrict__ g_xdot,
    real* g_jmn,
    int num_modes
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    real s_fx = ZERO; // force_x
    real s_fy = ZERO; // force_y
    real s_fz = ZERO; // force_z
    real s_sx = ZERO; // virial_stress_x
    real s_sy = ZERO; // virial_stress_y
    real s_sz = ZERO; // virial_stress_z
    real s_h1 = ZERO; // heat_x_in
    real s_h2 = ZERO; // heat_x_out
    real s_h3 = ZERO; // heat_y_in
    real s_h4 = ZERO; // heat_y_out
    real s_h5 = ZERO; // heat_z

    // driving force in the HNEMD method
    real fx_driving = ZERO;
    real fy_driving = ZERO;
    real fz_driving = ZERO;

    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        real x1 = LDG(g_x, n1); real y1 = LDG(g_y, n1); real z1 = LDG(g_z, n1);

        real vx1, vy1, vz1;
        vx1 = LDG(g_vx, n1);
        vy1 = LDG(g_vy, n1);
        vz1 = LDG(g_vz, n1);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * N + n1;
            int n2 = g_neighbor_list[index];
            int neighbor_number_2 = g_neighbor_number[n2];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);

            real f12x = LDG(g_f12x, index);
            real f12y = LDG(g_f12y, index);
            real f12z = LDG(g_f12z, index);
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_neighbor_list[n2 + N * k])
                { offset = k; break; }
            }
            index = offset * N + n2;
            real f21x = LDG(g_f12x, index);
            real f21y = LDG(g_f12y, index);
            real f21z = LDG(g_f12z, index);

            // per atom force
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z; 

            // driving force
            if (calculate_hnemd)
            { 
                fx_driving += f21x * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
                fy_driving += f21y * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
                fz_driving += f21z * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
            }

            // per-atom virial
            s_sx -= x12 * (f12x - f21x) * HALF;
            s_sy -= y12 * (f12y - f21y) * HALF;
            s_sz -= z12 * (f12z - f21z) * HALF;

            // per-atom heat current
            s_h1 += (f21x * vx1 + f21y * vy1) * x12;  // x-in
            s_h2 += (f21z * vz1) * x12;               // x-out
            s_h3 += (f21x * vx1 + f21y * vy1) * y12;  // y-in
            s_h4 += (f21z * vz1) * y12;               // y-out
            s_h5 += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all

            if (calculate_gkma)
            {
                real vx, vy, vz;
                real rsqrtmass = rsqrt(LDG(g_mass, n1));
                for (int i = 0; i < num_modes; i++)
                {
                    vx=rsqrtmass*g_eig[n1 + i*3*N]*g_xdot[i];
                    vy=rsqrtmass*g_eig[n1 + (1 + i*3)*N]*g_xdot[i + N];
                    vz=rsqrtmass*g_eig[n1 + (2 + i*3)*N]*g_xdot[i + 2*N];

                    g_jmn[n1 + i*num_modes] +=
                            (f21x*vx + f21y*vy + f21z*vz)*x12; // x-all
                    g_jmn[n1 + (i+num_modes)*num_modes] +=
                            (f21x*vx + f21y*vy + f21z*vz)*y12; // y-all
                    g_jmn[n1 + (i+2*num_modes)*num_modes] +=
                            (f21x*vx + f21y*vy + f21z*vz)*z12; // z-all

                }
            }


            // accumulate heat across some sections (for NEMD)
            // check if AB pair possible & exists
            if (calculate_shc && g_a_map[n1] != -1 && g_b_map[n2] != -1 &&
                g_fv_index[g_a_map[n1] * g_count_b + g_b_map[n2]] != -1)
            {
                int index_12 =
                    g_fv_index[g_a_map[n1] * g_count_b + g_b_map[n2]] * 12;
                g_fv[index_12 + 0]  += f12x;
                g_fv[index_12 + 1]  += f12y;
                g_fv[index_12 + 2]  += f12z;
                g_fv[index_12 + 3]  += f21x;
                g_fv[index_12 + 4]  += f21y;
                g_fv[index_12 + 5]  += f21z;
                g_fv[index_12 + 6]  = vx1;
                g_fv[index_12 + 7]  = vy1;
                g_fv[index_12 + 8]  = vz1;
                g_fv[index_12 + 9]  = LDG(g_vx, n2);
                g_fv[index_12 + 10] = LDG(g_vy, n2);
                g_fv[index_12 + 11] = LDG(g_vz, n2);
            }
        }

        // add driving force
        if (calculate_hnemd)
        {
            s_fx += fx_driving;
            s_fy += fy_driving;
            s_fz += fz_driving;
        }

        // save force
        g_fx[n1] += s_fx;
        g_fy[n1] += s_fy;
        g_fz[n1] += s_fz;

        // save virial
        g_sx[n1] += s_sx;
        g_sy[n1] += s_sy;
        g_sz[n1] += s_sz;

        g_h[n1 + 0 * N] += s_h1;
        g_h[n1 + 1 * N] += s_h2;
        g_h[n1 + 2 * N] += s_h3;
        g_h[n1 + 3 * N] += s_h4;
        g_h[n1 + 4 * N] += s_h5;
    }
}

static __global__ void gpu_calc_xdot
(
        int N1, int N2, int num_modes,
        const real* __restrict__ g_vx,
        const real* __restrict__ g_vy,
        const real* __restrict__ g_vz,
        const real* __restrict__ g_mass,
        const real* __restrict__ g_eig,
        real* g_xdot
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {

        real vx1, vy1, vz1;
        vx1 = LDG(g_vx, n1);
        vy1 = LDG(g_vy, n1);
        vz1 = LDG(g_vz, n1);

        real sqrtmass = sqrt(LDG(g_mass, n1));
        for (int i = 0; i < num_modes; i++)
        {
            g_xdot[i] = sqrtmass*g_eig[n1 + i*3*num_modes]*vx1;
            g_xdot[i + num_modes] =
                    sqrtmass*g_eig[n1 + (1 + i*3)*num_modes]*vy1;
            g_xdot[i + 2*num_modes] =
                    sqrtmass*g_eig[n1 + (2 + i*3)*num_modes]*vz1;
        }
    }
}

static __global__ void gpu_reset_xdot
(
        int num_elements, real* xdot
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        xdot[n] = ZERO;
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
    find_measurement_flags(atom, measure);

    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    if (compute_gkma)
    {
        int num_elements = measure->gkma.num_modes*3;
        gpu_reset_xdot<<<(num_elements-1)/BLOCK_GK_RESET+1, BLOCK_GK_RESET>>>
        (
                num_elements, measure->gkma.xdot
        );
        CUDA_CHECK_KERNEL

        gpu_calc_xdot<<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N1, N2, measure->gkma.num_modes,
            atom->vx, atom->vy, atom->vz,
            atom->mass, measure->gkma.eig, measure->gkma.xdot
        );
        CUDA_CHECK_KERNEL
    }

    gpu_find_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        compute_shc, measure->hnemd.compute, measure->gkma.compute,
        measure->hnemd.fe_x, measure->hnemd.fe_y, measure->hnemd.fe_z,
        atom->N, N1, N2, atom->box.triclinic, 
        atom->box.pbc_x, atom->box.pbc_y, atom->box.pbc_z, NN,
        NL, f12x, f12y, f12z, atom->x, atom->y, atom->z, atom->vx,
        atom->vy, atom->vz, atom->box.h, atom->fx, atom->fy, atom->fz,
        atom->virial_per_atom_x, atom->virial_per_atom_y,
        atom->virial_per_atom_z, atom->heat_per_atom, atom->group[0].label,
        measure->shc.fv_index, measure->shc.fv, measure->shc.a_map,
        measure->shc.b_map, measure->shc.count_b, atom->mass,
        measure->gkma.eig, measure->gkma.xdot, measure->gkma.jmn,
        measure->gkma.num_modes
    );
    CUDA_CHECK_KERNEL
}


void Potential::find_measurement_flags(Atom* atom, Measure* measure)
{
    int compute_hac = 0;
    if (measure->hac.compute)
    {
        compute_hac = (atom->step + 1) % measure->hac.sample_interval == 0;
    }
    compute_j = 0;
    if (measure->compute.compute_jp)
    {
        compute_j = (atom->step + 1) % measure->compute.sample_interval == 0;
    }
    compute_j = (compute_j || compute_hac);
    compute_shc = 0;
    if (measure->shc.compute)
    {
        compute_shc = (atom->step + 1) % measure->shc.sample_interval == 0;
    }
    compute_gkma = 0;
    if (measure->gkma.compute)
    {
        compute_gkma = (atom->step + 1) % measure->gkma.sample_interval == 0;
    }
}


