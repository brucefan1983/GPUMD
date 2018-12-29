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




#include "ldg.cuh"




/*------------------------------------------------------------------------------
    This file will be directly included in some other files.
    It contains a CUDA kernel function that is universal to 
    short-range many-body potentials.
------------------------------------------------------------------------------*/




static __global__ void find_force_many_body
(
    int calculate_hac, int calculate_shc, int calculate_hnemd,
    real fe_x, real fe_y, real fe_z,
    int number_of_particles, int N1, int N2, int pbc_x, int pbc_y, int pbc_z,
    int *g_neighbor_number, int *g_neighbor_list,
#ifdef USE_LDG
    const real* __restrict__ g_f12x,
    const real* __restrict__ g_f12y,
    const real* __restrict__ g_f12z,
    const real* __restrict__ g_x,
    const real* __restrict__ g_y,
    const real* __restrict__ g_z,
    const real* __restrict__ g_vx,
    const real* __restrict__ g_vy,
    const real* __restrict__ g_vz,
    const real* __restrict__ g_box_length,
#else
    real* g_f12x, real* g_f12y, real* g_f12z, real* g_x, real* g_y, real* g_z,
    real* g_vx, real* g_vy, real* g_vz, real* g_box_length,
#endif
    real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz,
    real *g_h, int *g_label, int *g_fv_index, real *g_fv,
    int *g_a_map, int *g_b_map, int *g_count_b
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
        real lx = LDG(g_box_length, 0);
        real ly = LDG(g_box_length, 1);
        real lz = LDG(g_box_length, 2);

        real vx1, vy1, vz1;
        if (calculate_hac || calculate_shc || calculate_hnemd)
        {
            vx1 = LDG(g_vx, n1);
            vy1 = LDG(g_vy, n1); 
            vz1 = LDG(g_vz, n1);
        }

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int neighbor_number_2 = g_neighbor_number[n2];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);

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
            if (calculate_hac || calculate_hnemd)
            {
                s_h1 += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s_h2 += (f21z * vz1) * x12;               // x-out
                s_h3 += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s_h4 += (f21z * vz1) * y12;               // y-out
                s_h5 += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            }

            // accumulate heat across some sections (for NEMD)
            // check if AB pair possible & exists
            if (calculate_shc && g_a_map[n1] != -1 && g_b_map[n2] != -1 &&
                g_fv_index[g_a_map[n1] * *(g_count_b) + g_b_map[n2]] != -1)
            {
                int index_12 =
                    g_fv_index[g_a_map[n1] * *(g_count_b) + g_b_map[n2]] * 12;
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

        if (calculate_hac || calculate_hnemd) // save heat current
        {
            g_h[n1 + 0 * number_of_particles] += s_h1;
            g_h[n1 + 1 * number_of_particles] += s_h2;
            g_h[n1 + 2 * number_of_particles] += s_h3;
            g_h[n1 + 3 * number_of_particles] += s_h4;
            g_h[n1 + 4 * number_of_particles] += s_h5;
        }
    }
}




