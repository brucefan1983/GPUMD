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
#include "model/mic.cuh"
#include "utilities/error.cuh"
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
    const int number_of_particles,
    const int N1,
    const int N2,
    const Box box,
    const int *g_neighbor_number,
    const int *g_neighbor_list,
    const double* __restrict__ g_f12x,
    const double* __restrict__ g_f12y,
    const double* __restrict__ g_f12z,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    double *g_fx, double *g_fy, double *g_fz,
    double *g_virial
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    double s_fx = 0.0; // force_x
    double s_fy = 0.0; // force_y
    double s_fz = 0.0; // force_z
    double s_sxx = 0.0; // virial_stress_xx
    double s_sxy = 0.0; // virial_stress_xy
    double s_sxz = 0.0; // virial_stress_xz
    double s_syx = 0.0; // virial_stress_yx
    double s_syy = 0.0; // virial_stress_yy
    double s_syz = 0.0; // virial_stress_yz
    double s_szx = 0.0; // virial_stress_zx
    double s_szy = 0.0; // virial_stress_zy
    double s_szz = 0.0; // virial_stress_zz

    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int neighbor_number_2 = g_neighbor_number[n2];

            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);

            double f12x = g_f12x[index];
            double f12y = g_f12y[index];
            double f12z = g_f12z[index];
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_neighbor_list[n2 + number_of_particles * k])
                { offset = k; break; }
            }
            index = offset * number_of_particles + n2;
            double f21x = g_f12x[index];
            double f21y = g_f12y[index];
            double f21z = g_f12z[index];

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
    const Box& box,
    const int* NN,
    const int* NL,
    const double* f12x,
    const double* f12y,
    const double* f12z,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom
)
{
    const int number_of_atoms = position_per_atom.size() / 3;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

    gpu_find_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        number_of_atoms, N1, N2, box, NN,
        NL, f12x, f12y, f12z,
        position_per_atom.data(),
        position_per_atom.data() + number_of_atoms,
        position_per_atom.data() + number_of_atoms * 2,
        force_per_atom.data(),
        force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms,
        virial_per_atom.data()
    );
    CUDA_CHECK_KERNEL
}


