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
The class dealing with the Lennard-Jones (LJ) pairwise potentials.
------------------------------------------------------------------------------*/

#include "lj.cuh"
#include "mic.cuh"
#include "atom.cuh"
#include "error.cuh"

// best block size here: 128
#define BLOCK_SIZE_FORCE 128


LJ::LJ
(
    FILE *fid, int num_types,
    const std::vector<int> participating_kinds, int type_range
)
{
    printf("Use %d-element LJ potential.\n", num_types);
    if (!(num_types >= 1 && num_types <= MAX_TYPE))
    {
        PRINT_INPUT_ERROR("Incorrect number of LJ parameters.\n");
    }

    double epsilon, sigma, cutoff;
    rc = 0.0;
    for (int n = 0; n < type_range; n++)
    {
        for (int m = 0; m < type_range; m++)
        {
            if (pair_participating(n,m,participating_kinds))
            {
                int count = fscanf(fid, "%lf%lf%lf", &epsilon, &sigma, &cutoff);
                PRINT_SCANF_ERROR(count, 3, "Reading error for LJ potential.");

                lj_para.s6e4[n][m]   = pow(sigma, 6.0)  * epsilon * 4.0;
                lj_para.s12e4[n][m]  = pow(sigma, 12.0) * epsilon * 4.0;
                lj_para.cutoff_square[n][m] = cutoff * cutoff;
                if (rc < cutoff) rc = cutoff;
            }
            else // pair not participating, but must still be defined
            {
                lj_para.s6e4[n][m]   = 1.0;
                lj_para.s12e4[n][m]  = 1.0;
                lj_para.cutoff_square[n][m] = 0.0;
            }
        }
    }
}

LJ::~LJ(void)
{
    // nothing
}

bool LJ::pair_participating
(
    int n, int m, const std::vector<int> participating_kinds
)
{
    bool m_part = false;
    bool n_part = false;
    for (int i = 0; i < (int)participating_kinds.size(); i++)
    {
        if (participating_kinds[i] == m) m_part = true;
        if (participating_kinds[i] == n) n_part = true;
        if (m_part && n_part) return true; // early exit
    }
    return false;
}

// get U_ij and (d U_ij / d r_ij) / r_ij (the LJ potential)
static __device__ void find_p2_and_f2
(double s6e4, double s12e4, double d12sq, double &p2, double &f2)
{
    double d12inv2 = 1.0 / d12sq;
    double d12inv6 = d12inv2 * d12inv2 * d12inv2;
    f2 = 6.0 * (s6e4 * d12inv6 - s12e4 * 2.0 * d12inv6 * d12inv6) * d12inv2;
    p2 = s12e4 * d12inv6 * d12inv6 - s6e4 * d12inv6;
}

// force evaluation kernel
static __global__ void gpu_find_force
(
    LJ_Para lj,
    int number_of_particles, int N1, int N2, Box box,
    int *g_neighbor_number, int *g_neighbor_list, int *g_type, int shift,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    const double* __restrict__ g_vx,
    const double* __restrict__ g_vy,
    const double* __restrict__ g_vz,
    double *g_fx, double *g_fy, double *g_fz,
    double *g_virial, double *g_potential
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    double s_fx = 0.0; // force_x
    double s_fy = 0.0; // force_y
    double s_fz = 0.0; // force_z
    double s_pe = 0.0; // potential energy
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
        int type1 = g_type[n1] - shift;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int type2 = g_type[n2] - shift;

            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12sq = x12 * x12 + y12 * y12 + z12 * z12;

            double p2, f2;
            if (d12sq >= lj.cutoff_square[type1][type2]) {continue;}
            find_p2_and_f2
            (lj.s6e4[type1][type2], lj.s12e4[type1][type2], d12sq, p2, f2);

            // treat two-body potential in the same way as many-body potential
            double f12x = f2 * x12 * 0.5;
            double f12y = f2 * y12 * 0.5;
            double f12z = f2 * z12 * 0.5;
            double f21x = -f12x;
            double f21y = -f12y;
            double f21z = -f12z;

            // accumulate force
            s_fx += f12x - f21x;
            s_fy += f12y - f21y;
            s_fz += f12z - f21z;

            // accumulate potential energy and virial
            s_pe += p2 * 0.5; // two-body potential
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

        // save potential
        g_potential[n1] += s_pe;
    }
}


// Find force and related quantities for pair potentials (A wrapper)
void LJ::compute(Atom *atom, int potential_number)
{
    int shift = atom->shift[potential_number];
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

    gpu_find_force<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        lj_para,
        atom->N,
        N1,
        N2,
        atom->box,
        atom->neighbor.NN_local.data(),
        atom->neighbor.NL_local.data(),
        atom->type.data(),
        shift,
        atom->x.data(),
        atom->y.data(),
        atom->z.data(),
        atom->vx.data(),
        atom->vy.data(),
        atom->vz.data(),
        atom->fx.data(),
        atom->fy.data(),
        atom->fz.data(),
        atom->virial_per_atom.data(),
        atom->potential_per_atom.data()
    );
    CUDA_CHECK_KERNEL
}


