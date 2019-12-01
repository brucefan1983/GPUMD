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
The class dealing with the rigid-ion potential.

Reference for the method of evaluating the Coulomb force in the rigid-ion
potential:

[1] C. J. Fennell and J. D. Gezelter. Is the Ewald summation still necessary?
Pairwise alternatives to the accepted standard for long-range electrostatics,
J. Chem. Phys. 124, 234104 (2006).
------------------------------------------------------------------------------*/

#include "ri.cuh"
#include "mic.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE_FORCE 128
#define RI_ALPHA     0.2
#define RI_ALPHA_SQ  0.04
#define RI_PI_FACTOR 0.225675833419103 // ALPHA * 2 / SQRT(PI)
#define GPU_FIND_FORCE(A)                                                      \
    gpu_find_force<A>                                                          \
    <<<grid_size, BLOCK_SIZE_FORCE>>>                                          \
    (                                                                          \
        measure->hnemd.fe_x, measure->hnemd.fe_y, measure->hnemd.fe_z,         \
        ri_para, atom->N, N1, N2, atom->box,                                   \
        atom->NN_local, atom->NL_local,                                        \
        atom->type, shift, atom->x, atom->y, atom->z,                          \
        atom->vx, atom->vy, atom->vz,                                          \
        atom->fx, atom->fy, atom->fz, atom->virial_per_atom,                   \
        atom->potential_per_atom                                               \
    )


RI::RI(FILE *fid)
{
    printf("Use the rigid-ion potential.\n");
    double x[4][3];
    for (int n = 0; n < 4; n++)
    {
        int count = fscanf(fid, "%lf%lf%lf", &x[n][0], &x[n][1], &x[n][2]);
        if (count != 3)
        {print_error("reading error for potential.in.\n"); exit(1);}
    }
    ri_para.cutoff = x[0][2];
    ri_para.qq11   = x[0][0] * x[0][0] * K_C;
    ri_para.qq22   = x[0][1] * x[0][1] * K_C;
    ri_para.qq12   = x[0][0] * x[0][1] * K_C;
    ri_para.a11    = x[1][0];
    ri_para.b11    = x[1][1];
    ri_para.c11    = x[1][2];
    ri_para.a22    = x[2][0];
    ri_para.b22    = x[2][1];
    ri_para.c22    = x[2][2];
    ri_para.a12    = x[3][0];
    ri_para.b12    = x[3][1];
    ri_para.c12    = x[3][2];
    ri_para.b11 = ONE / ri_para.b11;
    ri_para.b22 = ONE / ri_para.b22;
    ri_para.b12 = ONE / ri_para.b12;

    rc = ri_para.cutoff; // force cutoff

    ri_para.v_rc = erfc(RI_ALPHA * rc) / rc;
    ri_para.dv_rc = -erfc(RI_ALPHA * rc) / (rc * rc);
    ri_para.dv_rc -= RI_PI_FACTOR * exp(-RI_ALPHA_SQ * rc * rc) / rc;
}

RI::~RI(void)
{
    // nothing
}

// get U_ij and (d U_ij / d r_ij) / r_ij (the RI potential)
static __device__ void find_p2_and_f2
(int type1, int type2, RI_Para ri, real d12sq, real &p2, real &f2)
{
    real a, b, c, qq;
    if (type1 == 0 && type2 == 0)
    {
        a  = ri.a11;
        b  = ri.b11;
        c  = ri.c11;
        qq = ri.qq11;
    }
    else if (type1 == 1 && type2 == 1)
    {
        a  = ri.a22;
        b  = ri.b22;
        c  = ri.c22;
        qq = ri.qq22;
    }
    else
    {
        a  = ri.a12;
        b  = ri.b12;
        c  = ri.c12;
        qq = ri.qq12;
    }

    real d12         = sqrt(d12sq);
    real d12inv      = ONE / d12;
    real d12inv3     = d12inv * d12inv * d12inv;
    real exponential = exp(-d12 * b);     // b = 1/rho
    real erfc_r = erfc(RI_ALPHA * d12) * d12inv;
    p2 = a * exponential - c * d12inv3 * d12inv3;
    p2 += qq * ( erfc_r - ri.v_rc - ri.dv_rc * (d12 - ri.cutoff) );
    f2 = SIX*c*(d12inv3*d12inv3*d12inv) - a*exponential*b;
    f2-=qq*(erfc_r*d12inv+RI_PI_FACTOR*d12inv*exp(-RI_ALPHA_SQ*d12sq)+ri.dv_rc);
    f2 *= d12inv;
}


// force evaluation kernel
template <int cal_k>
static __global__ void gpu_find_force
(
    real fe_x, real fe_y, real fe_z,
    RI_Para ri,
    int number_of_particles, int N1, int N2, Box box,
    int *g_neighbor_number, int *g_neighbor_list, int *g_type, int shift,
    const real* __restrict__ g_x,
    const real* __restrict__ g_y,
    const real* __restrict__ g_z,
    const real* __restrict__ g_vx,
    const real* __restrict__ g_vy,
    const real* __restrict__ g_vz,
    real *g_fx, real *g_fy, real *g_fz,
    real *g_virial, real *g_potential
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
    real s_fx = ZERO; // force_x
    real s_fy = ZERO; // force_y
    real s_fz = ZERO; // force_z
    real s_pe = ZERO; // potential energy
    real s_sxx = ZERO; // virial_stress_xx
    real s_sxy = ZERO; // virial_stress_xy
    real s_sxz = ZERO; // virial_stress_xz
    real s_syx = ZERO; // virial_stress_yx
    real s_syy = ZERO; // virial_stress_yy
    real s_syz = ZERO; // virial_stress_yz
    real s_szx = ZERO; // virial_stress_zx
    real s_szy = ZERO; // virial_stress_zy
    real s_szz = ZERO; // virial_stress_zz

    // driving force
    real fx_driving = ZERO;
    real fy_driving = ZERO;
    real fz_driving = ZERO;

    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1] - shift;
        real x1 = LDG(g_x, n1);
        real y1 = LDG(g_y, n1);
        real z1 = LDG(g_z, n1);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int type2 = g_type[n2] - shift;

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);
            real d12sq = x12 * x12 + y12 * y12 + z12 * z12;

            real p2, f2;

            // RI
            if (d12sq >= ri.cutoff * ri.cutoff) {continue;}
            find_p2_and_f2(type1, type2, ri, d12sq, p2, f2);

            // treat two-body potential in the same way as many-body potential
            real f12x = f2 * x12 * HALF;
            real f12y = f2 * y12 * HALF;
            real f12z = f2 * z12 * HALF;
            real f21x = -f12x;
            real f21y = -f12y;
            real f21z = -f12z;

            // accumulate force
            s_fx += f12x - f21x;
            s_fy += f12y - f21y;
            s_fz += f12z - f21z;

            // driving force
            if (cal_k)
            {
                fx_driving += f21x * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
                fy_driving += f21y * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
                fz_driving += f21z * (x12 * fe_x + y12 * fe_y + z12 * fe_z);
            }

            // accumulate potential energy and virial
            s_pe += p2 * HALF; // two-body potential
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

        // add driving force
        if (cal_k)
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
void RI::compute(Atom *atom, Measure *measure, int potential_number)
{
    find_measurement_flags(atom, measure);
    int shift = atom->shift[potential_number];
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

    if (compute_hnemd)
    {
        GPU_FIND_FORCE(1);
    }
    else
    {
        GPU_FIND_FORCE(0);
    }
    CUDA_CHECK_KERNEL
}


