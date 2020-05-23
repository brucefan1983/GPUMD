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
#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE_FORCE 128
#define RI_ALPHA     0.2
#define RI_ALPHA_SQ  0.04
#define RI_PI_FACTOR 0.225675833419103 // ALPHA * 2 / SQRT(PI)


RI::RI(FILE *fid)
{
    printf("Use the rigid-ion potential.\n");
    double x[4][3];
    for (int n = 0; n < 4; n++)
    {
        int count = fscanf(fid, "%lf%lf%lf", &x[n][0], &x[n][1], &x[n][2]);
        PRINT_SCANF_ERROR(count, 3, "Reading error for rigid-ion potential.");
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
    ri_para.b11 = 1.0 / ri_para.b11;
    ri_para.b22 = 1.0 / ri_para.b22;
    ri_para.b12 = 1.0 / ri_para.b12;

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
(int type1, int type2, RI_Para ri, double d12sq, double &p2, double &f2)
{
    double a, b, c, qq;
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

    double d12         = sqrt(d12sq);
    double d12inv      = 1.0 / d12;
    double d12inv3     = d12inv * d12inv * d12inv;
    double exponential = exp(-d12 * b);     // b = 1/rho
    double erfc_r = erfc(RI_ALPHA * d12) * d12inv;
    p2 = a * exponential - c * d12inv3 * d12inv3;
    p2 += qq * ( erfc_r - ri.v_rc - ri.dv_rc * (d12 - ri.cutoff) );
    f2 = 6.0*c*(d12inv3*d12inv3*d12inv) - a*exponential*b;
    f2-=qq*(erfc_r*d12inv+RI_PI_FACTOR*d12inv*exp(-RI_ALPHA_SQ*d12sq)+ri.dv_rc);
    f2 *= d12inv;
}


// force evaluation kernel
static __global__ void gpu_find_force
(
    RI_Para ri,
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

            // RI
            if (d12sq >= ri.cutoff * ri.cutoff) {continue;}
            find_p2_and_f2(type1, type2, ri, d12sq, p2, f2);

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
void RI::compute(Atom *atom, int potential_number)
{
    int shift = atom->shift[potential_number];
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

    gpu_find_force<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        ri_para,
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
        atom->force_per_atom.data(),
        atom->force_per_atom.data() + atom->N,
        atom->force_per_atom.data() + 2 * atom->N,
        atom->virial_per_atom.data(),
        atom->potential_per_atom.data()
    );
    CUDA_CHECK_KERNEL
}


