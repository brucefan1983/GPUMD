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
The modified Tersoff potentials as described in
    [1] T. Kumagai et al., Comput. Mater. Sci. 39, 457 (2007);
    [2] G. P. Purja Pun and Y. Mishin, Phys. Rev. B 95, 224103 (2017).
------------------------------------------------------------------------------*/


#include "tersoff_modc.cuh"
#include "mic.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE_FORCE 64 // 128 is also good
#define ONE_OVER_16      0.0625
#define NINE_OVER_16     0.5625
//Easy labels for indexing
#define A          0
#define B          1
#define LAMBDA     2
#define MU         3
#define ETA        4
#define DELTA      5
#define ALPHA      6
#define BETA       7
#define C0         8
#define C1         9
#define C2         10
#define C3         11
#define C4         12
#define C5         13
#define H          14
#define R1         15
#define R2         16
#define PI_FACTOR1 17
#define PI_FACTOR3 18
#define NUM_PARAMS 19


Tersoff_modc::Tersoff_modc(FILE *fid, Atom* atom, int num_of_types)
{
    num_types = num_of_types;
    printf("Use Tersoff-modc (%d-element) potential.\n", num_types);
    int n_entries = num_types * num_types * num_types;
    std::vector<double> cpu_ters(n_entries * NUM_PARAMS);

    rc = 0;
    int count;
    double a, b, lambda, mu, eta, delta, alpha, beta;
    double c0, c1, c2, c3, c4, c5, h, r1, r2;
    for (int i = 0; i < n_entries; i++)
    {
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
            &a, &b, &lambda, &mu, &eta, &delta, &alpha, &beta, 
            &c0, &c1, &c2, &c3, &c4, &c5, &h, &r1, &r2
        );
        if (count!=17) {printf("Error: reading error for Tersoff-modc.\n"); exit(1);}

        cpu_ters[i*NUM_PARAMS + A] = a;
        cpu_ters[i*NUM_PARAMS + B] = b;
        cpu_ters[i*NUM_PARAMS + LAMBDA] = lambda;
        cpu_ters[i*NUM_PARAMS + MU] = mu;
        cpu_ters[i*NUM_PARAMS + ETA] = eta;
        cpu_ters[i*NUM_PARAMS + DELTA] = delta;
        cpu_ters[i*NUM_PARAMS + ALPHA] = alpha;
        cpu_ters[i*NUM_PARAMS + BETA] = beta;
        cpu_ters[i*NUM_PARAMS + C0] = c0;
        cpu_ters[i*NUM_PARAMS + C1] = c1;
        cpu_ters[i*NUM_PARAMS + C2] = c2;
        cpu_ters[i*NUM_PARAMS + C3] = c3;
        cpu_ters[i*NUM_PARAMS + C4] = c4;
        cpu_ters[i*NUM_PARAMS + C5] = c5;
        cpu_ters[i*NUM_PARAMS + H] = h;
        cpu_ters[i*NUM_PARAMS + R1] = r1;
        cpu_ters[i*NUM_PARAMS + R2] = r2;
        cpu_ters[i*NUM_PARAMS + PI_FACTOR1] = PI / (r2 - r1);
        cpu_ters[i*NUM_PARAMS + PI_FACTOR3] = 3.0 * PI / (r2 - r1);
        rc = r2 > rc ? r2 : rc;
    }

    int num_of_neighbors = (atom->neighbor.MN < 50) ? atom->neighbor.MN : 50;
    tersoff_data.b.resize(atom->N * num_of_neighbors);
    tersoff_data.bp.resize(atom->N * num_of_neighbors);
    tersoff_data.f12x.resize(atom->N * num_of_neighbors);
    tersoff_data.f12y.resize(atom->N * num_of_neighbors);
    tersoff_data.f12z.resize(atom->N * num_of_neighbors);
    ters.resize(n_entries * NUM_PARAMS);
    ters.copy_from_host(cpu_ters.data());
}


Tersoff_modc::~Tersoff_modc(void)
{
    // nothing
}


static __device__ void find_fr_and_frp
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double &fr,
    double &frp
)
{
    fr  = ters[i + A] * exp(- ters[i + LAMBDA] * d12);
    frp = - ters[i + LAMBDA] * fr;
}


static __device__ void find_fa_and_fap
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double &fa,
    double &fap
)
{
    fa  = ters[i + B] * exp(- ters[i + MU] * d12);
    fap = - ters[i + MU] * fa;
}


static __device__ void find_fa
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double &fa
)
{
    fa  = ters[i + B] * exp(- ters[i + MU] * d12);
}


static __device__ void find_fc_and_fcp
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double &fc,
    double &fcp
)
{
    if (d12 < ters[i + R1]) {fc = 1.0; fcp = 0.0;}
    else if (d12 < ters[i + R2])
    {
        double tmp = d12 - ters[i + R1];
        double pi_factor1 = ters[i + PI_FACTOR1];
        double pi_factor3 = ters[i + PI_FACTOR3];

        fc = NINE_OVER_16 * cos(pi_factor1 * tmp)
           - ONE_OVER_16  * cos(pi_factor3 * tmp)
           + 0.5;

        fcp = sin(pi_factor3 * tmp) * pi_factor3 * ONE_OVER_16
            - sin(pi_factor1 * tmp) * pi_factor1 * NINE_OVER_16;
    }
    else {fc  = 0.0; fcp = 0.0;}
}


static __device__ void find_fc
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double &fc
)
{
    if (d12 < ters[i + R1]) {fc  = 1.0;}
    else if (d12 < ters[i + R2])
    {
        double tmp = d12 - ters[i + R1];
        fc = NINE_OVER_16 * cos(ters[i + PI_FACTOR1] * tmp)
           - ONE_OVER_16  * cos(ters[i + PI_FACTOR3] * tmp)
           + 0.5;
    }
    else {fc  = 0.0;}
}


static __device__ void find_g_and_gp
(
    int i,
    const double* __restrict__ ters,
    double cos,
    double &g,
    double &gp
)
{
    double x = (cos - ters[i + H]) * (cos - ters[i + H]);
    double exp_factor = exp(-ters[i + C5] * x);
    double c2c3_factor = ters[i + C2] * x / (ters[i + C3] + x);
    g  = (1.0 + ters[i + C4] * exp_factor) * c2c3_factor
       + ters[i + C1];
    gp = ters[i + C2] * ters[i + C3]
       / ( (ters[i + C3] + x) * (ters[i + C3] + x) )
       * (1.0 + ters[i + C4] * exp_factor)
       - ters[i + C4] * ters[i + C5] * c2c3_factor
       * exp_factor;
    gp *= 2.0 * (cos - ters[i + H]);
}


static __device__ void find_g
(
    int i,
    const double* __restrict__ ters,
    double cos,
    double &g
)
{
    double x = (cos - ters[i + H]) * (cos - ters[i + H]);
    g  = (1.0 + ters[i + C4] * exp(-ters[i + C5] * x))
       * ters[i + C2] * x / (ters[i + C3] + x)
       + ters[i + C1];
}


static __device__ void find_e_and_ep
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double d13,
    double &e,
    double &ep
)
{
    double r = d12 - d13;
    if (ters[i + BETA] > 2.0) //if beta == 3
    {
        e = exp(ters[i + ALPHA] * r * r * r);
        ep = ters[i + ALPHA] * 3.0 * r * r * e;
    }
    else // beta = 1
    {
        e = exp(ters[i + ALPHA] * r);
        ep = ters[i + ALPHA] * e;
    }
}

static __device__ void find_e
(
    int i,
    const double* __restrict__ ters,
    double d12,
    double d13,
    double &e
)
{
    double r = d12 - d13;
    if (ters[i + BETA] > 2.0) { e = exp(ters[i + ALPHA] * r * r * r);}
    else {e = exp(ters[i + ALPHA] * r);}
}


// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1
(
    int number_of_particles, int N1, int N2, Box box,
    int num_types, int* g_neighbor_number, int* g_neighbor_list,
    int* g_type, int shift,
    const double* __restrict__ ters,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    double* g_b, double* g_bp
)
{
    int num_types2 = num_types * num_types;
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    // to the (N2-1)-th atom
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
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double zeta = 0.0;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3] - shift;
                double x13 = g_x[n3] - x1;
                double y13 = g_y[n3] - y1;
                double z13 = g_z[n3] - z1;
                dev_apply_mic(box, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                double fc_ijk_13, g_ijk, e_ijk_12_13;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                if (d13 > ters[ijk*NUM_PARAMS + R2]) {continue;}
                find_fc(ijk*NUM_PARAMS, ters, d13, fc_ijk_13);
                find_g(ijk*NUM_PARAMS, ters, cos123, g_ijk);
                find_e(ijk*NUM_PARAMS, ters, d12, d13, e_ijk_12_13);
                zeta += fc_ijk_13 * g_ijk * e_ijk_12_13;
            }
            double zn, b_ijj;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            zn = pow(zeta, ters[ijj*NUM_PARAMS + ETA]);
            b_ijj = pow(1.0 + zn, -ters[ijj*NUM_PARAMS + DELTA]);
            if (zeta < 1.0e-16) // avoid division by 0
            {
                g_b[i1 * number_of_particles + n1]  = 1.0;
                g_bp[i1 * number_of_particles + n1] = 0.0;
            }
            else
            {
                g_b[i1 * number_of_particles + n1]  = b_ijj;
                g_bp[i1 * number_of_particles + n1]
                    = - b_ijj * zn * ters[ijj*NUM_PARAMS + ETA]
                    * ters[ijj*NUM_PARAMS + DELTA] / ((1.0 + zn) * zeta);
            }
        }
    }
}


// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_tersoff_step2
(
    int number_of_particles, int N1, int N2, Box box,
    int num_types, int *g_neighbor_number, int *g_neighbor_list,
    int *g_type, int shift,
    const double* __restrict__ ters,
    const double* __restrict__ g_b,
    const double* __restrict__ g_bp,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    double *g_potential, double *g_f12x, double *g_f12y, double *g_f12z
)
{
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    int num_types2 = num_types * num_types;
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1] - shift;
        double x1 = g_x[n1];
        double y1 = g_y[n1];
        double z1 = g_z[n1];
        double pot_energy = 0.0;
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2] - shift;

            double x12  = g_x[n2] - x1;
            double y12  = g_y[n2] - y1;
            double z12  = g_z[n2] - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = 1.0 / d12;
            double fc_ijj_12, fcp_ijj_12;
            double fa_ijj_12, fap_ijj_12, fr_ijj_12, frp_ijj_12;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            find_fc_and_fcp(ijj*NUM_PARAMS, ters, d12, fc_ijj_12, fcp_ijj_12);
            find_fa_and_fap(ijj*NUM_PARAMS, ters, d12, fa_ijj_12, fap_ijj_12);
            find_fr_and_frp(ijj*NUM_PARAMS, ters, d12, fr_ijj_12, frp_ijj_12);

            // (i,j) part
            double b12 = g_b[index];
            double factor3 = d12inv *
            (
                fcp_ijj_12 * 
                (fr_ijj_12 - b12 * fa_ijj_12 + ters[ijj*NUM_PARAMS + C0])
                + fc_ijj_12 * (frp_ijj_12 - b12 * fap_ijj_12)
            );
            double f12x = x12 * factor3 * 0.5;
            double f12y = y12 * factor3 * 0.5;
            double f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            pot_energy += fc_ijj_12 * 0.5 *
            (
                fr_ijj_12 - b12 * fa_ijj_12 + ters[ijj*NUM_PARAMS + C0]
            );

            // (i,j,k) part
            double bp12 = g_bp[index];
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3] - shift;
                double x13 = g_x[n3] - x1;
                double y13 = g_y[n3] - y1;
                double z13 = g_z[n3] - z1;
                dev_apply_mic(box, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double fc_ikk_13, fc_ijk_13, fa_ikk_13, fc_ikj_12, fcp_ikj_12;
                int ikj = type1 * num_types2 + type3 * num_types + type2;
                int ikk = type1 * num_types2 + type3 * num_types + type3;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                find_fc(ikk*NUM_PARAMS, ters, d13, fc_ikk_13);
                find_fc(ijk*NUM_PARAMS, ters, d13, fc_ijk_13);
                find_fa(ikk*NUM_PARAMS, ters, d13, fa_ikk_13);
                find_fc_and_fcp(ikj*NUM_PARAMS, ters, d12,
                                	fc_ikj_12, fcp_ikj_12);
                double bp13 = g_bp[index_2];
                double one_over_d12d13 = 1.0 / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double g_ijk, gp_ijk;
                find_g_and_gp(ijk*NUM_PARAMS, ters, cos123, g_ijk, gp_ijk);

                double g_ikj, gp_ikj;
                find_g_and_gp(ikj*NUM_PARAMS, ters, cos123, g_ikj, gp_ikj);

                // exp with d12 - d13
                double e_ijk_12_13, ep_ijk_12_13;
                find_e_and_ep(ijk*NUM_PARAMS, ters, d12, d13,
                                	e_ijk_12_13, ep_ijk_12_13);

                // exp with d13 - d12
                double e_ikj_13_12, ep_ikj_13_12;
                find_e_and_ep(ikj*NUM_PARAMS, ters, d13, d12,
                                	e_ikj_13_12, ep_ikj_13_12);

                // derivatives with cosine
                double dc=-fc_ijj_12*bp12*fa_ijj_12*fc_ijk_13*gp_ijk*e_ijk_12_13+
                        -fc_ikj_12*bp13*fa_ikk_13*fc_ikk_13*gp_ikj*e_ikj_13_12;
                // derivatives with rij
                double dr=(-fc_ijj_12*bp12*fa_ijj_12*fc_ijk_13*g_ijk*ep_ijk_12_13 +
                  (-fcp_ikj_12*bp13*fa_ikk_13*g_ikj*e_ikj_13_12 +
                  fc_ikj_12*bp13*fa_ikk_13*g_ikj*ep_ikj_13_12)*fc_ikk_13)*d12inv;
                double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * dr + dc * cos_d)*0.5;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * dr + dc * cos_d)*0.5;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * dr + dc * cos_d)*0.5;
            }
            g_f12x[index] = f12x; g_f12y[index] = f12y; g_f12z[index] = f12z;
        }
        // save potential
        g_potential[n1] += pot_energy;
    }
}


// Wrapper of force evaluation for the Tersoff potential
void Tersoff_modc::compute(Atom *atom, int potential_number)
{
    int N = atom->N;
    int shift = atom->shift[potential_number];
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    int *NN = atom->neighbor.NN_local.data();
    int *NL = atom->neighbor.NL_local.data();
    int *type = atom->type.data();
    double *x = atom->x.data();
    double *y = atom->y.data();
    double *z = atom->z.data();
    double *pe = atom->potential_per_atom.data();

    // special data for Tersoff potential
    double *f12x = tersoff_data.f12x.data();
    double *f12y = tersoff_data.f12y.data();
    double *f12z = tersoff_data.f12z.data();
    double *b    = tersoff_data.b.data();
    double *bp   = tersoff_data.bp.data();

    // pre-compute the bond order functions and their derivatives
    find_force_tersoff_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, num_types,
        NN, NL, type, shift, ters.data(), x, y, z, b, bp
    );
    CUDA_CHECK_KERNEL

    // pre-compute the partial forces
    find_force_tersoff_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, num_types,
        NN, NL, type, shift, ters.data(), b, bp, x, y, z, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL

    // the final step: calculate force and related quantities
    find_properties_many_body(atom, NN, NL, f12x, f12y, f12z);
}
