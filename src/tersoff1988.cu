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
The version of the Tersoff potential as described in
    [1] J. Tersoff, New empirical approach for the structure and energy
    of covalent systems, PRB 37, 6991 (1988).
------------------------------------------------------------------------------*/


#include "tersoff1988.cuh"

#include "mic.cuh"
#include "atom.cuh"
#include "error.cuh"

#define LDG(a, n) __ldg(a + n)
#define BLOCK_SIZE_FORCE 64 // 128 is also good
#define EPSILON 1.0e-15

//Easy labels for indexing
#define A 0
#define B 1
#define LAMBDA 2
#define MU 3
#define BETA 4
#define EN 5 //special name for n to avoid conflict
#define C 6
#define D 7
#define H 8
#define R1 9
#define R2 10
#define M 11
#define ALPHA 12
#define GAMMA 13
#define C2 14
#define D2 15
#define ONE_PLUS_C2OVERD2 16
#define PI_FACTOR 17
#define MINUS_HALF_OVER_N 18

#define NUM_PARAMS 19


Tersoff1988::Tersoff1988(FILE *fid, Atom* atom, int num_of_types)
{
    num_types = num_of_types;
    printf("Use Tersoff-1988 (%d-element) potential.\n", num_types);
    int n_entries = num_types * num_types * num_types;
    // 14 parameters per entry of tersoff1988 + 5 pre-calculated values
    std::vector<double> cpu_ters(n_entries * NUM_PARAMS);

    char err[50] = "Error: Illegal Tersoff parameter.";
    rc = 0;
    int count;
    double a, b,lambda, mu, beta, n, c, d, h, r1, r2, m, alpha, gamma;
    for (int i = 0; i < n_entries; i++)
    {
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
            &a, &b, &lambda, &mu, &beta, &n, &c, &d, &h, &r1, &r2, &m, &alpha, &gamma
        );
        if (count!=14) {printf("Error: reading error for potential.in.\n");exit(1);}

        int m_int = round(m);
        // Parameter checking
        if (a < 0.0)
            {printf("%s A must be >= 0.\n",err); exit(1);}
        if (b < 0.0)
            {printf("%s B must be >= 0.\n",err); exit(1);}
        if (lambda < 0.0)
            {printf("%s Lambda must be >= 0.\n",err); exit(1);}
        if(mu < 0.0)
            {printf("%s Mu must be >= 0.\n",err); exit(1);}
        if(beta < 0.0)
            {printf("%s Beta must be >= 0.\n",err); exit(1);}
        if(n < 0.0)
            {printf("%s n must be >= 0.\n",err); exit(1);}
        if(c < 0.0)
            {printf("%s c must be >= 0.\n",err); exit(1);}
        if(d < 0.0)
            {printf("%s d must be >= 0.\n",err); exit(1);}
        if(r1 < 0.0)
            {printf("%s R must be >= 0.\n",err); exit(1);}
        if(r2 < 0.0)
            {printf("%s S must be >= 0.\n",err); exit(1);}
        if(r2 < r1)
            {printf("%s S-R must be >= 0.\n",err); exit(1);}
        if(m_int != 3 && m_int != 1)
            {printf("%s m must be 1 or 3.\n",err); exit(1);}
        if(gamma < 0.0)
            {printf("%s Gamma must be >= 0.\n",err); exit(1);}

        cpu_ters[i*NUM_PARAMS + A] = a;
        cpu_ters[i*NUM_PARAMS + B] = b;
        cpu_ters[i*NUM_PARAMS + LAMBDA] = lambda;
        cpu_ters[i*NUM_PARAMS + MU] = mu;
        cpu_ters[i*NUM_PARAMS + BETA] = beta;
        cpu_ters[i*NUM_PARAMS + EN] = n;
        cpu_ters[i*NUM_PARAMS + C] = c;
        cpu_ters[i*NUM_PARAMS + D] = d;
        cpu_ters[i*NUM_PARAMS + H] = h;
        cpu_ters[i*NUM_PARAMS + R1] = r1;
        cpu_ters[i*NUM_PARAMS + R2] = r2;
        cpu_ters[i*NUM_PARAMS + M] = m_int;
        if (alpha < EPSILON)
        {
            cpu_ters[i*NUM_PARAMS + ALPHA] = ZERO;
        }
        else
        {
            cpu_ters[i*NUM_PARAMS + ALPHA] = alpha;
        }
        cpu_ters[i*NUM_PARAMS + GAMMA] = gamma;
        cpu_ters[i*NUM_PARAMS + C2] = c * c;
        cpu_ters[i*NUM_PARAMS + D2] = d * d;
        cpu_ters[i*NUM_PARAMS + ONE_PLUS_C2OVERD2] = 1.0 +
        cpu_ters[i*NUM_PARAMS + C2] / cpu_ters[i*NUM_PARAMS + D2];
        cpu_ters[i*NUM_PARAMS + PI_FACTOR] = PI / (r2 - r1);
        cpu_ters[i*NUM_PARAMS + MINUS_HALF_OVER_N] = - 0.5 / n;
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


Tersoff1988::~Tersoff1988(void)
{
    // nothing
}


static __device__ void find_fr_and_frp
(int i, const double* __restrict__ ters, double d12, double &fr, double &frp)
{
    fr  = LDG(ters,i + A) * exp(- LDG(ters,i + LAMBDA) * d12);
    frp = - LDG(ters,i + LAMBDA) * fr;
}


static __device__ void find_fa_and_fap
(int i, const double* __restrict__ ters, double d12, double &fa, double &fap)
{
    fa  = LDG(ters, i + B) * exp(- LDG(ters, i + MU) * d12);
    fap = - LDG(ters, i + MU) * fa;
}


static __device__ void find_fa
(int i, const double* __restrict__ ters, double d12, double &fa)
{
    fa  = LDG(ters, i + B) * exp(- LDG(ters, i + MU) * d12);
}


static __device__ void find_fc_and_fcp
(int i, const double* __restrict__ ters, double d12, double &fc, double &fcp)
{
    if (d12 < LDG(ters, i + R1)){fc = ONE; fcp = ZERO;}
    else if (d12 < LDG(ters, i + R2))
    {
        fc  =  cos(LDG(ters, i + PI_FACTOR) *
                (d12 - LDG(ters, i + R1))) * HALF + HALF;
        fcp = -sin(LDG(ters, i + PI_FACTOR) *
                (d12 - LDG(ters, i + R1)))*LDG(ters, i + PI_FACTOR)*HALF;
    }
    else {fc  = ZERO; fcp = ZERO;}
}


static __device__ void find_fc
(int i, const double* __restrict__ ters, double d12, double &fc)
{
    if (d12 < LDG(ters, i + R1)) {fc  = ONE;}
    else if (d12 < LDG(ters, i + R2))
    {
        fc = cos(LDG(ters, i + PI_FACTOR) *
                (d12 - LDG(ters, i + R1))) * HALF + HALF;
    }
    else {fc  = ZERO;}
}


static __device__ void find_g_and_gp
(int i, const double* __restrict__ ters, double cos, double &g, double &gp)
{
    double temp = LDG(ters, i + D2) + (cos - LDG(ters, i + H)) *
                (cos - LDG(ters, i + H));
    g  = LDG(ters, i + GAMMA) *
                (LDG(ters, i + ONE_PLUS_C2OVERD2) - LDG(ters, i + C2) / temp);
    gp = LDG(ters, i + GAMMA) *
            (TWO * LDG(ters, i + C2) * (cos - LDG(ters, i + H)) / (temp * temp));
}


static __device__ void find_g
(int i, const double* __restrict__ ters, double cos, double &g)
{
    double temp = LDG(ters, i + D2) + (cos - LDG(ters, i + H)) *
                (cos - LDG(ters, i + H));
    g  = LDG(ters, i + GAMMA) *
                (LDG(ters, i + ONE_PLUS_C2OVERD2) - LDG(ters, i + C2) / temp);
}


static __device__ void find_e_and_ep
(int i, const double* __restrict__ ters, double d12, double d13, double &e, double &ep)
{
    if (LDG(ters, i + ALPHA) < EPSILON){ e = ONE; ep = ZERO;}
    else
    {
        double r = d12 - d13;
        if (LDG(ters, i + M) > TWO) //if m == 3.0
        {
            e = exp(LDG(ters, i + ALPHA) * r * r * r);
            ep = LDG(ters, i + ALPHA) * THREE * r * r * e;
        }
        else
        {
            e = exp(LDG(ters, i + ALPHA) * r);
            ep = LDG(ters, i + ALPHA) * e;
        }
    }
}

static __device__ void find_e
(int i, const double* __restrict__ ters, double d12, double d13, double &e)
{
    if (LDG(ters, i + ALPHA) < EPSILON){ e = ONE;}
    else
    {
        double r = d12 - d13;
        if (LDG(ters, i + M) > TWO){ e = exp(LDG(ters, i + ALPHA) * r * r * r);}
        else{e = exp(LDG(ters, i + ALPHA) * r);}
    }
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
        double x1 = LDG(g_x, n1); double y1 = LDG(g_y, n1); double z1 = LDG(g_z, n1);
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int type2 = g_type[n2] - shift;
            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double zeta = ZERO;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3] - shift;
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(box, x13, y13, z13);
                double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                double fc_ijk_13, g_ijk, e_ijk_12_13;
                int ijk = type1 * num_types2 + type2 * num_types + type3;
                if (d13 > LDG(ters, ijk*NUM_PARAMS + R2)) {continue;}
                find_fc(ijk*NUM_PARAMS, ters, d13, fc_ijk_13);
                find_g(ijk*NUM_PARAMS, ters, cos123, g_ijk);
                find_e(ijk*NUM_PARAMS, ters, d12, d13, e_ijk_12_13);
                zeta += fc_ijk_13 * g_ijk * e_ijk_12_13;
            }
            double bzn, b_ijj;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            bzn = pow(LDG(ters, ijj*NUM_PARAMS + BETA) *
                zeta, LDG(ters, ijj*NUM_PARAMS + EN));
            b_ijj = pow(ONE + bzn, LDG(ters, ijj*NUM_PARAMS + MINUS_HALF_OVER_N));
            if (zeta < 1.0e-16) // avoid division by 0
            {
                g_b[i1 * number_of_particles + n1]  = ONE;
                g_bp[i1 * number_of_particles + n1] = ZERO;
            }
            else
            {
                g_b[i1 * number_of_particles + n1]  = b_ijj;
                g_bp[i1 * number_of_particles + n1]
                    = - b_ijj * bzn * HALF / ((ONE + bzn) * zeta);
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
        double x1 = LDG(g_x, n1); double y1 = LDG(g_y, n1); double z1 = LDG(g_z, n1);
        double pot_energy = ZERO;
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2] - shift;

            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double d12inv = ONE / d12;
            double fc_ijj_12, fcp_ijj_12;
            double fa_ijj_12, fap_ijj_12, fr_ijj_12, frp_ijj_12;
            int ijj = type1 * num_types2 + type2 * num_types + type2;
            find_fc_and_fcp(ijj*NUM_PARAMS, ters, d12, fc_ijj_12, fcp_ijj_12);
            find_fa_and_fap(ijj*NUM_PARAMS, ters, d12, fa_ijj_12, fap_ijj_12);
            find_fr_and_frp(ijj*NUM_PARAMS, ters, d12, fr_ijj_12, frp_ijj_12);

            // (i,j) part
            double b12 = LDG(g_b, index);
            double factor3=(fcp_ijj_12*(fr_ijj_12-b12*fa_ijj_12)+
                          fc_ijj_12*(frp_ijj_12-b12*fap_ijj_12))*d12inv;
            double f12x = x12 * factor3 * HALF;
            double f12y = y12 * factor3 * HALF;
            double f12z = z12 * factor3 * HALF;

            // accumulate potential energy
            pot_energy += fc_ijj_12 * (fr_ijj_12 - b12 * fa_ijj_12) * HALF;

            // (i,j,k) part
            double bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3] - shift;
                double x13 = LDG(g_x, n3) - x1;
                double y13 = LDG(g_y, n3) - y1;
                double z13 = LDG(g_z, n3) - z1;
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
                double bp13 = LDG(g_bp, index_2);
                double one_over_d12d13 = ONE / (d12 * d13);
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
                f12x += (x12 * dr + dc * cos_d)*HALF;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * dr + dc * cos_d)*HALF;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * dr + dc * cos_d)*HALF;
            }
            g_f12x[index] = f12x; g_f12y[index] = f12y; g_f12z[index] = f12z;
        }
        // save potential
        g_potential[n1] += pot_energy;
    }
}


// Wrapper of force evaluation for the Tersoff potential
void Tersoff1988::compute(Atom *atom, int potential_number)
{
    int N = atom->N;
    int shift = atom->shift[potential_number];
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    int *NN = atom->NN_local;
    int *NL = atom->NL_local;
    int *type = atom->type;
    double *x = atom->x;
    double *y = atom->y;
    double *z = atom->z;
    double *pe = atom->potential_per_atom;

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
