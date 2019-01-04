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




#include "tersoff.cuh"
#include "ldg.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"


#define BLOCK_SIZE_FORCE 64 // 128 is also good

#ifdef USE_DP
    #define HALF  0.5
    #define ZERO  0.0
    #define ONE   1.0
    #define TWO   2.0
    #define THREE 3.0
    #define PI    3.14159265358979
    #define ONE_OVER_16   0.0625
    #define THREE_OVER_16 0.1875
    #define NINE_OVER_16  0.5625
#else
    #define HALF  0.5f
    #define ZERO  0.0f
    #define ONE   1.0f
    #define TWO   2.0f
    #define THREE 3.0f
    #define PI    3.141593f
    #define ONE_OVER_16   0.0625f
    #define THREE_OVER_16 0.1875f
    #define NINE_OVER_16  0.5625f
#endif




/*----------------------------------------------------------------------------80
The double-element version of the Tersoff potential as described in  
    [1] J. Tersoff, Modeling solid-state chemistry: Interatomic potentials 
        for multicomponent systems, PRB 39, 5566 (1989).
------------------------------------------------------------------------------*/




Tersoff2::Tersoff2(FILE *fid, Atom* atom, int num_of_types)
{
    if (num_of_types == 1)
        printf("INPUT: use Tersoff-1989 (single-element) potential.\n");
    if (num_of_types == 2)
        printf("INPUT: use Tersoff-1989 (double-element) potential.\n");

    // first line
    int count;
    double a, b,lambda, mu, beta, n, c, d, h, r1, r2;
    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
        &a, &b, &lambda, &mu, &beta, &n, &c, &d, &h, &r1, &r2
    );
    if (count!=11) {printf("Error: reading error for potential.in.\n");exit(1);}
    ters0.a      = a;
    ters0.b      = b;
    ters0.lambda = lambda;
    ters0.mu     = mu;
    ters0.beta   = beta;
    ters0.n      = n;
    ters0.c      = c;
    ters0.d      = d;
    ters0.h      = h;
    ters0.r1     = r1;
    ters0.r2     = r2;
    ters0.c2 = c * c;
    ters0.d2 = d * d;
    ters0.one_plus_c2overd2 = 1.0 + ters0.c2 / ters0.d2;
    ters0.pi_factor = PI / (r2 - r1);
    ters0.minus_half_over_n = - 0.5 / n;
    rc = ters0.r2;

    if (num_of_types == 2)
    { 
        // second line
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
            &a, &b, &lambda, &mu, &beta, &n, &c, &d, &h, &r1, &r2
        );
        if (count!=11) 
            {printf("Error: reading error for potential.in.\n");exit(1);}
        ters1.a      = a;
        ters1.b      = b;
        ters1.lambda = lambda;
        ters1.mu     = mu;
        ters1.beta   = beta;
        ters1.n      = n;
        ters1.c      = c;
        ters1.d      = d;
        ters1.h      = h;
        ters1.r1     = r1;
        ters1.r2     = r2;
        ters1.c2 = c * c;
        ters1.d2 = d * d;
        ters1.one_plus_c2overd2 = 1.0 + ters1.c2 / ters1.d2;
        ters1.pi_factor = PI / (r2 - r1);
        ters1.minus_half_over_n = - 0.5 / n;

        // third line
        double chi;
        count = fscanf(fid, "%lf", &chi);
        if (count != 1)
            {print_error("reading error for potential.in.\n"); exit(1);}

        // mixing type 0 and type 1
        ters2.a = sqrt(ters0.a * ters1.a);
        ters2.b = sqrt(ters0.b * ters1.b);
        ters2.b *= chi;
        ters2.lambda = 0.5 * (ters0.lambda + ters1.lambda);
        ters2.mu     = 0.5 * (ters0.mu     + ters1.mu);
        ters2.beta = 0.0; // not used
        ters2.n = 0.0;    // not used
        ters2.c2 = 0.0;   // not used
        ters2.d2 = 0.0;   // not used
        ters2.h = 0.0;    // not used
        ters2.r1 = sqrt(ters0.r1 * ters1.r1);
        ters2.r2 = sqrt(ters0.r2 * ters1.r2);
        ters2.one_plus_c2overd2 = 0.0; // not used
        ters2.pi_factor = PI / (ters2.r2 - ters2.r1);
        ters2.minus_half_over_n = 0.0; // not used

        // force cutoff
        rc = (ters0.r2 > ters1.r2) ?  ters0.r2 : ters1.r2;
    }

    // memory for the bond-order function b and its derivative bp
    int memory = sizeof(real) * atom->N * atom->neighbor.MN;
    CHECK(cudaMalloc((void**)&tersoff_data.b,  memory));
    CHECK(cudaMalloc((void**)&tersoff_data.bp, memory));

    // memory for the partial forces dU_i/dr_ij
    memory = sizeof(real) * atom->N * 20;
    CHECK(cudaMalloc((void**)&tersoff_data.f12x, memory));
    CHECK(cudaMalloc((void**)&tersoff_data.f12y, memory));
    CHECK(cudaMalloc((void**)&tersoff_data.f12z, memory));
}




Tersoff2::~Tersoff2(void)
{
    CHECK(cudaFree(tersoff_data.b));
    CHECK(cudaFree(tersoff_data.bp));
    CHECK(cudaFree(tersoff_data.f12x));
    CHECK(cudaFree(tersoff_data.f12y));
    CHECK(cudaFree(tersoff_data.f12z));
}




static __device__ void find_fr_and_frp
(
    int type1, int type2,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2,
    real d12, real &fr, real &frp
)
{
    if (type1 == 0 && type2 == 0)
    {   
        fr  = ters0.a * exp(- ters0.lambda * d12);
        frp = - ters0.lambda * fr;
    }
    else if (type1 == 1 && type2 == 1)
    {
        fr  = ters1.a * exp(- ters1.lambda * d12);
        frp = - ters1.lambda * fr;
    }
    else
    {
        fr  = ters2.a * exp(- ters2.lambda * d12);
        frp = - ters2.lambda * fr;
    }
}




static __device__ void find_fa_and_fap
(
#ifdef CBN
    real mass_sum,
#endif
    int type1, int type2,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2,
    real d12, real &fa, real &fap
)
{
    if (type1 == 0 && type2 == 0)
    {
        fa  = ters0.b * exp(- ters0.mu * d12);
        fap = - ters0.mu * fa;
    }
    else if (type1 == 1 && type2 == 1)
    {
        fa  = ters1.b * exp(- ters1.mu * d12);
        fap = - ters1.mu * fa;
    }
    else
    {
        fa = ters2.b * exp(- ters2.mu * d12);
#ifdef CBN // special case of CB and CN interactions
        if (mass_sum < 24.0) // CB bond is weaker
            fa *= 0.886777;
        else                 // CN bond is stronger
            fa *= 1.013636;
#endif
        fap = - ters2.mu * fa;
    }
}




static __device__ void find_fa
(
#ifdef CBN
    real mass_sum,
#endif
    int type1, int type2,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2,
    real d12, real &fa
)
{
    if (type1 == 0 && type2 == 0)
    {
        fa  = ters0.b * exp(- ters0.mu * d12);
    }
    else if (type1 == 1 && type2 == 1)
    {
        fa  = ters1.b * exp(- ters1.mu * d12);
    }
    else
    {
        fa = ters2.b * exp(- ters2.mu * d12);
#ifdef CBN // special case of CB and CN interactions
        if (mass_sum < 24.0) // CB bond is weaker
            fa *= 0.886777;
        else                 // CN bond is stronger
            fa *= 1.013636;
#endif
    }
}




static __device__ void find_fc_and_fcp
(
    int type1, int type2,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2,
    real d12, real &fc, real &fcp
)
{
    if (type1 == 0 && type2 == 0)
    { 
        if (d12 < ters0.r1) {fc = ONE; fcp = ZERO;}
        else if (d12 < ters0.r2)
        {
#ifdef TERSOFF_CUTOFF
            fc  =  cos(ters0.pi_factor * (d12 - ters0.r1)) * HALF + HALF;
            fcp = -sin(ters0.pi_factor * (d12 - ters0.r1))*ters0.pi_factor*HALF;
#else
            fc = NINE_OVER_16 * cos(ters0.pi_factor * (d12 - ters0.r1)) 
                - ONE_OVER_16 * cos(ters0.pi_factor * (d12 - ters0.r1) * THREE)
                + HALF;
            fcp = sin(ters0.pi_factor * (d12 - ters0.r1) * THREE) 
                * ters0.pi_factor * THREE_OVER_16
                - sin(ters0.pi_factor * (d12 - ters0.r1)) 
                * ters0.pi_factor * NINE_OVER_16;
#endif
        }
        else {fc  = ZERO; fcp = ZERO;}
    }
    else if (type1 == 1 && type2 == 1)
    {
        if (d12 < ters1.r1) {fc  = ONE; fcp = ZERO;}
        else if (d12 < ters1.r2)
        {
#ifdef TERSOFF_CUTOFF
            fc  =  cos(ters1.pi_factor * (d12 - ters1.r1)) * HALF + HALF;
            fcp = -sin(ters1.pi_factor * (d12 - ters1.r1))*ters1.pi_factor*HALF;
#else
            fc = NINE_OVER_16 * cos(ters1.pi_factor * (d12 - ters1.r1)) 
                - ONE_OVER_16 * cos(ters1.pi_factor * (d12 - ters1.r1) * THREE)
                + HALF;
            fcp = sin(ters1.pi_factor * (d12 - ters1.r1) * THREE) 
                * ters1.pi_factor * THREE_OVER_16
                - sin(ters1.pi_factor * (d12 - ters1.r1)) 
                * ters1.pi_factor * NINE_OVER_16;
#endif
        }
        else {fc = ZERO; fcp = ZERO;}
    }
    else
    {
        if (d12 < ters2.r1) {fc  = ONE; fcp = ZERO;}
        else if (d12 < ters2.r2)
        {
#ifdef TERSOFF_CUTOFF
            fc  =  cos(ters2.pi_factor * (d12 - ters2.r1)) * HALF + HALF;
            fcp = -sin(ters2.pi_factor * (d12 - ters2.r1))*ters2.pi_factor*HALF;
#else
            fc = NINE_OVER_16 * cos(ters2.pi_factor * (d12 - ters2.r1)) 
                - ONE_OVER_16 * cos(ters2.pi_factor * (d12 - ters2.r1) * THREE)
                + HALF;
            fcp = sin(ters2.pi_factor * (d12 - ters2.r1) * THREE) 
                * ters2.pi_factor * THREE_OVER_16
                - sin(ters2.pi_factor * (d12 - ters2.r1)) 
                * ters2.pi_factor * NINE_OVER_16;
#endif
        }
        else {fc  = ZERO; fcp = ZERO;}
    }
}




static __device__ void find_fc
(
    int type1, int type2,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2,
    real d12, real &fc
)
{
    if (type1 == 0 && type2 == 0)
    {
        if (d12 < ters0.r1) {fc  = ONE;}
        else if (d12 < ters0.r2)
        {
#ifdef TERSOFF_CUTOFF
            fc  =  cos(ters0.pi_factor * (d12 - ters0.r1)) * HALF + HALF;
#else
            fc = NINE_OVER_16 * cos(ters0.pi_factor * (d12 - ters0.r1)) 
                - ONE_OVER_16 * cos(ters0.pi_factor * (d12 - ters0.r1) * THREE)
                + HALF;
#endif
        }
        else {fc  = ZERO;}
    }
    else if (type1 == 1 && type2 == 1)
    {
        if (d12 < ters1.r1) {fc  = ONE;}
        else if (d12 < ters1.r2)
        {
#ifdef TERSOFF_CUTOFF
            fc  =  cos(ters1.pi_factor * (d12 - ters1.r1)) * HALF + HALF;
#else
            fc = NINE_OVER_16 * cos(ters1.pi_factor * (d12 - ters1.r1)) 
                - ONE_OVER_16 * cos(ters1.pi_factor * (d12 - ters1.r1) * THREE)
                + HALF;
#endif
        }
        else {fc  = ZERO;}
    }
    else
    {
        if (d12 < ters2.r1) {fc  = ONE;}
        else if (d12 < ters2.r2) 
        {
#ifdef TERSOFF_CUTOFF
            fc  =  cos(ters2.pi_factor * (d12 - ters2.r1)) * HALF + HALF;
#else
            fc = NINE_OVER_16 * cos(ters2.pi_factor * (d12 - ters2.r1)) 
                - ONE_OVER_16 * cos(ters2.pi_factor * (d12 - ters2.r1) * THREE)
                + HALF;
#endif
        }
        else {fc  = ZERO;}
    }
}




static __device__ void find_g_and_gp
(
    int type1,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    real cos, real &g, real &gp
)
{
    if (type1 == 0)
    {
        real temp = ters0.d2 + (cos - ters0.h) * (cos - ters0.h);
        g  = ters0.one_plus_c2overd2 - ters0.c2 / temp;
        gp = TWO * ters0.c2 * (cos - ters0.h) / (temp * temp);
    }
    else
    {
        real temp = ters1.d2 + (cos - ters1.h) * (cos - ters1.h);
        g  = ters1.one_plus_c2overd2 - ters1.c2 / temp;
        gp = TWO * ters1.c2 * (cos - ters1.h) / (temp * temp);
    }
}




static __device__ void find_g
(
    int type1,
    Tersoff2_Parameters ters0,
    Tersoff2_Parameters ters1,
    real cos, real &g
)
{
    if (type1 == 0)
    {
        real temp = ters0.d2 + (cos - ters0.h) * (cos - ters0.h);
        g  = ters0.one_plus_c2overd2 - ters0.c2 / temp;
    }
    else
    {
        real temp = ters1.d2 + (cos - ters1.h) * (cos - ters1.h);
        g  = ters1.one_plus_c2overd2 - ters1.c2 / temp;
    }
}




static __device__ void dev_apply_mic
(
    int pbc_x, int pbc_y, int pbc_z, real &x12, real &y12, real &z12, 
    real lx, real ly, real lz
)
{
    if      (pbc_x == 1 && x12 < - lx * HALF) {x12 += lx;}
    else if (pbc_x == 1 && x12 > + lx * HALF) {x12 -= lx;}
    if      (pbc_y == 1 && y12 < - ly * HALF) {y12 += ly;}
    else if (pbc_y == 1 && y12 > + ly * HALF) {y12 -= ly;}
    if      (pbc_z == 1 && z12 < - lz * HALF) {z12 += lz;}
    else if (pbc_z == 1 && z12 > + lz * HALF) {z12 -= lz;}
}




// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1
(
    int number_of_particles, int N1, int N2, int pbc_x, int pbc_y, int pbc_z,
    Tersoff2_Parameters ters0, Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2,
    int* g_neighbor_number, int* g_neighbor_list, int* g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x,
    const real* __restrict__ g_y,
    const real* __restrict__ g_z,
    const real* __restrict__ g_box_length,
#else
    real* g_x, real* g_y, real* g_z, real* g_box_length,
#endif
    real* g_b, real* g_bp
)
{
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); real y1 = LDG(g_y, n1); real z1 = LDG(g_z, n1);
        real lx = LDG(g_box_length, 0);
        real ly = LDG(g_box_length, 1);
        real lz = LDG(g_box_length, 2);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real zeta = ZERO;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                real cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                real fc13, g123;
                find_fc(type1, type3, ters0, ters1, ters2, d13, fc13);
                find_g(type1, ters0, ters1, cos123, g123);
                zeta += fc13 * g123;
            }
            real bzn, b12;
            if (type1 == 0)
            {
                bzn = pow(ters0.beta * zeta, ters0.n);
                b12 = pow(ONE + bzn, ters0.minus_half_over_n);
            }
            else
            {
                bzn = pow(ters1.beta * zeta, ters1.n);
                b12 = pow(ONE + bzn, ters1.minus_half_over_n);
            }
            if (zeta < 1.0e-16) // avoid division by 0
            {
                g_b[i1 * number_of_particles + n1]  = ONE;
                g_bp[i1 * number_of_particles + n1] = ZERO;
            }
            else
            {
                g_b[i1 * number_of_particles + n1]  = b12;
                g_bp[i1 * number_of_particles + n1] 
                    = - b12 * bzn * HALF / ((ONE + bzn) * zeta);
            }
        }
    }
}




// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_tersoff_step2
(
#ifdef CBN
    real *g_mass,
#endif
    int number_of_particles, int N1, int N2, int pbc_x, int pbc_y, int pbc_z,
    Tersoff2_Parameters ters0, Tersoff2_Parameters ters1,
    Tersoff2_Parameters ters2, 
    int *g_neighbor_number, int *g_neighbor_list, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_b,
    const real* __restrict__ g_bp,
    const real* __restrict__ g_x,
    const real* __restrict__ g_y,
    const real* __restrict__ g_z,
    const real* __restrict__ g_box_length,
#else
    real* g_b, real* g_bp, real* g_x, real* g_y, real* g_z, real* g_box_length,
#endif
    real *g_potential, real *g_f12x, real *g_f12y, real *g_f12z
)
{
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); real y1 = LDG(g_y, n1); real z1 = LDG(g_z, n1);
        real lx = LDG(g_box_length, 0);
        real ly = LDG(g_box_length, 1);
        real lz = LDG(g_box_length, 2);
        real potential_energy = ZERO;

#ifdef CBN
        real mass_1 = g_mass[n1];
#endif

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real d12inv = ONE / d12;
            real fc12, fcp12, fa12, fap12, fr12, frp12;
            find_fc_and_fcp
            (type1, type2, ters0, ters1, ters2, d12, fc12, fcp12);
            find_fa_and_fap
            (
#ifdef CBN
                mass_1 + g_mass[n2],
#endif
                type1, type2, ters0, ters1, ters2, d12, fa12, fap12
            );
            find_fr_and_frp
            (type1, type2, ters0, ters1, ters2, d12, fr12, frp12);

            // (i,j) part
            real b12 = LDG(g_b, index);
            real factor3=(fcp12*(fr12-b12*fa12)+fc12*(frp12-b12*fap12))*d12inv;
            real f12x = x12 * factor3 * HALF;
            real f12y = y12 * factor3 * HALF;
            real f12z = z12 * factor3 * HALF;

            // accumulate potential energy
            potential_energy += fc12 * (fr12 - b12 * fa12) * HALF;

            // (i,j,k) part
            real bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;
                int n3 = g_neighbor_list[index_2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                real fc13, fa13;
                find_fc(type1, type3, ters0, ters1, ters2, d13, fc13);
                find_fa
                (
#ifdef CBN
                    mass_1 + g_mass[n3],
#endif
                    type1, type3, ters0, ters1, ters2, d13, fa13
                );

                real bp13 = LDG(g_bp, index_2);
                real one_over_d12d13 = ONE / (d12 * d13);
                real cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv;
                real g123, gp123;
                find_g_and_gp(type1, ters0, ters1, cos123, g123, gp123);

                real temp123a=(-bp12*fc12*fa12*fc13-bp13*fc13*fa13*fc12)*gp123;
                real temp123b= - bp13 * fc13 * fa13 * fcp12 * g123 * d12inv;
                real cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * temp123b + temp123a * cos_d)*HALF;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * temp123b + temp123a * cos_d)*HALF;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * temp123b + temp123a * cos_d)*HALF;
            }
            g_f12x[index] = f12x; g_f12y[index] = f12y; g_f12z[index] = f12z;
        }
        // save potential
        g_potential[n1] += potential_energy;
    }
}




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
    int *g_a_map, int *g_b_map, int g_count_b
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




// Wrapper of force evaluation for the Tersoff potential
void Tersoff2::compute(Atom *atom, Measure *measure)
{
    int N = atom->N;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    int pbc_x = atom->pbc_x;
    int pbc_y = atom->pbc_y;
    int pbc_z = atom->pbc_z;
    int *NN = atom->NN_local;
    int *NL = atom->NL_local;
    int *type = atom->type_local;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *vx = atom->vx;
    real *vy = atom->vy;
    real *vz = atom->vz;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    real *box_length = atom->box_length;
    real *sx = atom->virial_per_atom_x;
    real *sy = atom->virial_per_atom_y;
    real *sz = atom->virial_per_atom_z;
    real *pe = atom->potential_per_atom;
    real *h = atom->heat_per_atom;

    // data related to the SHC method
    int *label = atom->label;
    int *fv_index = measure->shc.fv_index;
    int *a_map = measure->shc.a_map;
    int *b_map = measure->shc.b_map;
    int count_b = measure->shc.count_b;
    real *fv = measure->shc.fv;

    // special data for Tersoff potential
    real *f12x = tersoff_data.f12x;
    real *f12y = tersoff_data.f12y;
    real *f12z = tersoff_data.f12z;
    real *b    = tersoff_data.b;
    real *bp   = tersoff_data.bp;

    // parameters related to the HNEMD method
    real fe_x = measure->hnemd.fe_x;
    real fe_y = measure->hnemd.fe_y;
    real fe_z = measure->hnemd.fe_z;

    // pre-compute the bond order functions and their derivatives
    find_force_tersoff_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, pbc_x, pbc_y, pbc_z, ters0, ters1, ters2,
        NN, NL, type, x, y, z, box_length, b, bp
    );
    CUDA_CHECK_KERNEL

    // pre-compute the partial forces
    find_force_tersoff_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
#ifdef CBN
        atom->mass,
#endif
        N, N1, N2, pbc_x, pbc_y, pbc_z, ters0, ters1, ters2,
        NN, NL, type, b, bp, x, y, z, box_length, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL

    // the final step: calculate force and related quantities
    find_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        measure->hac.compute, measure->shc.compute, measure->hnemd.compute,
        fe_x, fe_y, fe_z, N, N1, N2, pbc_x, pbc_y, pbc_z, NN, NL,
        f12x, f12y, f12z, x, y, z, vx, vy, vz, box_length, fx, fy, fz,
        sx, sy, sz, h, label, fv_index, fv, a_map, b_map, count_b
    );
    CUDA_CHECK_KERNEL
}




