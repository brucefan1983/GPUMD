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
The REBO potential for C-H systems
References: 
    [1] D. W. Brenner et al., JPCM, 14, 783 (2002).
Not finished.
------------------------------------------------------------------------------*/


#include "rebo_ch.cuh"
#include "mic.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"
#define BLOCK_SIZE_FORCE 64
#define Q 0
#define A 1
#define ALPHA 2
#define B1 3
#define B3 4
#define BETA1 5
#define BETA3 6
#define PI_FACTOR 7
#define R1 8
#define R2 9


REBO_CH::REBO_CH(Atom* atom)
{
    int num = ((atom->neighbor.MN<20) ? atom->neighbor.MN : 20);
    int memory1 = sizeof(real) * atom->N;
    int memory2 = sizeof(real) * atom->N * num;
    CHECK(cudaMalloc((void**)&rebo_ch_data.p,    memory1));
    CHECK(cudaMalloc((void**)&rebo_ch_data.pp,   memory1));
    CHECK(cudaMalloc((void**)&rebo_ch_data.b,    memory2));
    CHECK(cudaMalloc((void**)&rebo_ch_data.bp,   memory2));
    CHECK(cudaMalloc((void**)&rebo_ch_data.f12x, memory2));
    CHECK(cudaMalloc((void**)&rebo_ch_data.f12y, memory2));
    CHECK(cudaMalloc((void**)&rebo_ch_data.f12z, memory2));
    printf("Use the REBO potential for C-H systems.\n");
}


REBO_CH::~REBO_CH(void)
{
    CHECK(cudaFree(rebo_ch_data.p));
    CHECK(cudaFree(rebo_ch_data.pp));
    CHECK(cudaFree(rebo_ch_data.b));
    CHECK(cudaFree(rebo_ch_data.bp));
    CHECK(cudaFree(rebo_ch_data.f12x));
    CHECK(cudaFree(rebo_ch_data.f12y));
    CHECK(cudaFree(rebo_ch_data.f12z));
}


static __device__ void find_fr_and_frp
(int i, const real* __restrict__ para, real d12, real& fr, real& frp)
{
    fr  = (ONE + LDG(para, i+Q) / d12) * A * exp(-LDG(para, i+ALPHA) * d12);
    frp = LDG(para, i+ALPHA) + LDG(para, i+Q) / (d12 * (d12 + LDG(para, i+Q)));
    frp *= -fr;
}


static __device__ void find_fa_and_fap
(int i, const real* __restrict__ para, real d12, real &fa, real &fap)
{
    real tmp = LDG(para, i+B1) * exp(- LDG(para, i+BETA1) * d12);
    fa  = tmp;
    fap = - LDG(para, i+BETA1) * tmp;
    tmp = LDG(para, i+B3) * exp(- LDG(para, i+BETA3) * d12);
    fa  += tmp;
    fap -= LDG(para, i+BETA3) * tmp;
}


static __device__ void find_fa
(int i, const real* __restrict__ para, real d12, real &fa)
{
    fa = LDG(para, i+B1) * exp(- LDG(para, i+BETA1) * d12)
       + LDG(para, i+B3) * exp(- LDG(para, i+BETA3) * d12);
}


static __device__ void find_fc_and_fcp
(int i, const real* __restrict__ para, real d12, real &fc, real &fcp)
{
    if (d12 < LDG(para, i + R1)) {fc = ONE; fcp = ZERO;}
    else if (d12 < LDG(para, i + R2))
    {
        fc = cos(LDG(para, i+PI_FACTOR) *(d12 - LDG(para, i+R1))) * HALF + HALF;
        fcp = -sin(LDG(para, i + PI_FACTOR) * (d12 - LDG(para, i + R1)))
            * LDG(para, i + PI_FACTOR) * HALF;
    }
    else {fc  = ZERO; fcp = ZERO;}
}


static __device__ void find_fc
(int i, const real* __restrict__ para, real d12, real &fc)
{
    if (d12 < LDG(para, i + R1)) {fc = ONE;}
    else if (d12 < LDG(para, i + R2))
    {
        fc = cos(LDG(para, i+PI_FACTOR) *(d12 - LDG(para, i+R1))) * HALF + HALF;
    }
    else {fc  = ZERO;}
}


static __device__ void find_g_and_gp
(int i, const real* __restrict__ para, real x, real &g, real &gp)
{
    // to be written
}


static __device__ void find_g
(int i, const real* __restrict__ para, real x, real &g)
{
    // to be written
}


static __global__ void find_force_step1
(
    int N, int N1, int N2, int triclinic, int pbc_x, int pbc_y, int pbc_z,
    int* g_NN, int* g_NL, int* g_type,
    const real* __restrict__ para, 
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z,
    const real* __restrict__ g_box,
    real* g_b, real* g_bp, real *g_p
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real p = g_p[n1]; // coordination number function P(N)

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_NL[n1 + N * i1];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;

            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

            real zeta = ZERO;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_NL[n1 + N * i2];
                if (n3 == n2) { continue; } // ensure that n3 != n2
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;

                dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box,
                    x13, y13, z13);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                real cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                real fc13, g123; 
                int type13 = type1 + type3;
                find_fc(type13, para, d13, fc13);
                find_g(type1, para, cos123, g123);
                zeta += fc13 * g123;
            }

            zeta += p;
            real b12 = pow(ONE + zeta, -HALF);
            g_b[i1 * N + n1]  = b12;
            g_bp[i1 * N + n1] = (-HALF)*b12/(ONE+zeta); 
        }
    }
}


// calculate and save the partial forces dU_i/dr_ij
static __global__ void find_force_step2
(
    int N, int N1, int N2, int triclinic, int pbc_x, int pbc_y, int pbc_z,
    int *g_NN, int *g_NL, int *g_type,
    const real* __restrict__ para, 
    const real* __restrict__ g_b, 
    const real* __restrict__ g_bp,
    const real* __restrict__ g_pp,
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z,
    const real* __restrict__ g_box, 
    real *g_potential, real *g_f12x, real *g_f12y, real *g_f12z
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_NN[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real pp1 = LDG(g_pp, n1); 
        real potential_energy = ZERO;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int index = i1 * N + n1;
            int n2 = g_NL[index];
            int type2 = g_type[n2];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, x12, y12, z12);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real d12inv = ONE / d12;

            real fc12, fcp12, fa12, fap12, fr12, frp12;
            int type12 = type1 + type2;
            find_fc_and_fcp(type12, para, d12, fc12, fcp12);
            find_fa_and_fap(type12, para, d12, fa12, fap12);
            find_fr_and_frp(type12, para, d12, fr12, frp12);

            // accumulate_force_12 
            real b12 = LDG(g_b, index);
            real bp12 = LDG(g_bp, index);
            real factor3 = (fcp12*(fr12-b12*fa12) + fc12*(frp12-b12*fap12) 
                         - fc12*fcp12*fa12*bp12*pp1)/d12;
            real f12x = x12 * factor3 * HALF;
            real f12y = y12 * factor3 * HALF;
            real f12z = z12 * factor3 * HALF;

            // accumulate potential energy
            potential_energy += fc12 * (fr12 - b12 * fa12) * HALF;

            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_NL[n1 + N * i2];
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(triclinic, pbc_x, pbc_y, pbc_z, g_box, 
                    x13, y13, z13);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);

                real fc13, fa13;
                int type13 = type1 + type3;
                find_fc(type13, para, d13, fc13);
                find_fa(type13, para, d13, fa13);
                real bp13 = LDG(g_bp, i2 * N + n1);
                real one_over_d12d13 = ONE / (d12 * d13);
                real cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv;
                real g123, gp123;
                find_g_and_gp(type1, para, cos123, g123, gp123);

                real temp123a=(-bp12*fc12*fa12*fc13-bp13*fc13*fa13*fc12)*gp123;
                real temp123b= - bp13 * fc13 * fa13 * fcp12 * (g123+pp1) / d12;
                real cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
                f12x += (x12 * temp123b + temp123a * cos_d)*HALF;
                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += (y12 * temp123b + temp123a * cos_d)*HALF;
                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += (z12 * temp123b + temp123a * cos_d)*HALF;
            }
            g_f12x[index] = f12x;
            g_f12y[index] = f12y;
            g_f12z[index] = f12z;
        }
        // accumulate potential energy on top of the 2-body part
        g_potential[n1] += potential_energy;
    }
}


// Force evaluation wrapper
void REBO_CH::compute(Atom *atom, Measure *measure)
{
    int N = atom->N;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
    int triclinic = atom->box.triclinic;
    int pbc_x = atom->box.pbc_x;
    int pbc_y = atom->box.pbc_y;
    int pbc_z = atom->box.pbc_z;
    int *NN = atom->NN_local;
    int *NL = atom->NL_local;
    int *type = atom->type_local;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *box = atom->box.h;
    real *pe = atom->potential_per_atom;
    real *b    = rebo_ch_data.b;
    real *bp   = rebo_ch_data.bp;
    real *p    = rebo_ch_data.p;
    real *pp   = rebo_ch_data.pp;
    real *f12x = rebo_ch_data.f12x;
    real *f12y = rebo_ch_data.f12y;
    real *f12z = rebo_ch_data.f12z;
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, triclinic, pbc_x, pbc_y, pbc_z, NN, NL, type, para,
        x, y, z, box, b, bp, p
    );
    CUDA_CHECK_KERNEL
    find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, triclinic, pbc_x, pbc_y, pbc_z, NN, NL, type, para,
        b, bp, pp, x, y, z, box, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL
    find_properties_many_body(atom, measure, NN, NL, f12x, f12y, f12z);
}


