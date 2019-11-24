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
The minimal Tersoff potential
    Written by Zheyong Fan.
    This is a new potential I proposed. 
------------------------------------------------------------------------------*/


#include "tersoff_mini.cuh"
#include "mic.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"

#define BLOCK_SIZE_FORCE 64

// Easy labels for indexing
#define A                 0
#define B                 1
#define LAMBDA            2
#define MU                3
#define BETA              4
#define EN                5
#define H                 6
#define R1                7
#define R2                8
#define PI_FACTOR         9
#define MINUS_HALF_OVER_N 10
#define NUM_PARAMS        11


Tersoff_mini::Tersoff_mini(FILE *fid, Atom* atom, int num_of_types)
{
    num_types = num_of_types;
    printf("Use Tersoff-mini (%d-element) potential.\n", num_types);
    int n_entries = num_types*num_types*num_types;
    double *cpu_para;
    MY_MALLOC(cpu_para, double, n_entries*NUM_PARAMS);

    const char err[] = "Error: Illegal SBOP parameter.";
    rc = 0.0;
    int count;
    double d0, a, r0, beta, n, h, r1, r2;
    for (int i = 0; i < n_entries; i++)
    {
        count = fscanf
        (
            fid, "%lf%lf%lf%lf%lf%lf%lf%lf",
            &d0, &a, &r0, &beta, &n, &h, &r1, &r2
        );
        if (count != 8) 
            {printf("Error: reading error for SBOP potential.\n"); exit(1);}

        if (d0 <= 0.0)
            {printf("%s D0 must be > 0.\n",err); exit(1);}
        if (a <= 0.0)
            {printf("%s a must be > 0.\n",err); exit(1);}
        if (r0 <= 0.0)
            {printf("%s r0 must be > 0.\n",err); exit(1);}
        if(beta < 0.0)
            {printf("%s beta must be >= 0.\n",err); exit(1);}
        if(n < 0.0)
            {printf("%s n must be >= 0.\n",err); exit(1);}
        if(h < -1.0 || h > 1.0)
            {printf("%s |h| must be <= 1.\n",err); exit(1);}
        if(r1 < 0.0)
            {printf("%s R1 must be >= 0.\n",err); exit(1);}
        if(r2 <= 0.0)
            {printf("%s R2 must be > 0.\n",err); exit(1);}
        if(r2 <= r1)
            {printf("%s R2-R1 must be > 0.\n",err); exit(1);}

        cpu_para[i*NUM_PARAMS + A] = d0 * exp(2.0 * a * r0);
        cpu_para[i*NUM_PARAMS + B] = 2.0 * d0 * exp(a * r0);
        cpu_para[i*NUM_PARAMS + LAMBDA] = 2.0 * a;
        cpu_para[i*NUM_PARAMS + MU] = a;
        cpu_para[i*NUM_PARAMS + BETA] = beta;
        cpu_para[i*NUM_PARAMS + EN] = n;
        cpu_para[i*NUM_PARAMS + H] = h;
        cpu_para[i*NUM_PARAMS + R1] = r1;
        cpu_para[i*NUM_PARAMS + R2] = r2;
        cpu_para[i*NUM_PARAMS + PI_FACTOR] = PI / (r2 - r1);
        cpu_para[i*NUM_PARAMS + MINUS_HALF_OVER_N] = - 0.5 / n;
        rc = r2 > rc ? r2 : rc;
    }

    int num_of_neighbors = (atom->neighbor.MN < 50) ? atom->neighbor.MN : 50;
    int memory1 = sizeof(double)* atom->N * num_of_neighbors;
    int memory2 = sizeof(double)* n_entries * NUM_PARAMS;
    CHECK(cudaMalloc((void**)&tersoff_mini_data.b,    memory1));
    CHECK(cudaMalloc((void**)&tersoff_mini_data.bp,   memory1));
    CHECK(cudaMalloc((void**)&tersoff_mini_data.f12x, memory1));
    CHECK(cudaMalloc((void**)&tersoff_mini_data.f12y, memory1));
    CHECK(cudaMalloc((void**)&tersoff_mini_data.f12z, memory1));
    CHECK(cudaMalloc((void**)&para, memory2));
    CHECK(cudaMemcpy(para, cpu_para, memory2, cudaMemcpyHostToDevice));
    MY_FREE(cpu_para);
}


Tersoff_mini::~Tersoff_mini(void)
{
    CHECK(cudaFree(tersoff_mini_data.b));
    CHECK(cudaFree(tersoff_mini_data.bp));
    CHECK(cudaFree(tersoff_mini_data.f12x));
    CHECK(cudaFree(tersoff_mini_data.f12y));
    CHECK(cudaFree(tersoff_mini_data.f12z));
    CHECK(cudaFree(para));
}


static __device__ void find_fr_and_frp
(int i, const double* __restrict__ para, double d12, double &fr, double &frp)
{
    fr = LDG(para, i + A) * exp(- LDG(para, i + LAMBDA) * d12);
    frp = - LDG(para, i + LAMBDA) * fr;
}


static __device__ void find_fa_and_fap
(int i, const double* __restrict__ para, double d12, double &fa, double &fap)
{
    fa  = LDG(para, i + B) * exp(- LDG(para, i + MU) * d12);
    fap = - LDG(para, i + MU) * fa;
}


static __device__ void find_fa
(int i, const double* __restrict__ para, double d12, double &fa)
{
    fa = LDG(para, i + B) * exp(- LDG(para, i + MU) * d12);
}


static __device__ void find_fc_and_fcp
(int i, const double* __restrict__ para, double d12, double &fc, double &fcp)
{
    if (d12 < LDG(para, i + R1)){fc = 1.0; fcp = 0.0;}
    else if (d12 < LDG(para, i + R2))
    {
        fc = 0.5 * cos(LDG(para, i + PI_FACTOR) * (d12 - LDG(para, i + R1)))
           + 0.5;

        fcp = - sin(LDG(para, i + PI_FACTOR) * (d12 - LDG(para, i + R1))) 
            * LDG(para, i + PI_FACTOR) * 0.5;
    }
    else {fc  = 0.0; fcp = 0.0;}
}


static __device__ void find_fc
(int i, const double* __restrict__ para, double d12, double &fc)
{
    if (d12 < LDG(para, i + R1)) {fc  = 1.0;}
    else if (d12 < LDG(para, i + R2))
    {
        fc = 0.5 * cos(LDG(para, i + PI_FACTOR) * (d12 - LDG(para, i + R1)))
           + 0.5;
    }
    else {fc  = 0.0;}
}


static __device__ void find_g_and_gp
(int i, const double* __restrict__ para, double cos, double &g, double &gp)
{
    double tmp = cos - LDG(para, i + H);
    g  = tmp * tmp;
    gp = 2.0 * tmp;
}


static __device__ void find_g
(int i, const double* __restrict__ para, double cos, double &g)
{
    double tmp = cos - LDG(para, i + H);
    g = tmp * tmp;
}


// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_step1
(
    int number_of_particles, int N1, int N2, Box box,
    int num_types, int* g_neighbor_number, int* g_neighbor_list,
    int* g_type, int shift,
    const double* __restrict__ para,
    const double* __restrict__ g_x,
    const double* __restrict__ g_y,
    const double* __restrict__ g_z,
    double* g_b, double* g_bp
)
{
    // start from the N1-th atom
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1] - shift;
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int type2 = g_type[n2] - shift;
            double x12  = LDG(g_x, n2) - x1;
            double y12  = LDG(g_y, n2) - y1;
            double z12  = LDG(g_z, n2) - z1;
            dev_apply_mic(box, x12, y12, z12);
            double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            double zeta = 0.0;
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
                double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12*d13);
                double fc13, g123;
                int type13 = (type1 + type3) * NUM_PARAMS;
                if (d13 > LDG(para, type13 + R2)) {continue;}
                find_fc(type13, para, d13, fc13);
                find_g(type13, para, cos123, g123);
                zeta += fc13 * g123;
            }

            double bzn, b12;
            int type12 = (type1 + type2) * NUM_PARAMS;
            bzn = pow(LDG(para, type12 + BETA) * zeta, LDG(para, type12 + EN));
            b12 = pow(1.0 + bzn, LDG(para, type12 + MINUS_HALF_OVER_N));
            if (zeta < 1.0e-16) // avoid division by 0
            {
                g_b[i1 * number_of_particles + n1]  = 1.0;
                g_bp[i1 * number_of_particles + n1] = 0.0;
            }
            else
            {
                g_b[i1 * number_of_particles + n1]  = b12;
                g_bp[i1 * number_of_particles + n1]
                    = - b12 * bzn * 0.5 / ((1.0 + bzn) * zeta);
            }
        }
    }
}


// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void find_force_step2
(
    int number_of_particles, int N1, int N2, Box box,
    int num_types, int *g_neighbor_number, int *g_neighbor_list,
    int *g_type, int shift,
    const double* __restrict__ para,
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
    // to the (N2-1)-th atom
    if (n1 >= N1 && n1 < N2)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1] - shift;
        double x1 = LDG(g_x, n1); 
        double y1 = LDG(g_y, n1); 
        double z1 = LDG(g_z, n1);
        double pot_energy = 0.0;
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
            double fc12, fcp12, fa12, fap12, fr12, frp12;
            int type12 = type1 + type2;
            find_fc_and_fcp(type12, para, d12, fc12, fcp12);
            find_fa_and_fap(type12, para, d12, fa12, fap12);
            find_fr_and_frp(type12, para, d12, fr12, frp12);

            // (i,j) part
            double b12 = LDG(g_b, index);
            double factor3 = 
            (
                fcp12 * (fr12 - b12 * fa12) + fc12 * (frp12-b12 * fap12)
            ) * d12inv;
            double f12x = x12 * factor3 * 0.5;
            double f12y = y12 * factor3 * 0.5;
            double f12z = z12 * factor3 * 0.5;

            // accumulate potential energy
            pot_energy += fc12 * (fr12 - b12 * fa12) * 0.5;

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
                double fc13, fa13;
                int type13 = type1 + type3;
                find_fc(type13, para, d13, fc13);
                find_fa(type13, para, d13, fa13);
                double bp13 = LDG(g_bp, index_2);
                double one_over_d12d13 = ONE / (d12 * d13);
                double cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                double cos123_over_d12d12 = cos123*d12inv*d12inv;
                double g123, gp123;
                find_g_and_gp(type13, para, cos123, g123, gp123);
                // derivatives with cosine
                double dc = -fc12 * bp12 * fa12 * fc13 * gp123 
                            -fc12 * bp13 * fa13 * fc13 * gp123;
                // derivatives with rij
                double dr = -fcp12 * bp13 * fa13 * g123 * fc13 * d12inv;
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


// Wrapper of force evaluation for the SBOP potential
void Tersoff_mini::compute(Atom *atom, Measure *measure, int potential_number)
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

    // special data for SBOP potential
    double *f12x = tersoff_mini_data.f12x;
    double *f12y = tersoff_mini_data.f12y;
    double *f12z = tersoff_mini_data.f12z;
    double *b    = tersoff_mini_data.b;
    double *bp   = tersoff_mini_data.bp;

    // pre-compute the bond order functions and their derivatives
    find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, num_types,
        NN, NL, type, shift, para, x, y, z, b, bp
    );
    CUDA_CHECK_KERNEL

    // pre-compute the partial forces
    find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>
    (
        N, N1, N2, atom->box, num_types,
        NN, NL, type, shift, para, b, bp, x, y, z, pe, f12x, f12y, f12z
    );
    CUDA_CHECK_KERNEL

    // the final step: calculate force and related quantities
    find_properties_many_body(atom, measure, NN, NL, f12x, f12y, f12z);
}

