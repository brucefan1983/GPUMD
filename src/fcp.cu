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
The force constant potential (FCP)
TODO:
    *) atomicAdd => shared memory <<<128, 1024>>> and summation <<<4N, 128>>>
    *) higher order (5 and 6)
    *) heat current?
    *) stress?
------------------------------------------------------------------------------*/


#include "fcp.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "error.cuh"


FCP::FCP(FILE* fid, char *input_dir, Atom *atom)
{
    printf("Use the force constant potential.\n");
    int count = fscanf(fid, "%d", &order);
    if (count != 1) 
    { print_error("reading error for force constant potential\n"); }
    CHECK(cudaMalloc(&fcp_data.u, sizeof(float) * atom->N * 3));
    CHECK(cudaMalloc(&fcp_data.r0, sizeof(float) * atom->N * 3));
    CHECK(cudaMalloc(&fcp_data.pf, sizeof(float) * atom->N * 4));
    read_r0(input_dir, atom);
    read_fc2(input_dir, atom);
    read_fc3(input_dir, atom);
    read_fc4(input_dir, atom);
}


FCP::~FCP(void)
{
    CHECK(cudaFree(fcp_data.u));
    CHECK(cudaFree(fcp_data.r0));
    CHECK(cudaFree(fcp_data.pf));
    CHECK(cudaFree(fcp_data.ia2));
    CHECK(cudaFree(fcp_data.jb2));
    CHECK(cudaFree(fcp_data.phi2));
    CHECK(cudaFree(fcp_data.ia3));
    CHECK(cudaFree(fcp_data.jb3));
    CHECK(cudaFree(fcp_data.kc3));
    CHECK(cudaFree(fcp_data.phi3));
    CHECK(cudaFree(fcp_data.ia4));
    CHECK(cudaFree(fcp_data.jb4));
    CHECK(cudaFree(fcp_data.kc4));
    CHECK(cudaFree(fcp_data.ld4));
    CHECK(cudaFree(fcp_data.phi4));
}


void FCP::read_r0(char *input_dir, Atom *atom)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/r0.in");
    FILE *fid = my_fopen(file, "r");

    int N = atom->N;
    float *r0;
    MY_MALLOC(r0, float, N * 3);

    for (int n = 0; n < N; n++)
    {
        int count = fscanf
        (
            fid, "%f%f%f", &r0[n], &r0[n + N], &r0[n + N + N]
        );
        if (count != 3) { print_error("reading error for r0.in\n"); }
    }
    fclose(fid);

    CHECK(cudaMemcpy(fcp_data.r0, r0, sizeof(float) * N * 3, 
        cudaMemcpyHostToDevice));
    MY_FREE(r0);
    printf("    Data in r0.in has been read in.\n");
}


void FCP::read_fc2(char *input_dir, Atom *atom)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/fc2.in");
    FILE *fid = my_fopen(file, "r");

    int count = fscanf(fid, "%d", &number2);
    if (count != 1) { print_error("reading error for fc2.in\n"); }

    int *ia2, *jb2;
    float *phi2;
    MY_MALLOC(ia2, int, number2);
    MY_MALLOC(jb2, int, number2);
    MY_MALLOC(phi2, float, number2);
    CHECK(cudaMalloc(&fcp_data.ia2, sizeof(int) * number2));
    CHECK(cudaMalloc(&fcp_data.jb2, sizeof(int) * number2));
    CHECK(cudaMalloc(&fcp_data.phi2, sizeof(float) * number2));

    for (int n = 0; n < number2; n++)
    {
        int i, j, a, b;
        count = fscanf
        (
            fid, "%d%d%d%d%f", &i, &j, &a, &b, &phi2[n]
        );
        if (count != 5) { print_error("reading error for fc2.in\n"); }
        ia2[n] = a * atom->N + i;
        jb2[n] = b * atom->N + j;
        if (i == j) { phi2[n] /= 2; } // 11
    }
    fclose(fid);

    CHECK(cudaMemcpy(fcp_data.ia2, ia2, sizeof(int) * number2, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.jb2, jb2, sizeof(int) * number2, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.phi2, phi2, sizeof(float) * number2, 
        cudaMemcpyHostToDevice));
    MY_FREE(ia2);
    MY_FREE(jb2);
    MY_FREE(phi2);
    printf("    Data in fc2.in (%d entries) has been read in.\n", number2);
}


void FCP::read_fc3(char *input_dir, Atom *atom)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/fc3.in");
    FILE *fid = my_fopen(file, "r");

    int count = fscanf(fid, "%d", &number3);
    if (count != 1) { print_error("reading error for fc3.in\n"); }

    int *ia3, *jb3, *kc3;
    float *phi3;
    MY_MALLOC(ia3, int, number3);
    MY_MALLOC(jb3, int, number3);
    MY_MALLOC(kc3, int, number3);
    MY_MALLOC(phi3, float, number3);
    CHECK(cudaMalloc(&fcp_data.ia3, sizeof(int) * number3));
    CHECK(cudaMalloc(&fcp_data.jb3, sizeof(int) * number3));
    CHECK(cudaMalloc(&fcp_data.kc3, sizeof(int) * number3));
    CHECK(cudaMalloc(&fcp_data.phi3, sizeof(float) * number3));

    for (int n = 0; n < number3; n++)
    {
        int i, j, k, a, b, c;
        count = fscanf
        (
            fid, "%d%d%d%d%d%d%f", &i, &j, &k, &a, &b, &c, &phi3[n]
        );
        if (count != 7) { print_error("reading error for fc3.in\n"); }
        ia3[n] = a * atom->N + i;
        jb3[n] = b * atom->N + j;
        kc3[n] = c * atom->N + k;
        if (i == j && j != k) { phi3[n] /= 2; } // 112
        if (i != j && j == k) { phi3[n] /= 2; } // 122
        if (i == j && j == k) { phi3[n] /= 6; } // 111
    }
    fclose(fid);

    CHECK(cudaMemcpy(fcp_data.ia3, ia3, sizeof(int) * number3, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.jb3, jb3, sizeof(int) * number3, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.kc3, kc3, sizeof(int) * number3, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.phi3, phi3, sizeof(float) * number3, 
        cudaMemcpyHostToDevice));
    MY_FREE(ia3);
    MY_FREE(jb3);
    MY_FREE(kc3);
    MY_FREE(phi3);
    printf("    Data in fc3.in (%d entries) has been read in.\n", number3);
}


void FCP::read_fc4(char *input_dir, Atom *atom)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/fc4.in");
    FILE *fid = my_fopen(file, "r");

    int count = fscanf(fid, "%d", &number4);
    if (count != 1) { print_error("reading error for fc4.in\n"); }

    int *ia4, *jb4, *kc4, *ld4;
    float *phi4;
    MY_MALLOC(ia4, int, number4);
    MY_MALLOC(jb4, int, number4);
    MY_MALLOC(kc4, int, number4);
    MY_MALLOC(ld4, int, number4);
    MY_MALLOC(phi4, float, number4);
    CHECK(cudaMalloc(&fcp_data.ia4, sizeof(int) * number4));
    CHECK(cudaMalloc(&fcp_data.jb4, sizeof(int) * number4));
    CHECK(cudaMalloc(&fcp_data.kc4, sizeof(int) * number4));
    CHECK(cudaMalloc(&fcp_data.ld4, sizeof(int) * number4));
    CHECK(cudaMalloc(&fcp_data.phi4, sizeof(float) * number4));

    for (int n = 0; n < number4; n++)
    {
        int i, j, k, l, a, b, c, d;
        count = fscanf
        (
            fid, "%d%d%d%d%d%d%d%d%f", &i, &j, &k, &l, &a, &b, &c, &d, &phi4[n]
        );
        if (count != 9) { print_error("reading error for fc4.in\n"); }
        ia4[n] = a * atom->N + i;
        jb4[n] = b * atom->N + j;
        kc4[n] = c * atom->N + k;
        ld4[n] = d * atom->N + l;
        
        if (i == j && j != k && k != l) { phi4[n] /= 2; } // 1123
        if (i != j && j == k && k != l) { phi4[n] /= 2; } // 1223
        if (i != j && j != k && k == l) { phi4[n] /= 2; } // 1233
        if (i == j && j != k && k == l) { phi4[n] /= 4; } // 1122
        if (i == j && j == k && k != l) { phi4[n] /= 6; } // 1112
        if (i != j && j == k && k == l) { phi4[n] /= 6; } // 1222
        if (i == j && j == k && k == l) { phi4[n] /= 24; } // 1111
    }
    fclose(fid);

    CHECK(cudaMemcpy(fcp_data.ia4, ia4, sizeof(int) * number4, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.jb4, jb4, sizeof(int) * number4, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.kc4, kc4, sizeof(int) * number4, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.ld4, ld4, sizeof(int) * number4, 
        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(fcp_data.phi4, phi4, sizeof(float) * number4, 
        cudaMemcpyHostToDevice));
    MY_FREE(ia4);
    MY_FREE(jb4);
    MY_FREE(kc4);
    MY_FREE(ld4);
    MY_FREE(phi4);
    printf("    Data in fc4.in (%d entries) has been read in.\n", number4);
}


// potential and force from the second-order force constants
static __global__ void gpu_find_force_fcp2
(
    int N, int number2, int *g_ia2, int *g_jb2, float *g_phi2,
    const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number2)
    {
        int ia = g_ia2[n]; 
        int jb = g_jb2[n];
        float phi = g_phi2[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb);
        atomicAdd(&g_pf[ia + N], - phi * ujb);
        atomicAdd(&g_pf[jb + N], - phi * uia);
    }
}


// potential and force from the third-order force constants
static __global__ void gpu_find_force_fcp3
(
    int N, int number3, int *g_ia3, int *g_jb3, int *g_kc3, float *g_phi3,
    const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number3)
    {
        int ia = g_ia3[n];
        int jb = g_jb3[n]; 
        int kc = g_kc3[n]; 
        float phi = g_phi3[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb);
    }
}


// potential and force from the fourth-order force constants
static __global__ void gpu_find_force_fcp4
(
    int N, int number4, int *g_ia4, int *g_jb4, int *g_kc4, int *g_ld4,
    float *g_phi4, const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number4)
    {
        int ia = g_ia4[n];
        int jb = g_jb4[n]; 
        int kc = g_kc4[n];
        int ld = g_ld4[n];
        float phi = g_phi4[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        float uld = LDG(g_u, ld);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc * uld);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc * uld);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc * uld);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb * uld);
        atomicAdd(&g_pf[ld + N], - phi * uia * ujb * ukc);
    }
}


// initialize the local potential and force
static __global__ void gpu_initialize_pf(int N, float *pf)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        pf[n] = 0.0;
        pf[n + N] = 0.0;
        pf[n + N * 2] = 0.0;
        pf[n + N * 3] = 0.0;
    }
}


// change position from global to local
static __global__ void gpu_get_u
(
    int N, double *x, double *y, double *z, float *r0, float *u
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        u[n] = x[n] - r0[n];
        u[n + N] = y[n] - r0[n + N];
        u[n + N + N] = z[n] - r0[n + N + N];
    }
}


// change potential and force from local to global
static __global__ void gpu_save_pf
(
    int N, float *pf, double *p, double *fx, double *fy, double *fz
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        p[n] = pf[n];
        fx[n] = pf[n + N];
        fy[n] = pf[n + N * 2];
        fz[n] = pf[n + N * 3];
    }
}


// Wrapper of the above kernels
void FCP::compute(Atom *atom, Measure *measure, int potential_number)
{
    const int block_size = 1024;

    gpu_get_u<<<(atom->N - 1) / block_size + 1, block_size>>>
    (atom->N, atom->x, atom->y, atom->z, fcp_data.r0, fcp_data.u);

    gpu_initialize_pf<<<(atom->N - 1) / block_size + 1, block_size>>>
    (atom->N, fcp_data.pf);

    gpu_find_force_fcp2<<<(number2 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number2, fcp_data.ia2, fcp_data.jb2, fcp_data.phi2,
        fcp_data.u, fcp_data.pf
    );

    if (order >= 3)
    gpu_find_force_fcp3<<<(number3 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number3, fcp_data.ia3, fcp_data.jb3, fcp_data.kc3,
        fcp_data.phi3, fcp_data.u, fcp_data.pf
    );

    if (order >= 4)
    gpu_find_force_fcp4<<<(number4 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number4, fcp_data.ia4, fcp_data.jb4, fcp_data.kc4,
        fcp_data.ld4, fcp_data.phi4, fcp_data.u, fcp_data.pf
    );

    gpu_save_pf<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N, fcp_data.pf, atom->potential_per_atom,
        atom->fx, atom->fy, atom->fz
    );

    CUDA_CHECK_KERNEL
}



