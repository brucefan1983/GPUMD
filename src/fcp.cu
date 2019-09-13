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
TODO: HNEMD
------------------------------------------------------------------------------*/


#include "fcp.cuh"
#include "measure.cuh"
#include "atom.cuh"
#include "mic.cuh"
#include "error.cuh"


FCP::FCP(FILE* fid, char *input_dir, Atom *atom)
{
    printf("Use the force constant potential.\n");
    int count = fscanf(fid, "%d", &order);
    if (count != 1) 
    { print_error("reading error for force constant potential\n"); }
    CHECK(cudaMalloc(&fcp_data.uv, sizeof(float) * atom->N * 6));
    CHECK(cudaMallocManaged(&fcp_data.r0, sizeof(float) * atom->N * 3));
    CHECK(cudaMalloc(&fcp_data.pfj, sizeof(float) * atom->N * 7));
    read_r0(input_dir, atom);
    read_fc2(input_dir, atom);
    if (order >= 3) read_fc3(input_dir, atom);
    if (order >= 4) read_fc4(input_dir, atom);
    if (order >= 5) read_fc5(input_dir, atom);
    if (order >= 6) read_fc6(input_dir, atom);
}


FCP::~FCP(void)
{
    CHECK(cudaFree(fcp_data.uv));
    CHECK(cudaFree(fcp_data.r0));
    CHECK(cudaFree(fcp_data.pfj));
    CHECK(cudaFree(fcp_data.ia2));
    CHECK(cudaFree(fcp_data.jb2));
    CHECK(cudaFree(fcp_data.phi2));
    CHECK(cudaFree(fcp_data.xij2));
    CHECK(cudaFree(fcp_data.yij2));
    CHECK(cudaFree(fcp_data.zij2));
    if (order >= 3)
    {
        CHECK(cudaFree(fcp_data.ia3));
        CHECK(cudaFree(fcp_data.jb3));
        CHECK(cudaFree(fcp_data.kc3));
        CHECK(cudaFree(fcp_data.phi3));
    }
    if (order >= 4)
    {
        CHECK(cudaFree(fcp_data.ia4));
        CHECK(cudaFree(fcp_data.jb4));
        CHECK(cudaFree(fcp_data.kc4));
        CHECK(cudaFree(fcp_data.ld4));
        CHECK(cudaFree(fcp_data.phi4));
    }
    if (order >= 5)
    {
        CHECK(cudaFree(fcp_data.ia5));
        CHECK(cudaFree(fcp_data.jb5));
        CHECK(cudaFree(fcp_data.kc5));
        CHECK(cudaFree(fcp_data.ld5));
        CHECK(cudaFree(fcp_data.me5));
        CHECK(cudaFree(fcp_data.phi5));
    }
    if (order >= 6)
    {
        CHECK(cudaFree(fcp_data.ia6));
        CHECK(cudaFree(fcp_data.jb6));
        CHECK(cudaFree(fcp_data.kc6));
        CHECK(cudaFree(fcp_data.ld6));
        CHECK(cudaFree(fcp_data.me6));
        CHECK(cudaFree(fcp_data.nf6));
        CHECK(cudaFree(fcp_data.phi6));
    }
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

    CHECK(cudaMallocManaged(&fcp_data.ia2, sizeof(int) * number2));
    CHECK(cudaMallocManaged(&fcp_data.jb2, sizeof(int) * number2));
    CHECK(cudaMallocManaged(&fcp_data.phi2, sizeof(float) * number2));
    CHECK(cudaMallocManaged(&fcp_data.xij2, sizeof(float) * number2));
    CHECK(cudaMallocManaged(&fcp_data.yij2, sizeof(float) * number2));
    CHECK(cudaMallocManaged(&fcp_data.zij2, sizeof(float) * number2));

    for (int index = 0; index < number2; index++)
    {
        int i, j, a, b;
        count = fscanf
        (
            fid, "%d%d%d%d%f", &i, &j, &a, &b, &fcp_data.phi2[index]
        );
        if (count != 5) { print_error("reading error for fc2.in\n"); }
        fcp_data.ia2[index] = a * atom->N + i;
        fcp_data.jb2[index] = b * atom->N + j;

        // 2^1-1 = 1 case:
        if (i == j) { fcp_data.phi2[index] /= 2; } // 11
        
        double xij2 = fcp_data.r0[j] - fcp_data.r0[i];
        double yij2 = fcp_data.r0[j] - fcp_data.r0[i];
        double zij2 = fcp_data.r0[j] - fcp_data.r0[i];
        apply_mic
        (
            atom->box.triclinic, atom->box.pbc_x, atom->box.pbc_y, 
            atom->box.pbc_z, atom->box.cpu_h, xij2, yij2, zij2
        );
        fcp_data.xij2[index] = xij2 * 0.5;
        fcp_data.yij2[index] = yij2 * 0.5;
        fcp_data.zij2[index] = zij2 * 0.5;
    }

    fclose(fid);
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

    CHECK(cudaMallocManaged(&fcp_data.ia3, sizeof(int) * number3));
    CHECK(cudaMallocManaged(&fcp_data.jb3, sizeof(int) * number3));
    CHECK(cudaMallocManaged(&fcp_data.kc3, sizeof(int) * number3));
    CHECK(cudaMallocManaged(&fcp_data.phi3, sizeof(float) * number3));

    for (int index = 0; index < number3; index++)
    {
        int i, j, k, a, b, c;
        count = fscanf
        (
            fid, "%d%d%d%d%d%d%f", &i, &j, &k, &a, &b, &c, 
            &fcp_data.phi3[index]
        );
        if (count != 7) { print_error("reading error for fc3.in\n"); }
        fcp_data.ia3[index] = a * atom->N + i;
        fcp_data.jb3[index] = b * atom->N + j;
        fcp_data.kc3[index] = c * atom->N + k;

        // 2^2-1 = 3 cases:
        if (i == j && j != k) { fcp_data.phi3[index] /= 2; } // 112
        if (i != j && j == k) { fcp_data.phi3[index] /= 2; } // 122
        if (i == j && j == k) { fcp_data.phi3[index] /= 6; } // 111
    }

    fclose(fid);
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

    CHECK(cudaMallocManaged(&fcp_data.ia4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.jb4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.kc4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.ld4, sizeof(int) * number4));
    CHECK(cudaMallocManaged(&fcp_data.phi4, sizeof(float) * number4));

    for (int index = 0; index < number4; index++)
    {
        int i, j, k, l, a, b, c, d;
        count = fscanf
        (
            fid, "%d%d%d%d%d%d%d%d%f", &i, &j, &k, &l, &a, &b, &c, &d, 
            &fcp_data.phi4[index]
        );
        if (count != 9) { print_error("reading error for fc4.in\n"); }
        fcp_data.ia4[index] = a * atom->N + i;
        fcp_data.jb4[index] = b * atom->N + j;
        fcp_data.kc4[index] = c * atom->N + k;
        fcp_data.ld4[index] = d * atom->N + l;
        
        // 2^3-1 = 7 cases:
        if (i == j && j != k && k != l) { fcp_data.phi4[index] /= 2; }  // 1123
        if (i != j && j == k && k != l) { fcp_data.phi4[index] /= 2; }  // 1223
        if (i != j && j != k && k == l) { fcp_data.phi4[index] /= 2; }  // 1233
        if (i == j && j != k && k == l) { fcp_data.phi4[index] /= 4; }  // 1122
        if (i == j && j == k && k != l) { fcp_data.phi4[index] /= 6; }  // 1112
        if (i != j && j == k && k == l) { fcp_data.phi4[index] /= 6; }  // 1222
        if (i == j && j == k && k == l) { fcp_data.phi4[index] /= 24; } // 1111
    }

    fclose(fid);
    printf("    Data in fc4.in (%d entries) has been read in.\n", number4);
}


void FCP::read_fc5(char *input_dir, Atom *atom)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/fc5.in");
    FILE *fid = my_fopen(file, "r");

    int count = fscanf(fid, "%d", &number5);
    if (count != 1) { print_error("reading error for fc5.in\n"); }

    CHECK(cudaMallocManaged(&fcp_data.ia5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.jb5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.kc5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.ld5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.me5, sizeof(int) * number5));
    CHECK(cudaMallocManaged(&fcp_data.phi5, sizeof(float) * number5));

    for (int index = 0; index < number5; index++)
    {
        int i, j, k, l, m, a, b, c, d, e;
        count = fscanf
        (
            fid, "%d%d%d%d%d%d%d%d%d%d%f", &i, &j, &k, &l, &m, 
            &a, &b, &c, &d, &e, &fcp_data.phi5[index]
        );
        if (count != 11) { print_error("reading error for fc5.in\n"); }
        fcp_data.ia5[index] = a * atom->N + i;
        fcp_data.jb5[index] = b * atom->N + j;
        fcp_data.kc5[index] = c * atom->N + k;
        fcp_data.ld5[index] = d * atom->N + l;
        fcp_data.me5[index] = e * atom->N + m;
        
        // 2^4-1 = 15 cases:
        if (i == j && j != k && k != l && l != m) { fcp_data.phi5[index] /= 2; }   // 11234
        if (i != j && j == k && k != l && l != m) { fcp_data.phi5[index] /= 2; }   // 12234
        if (i != j && j != k && k == l && l != m) { fcp_data.phi5[index] /= 2; }   // 12334
        if (i != j && j != k && k != l && l == m) { fcp_data.phi5[index] /= 2; }   // 12344
        if (i == j && j == k && k != l && l != m) { fcp_data.phi5[index] /= 6; }   // 11123
        if (i != j && j == k && k == l && l != m) { fcp_data.phi5[index] /= 6; }   // 12223
        if (i != j && j != k && k == l && l == m) { fcp_data.phi5[index] /= 6; }   // 12333
        if (i == j && j == k && k == l && l != m) { fcp_data.phi5[index] /= 24; }  // 11112
        if (i != j && j == k && k == l && l == m) { fcp_data.phi5[index] /= 24; }  // 12222
        if (i == j && j == k && k == l && l == m) { fcp_data.phi5[index] /= 120; } // 11111
        if (i == j && j != k && k == l && l != m) { fcp_data.phi5[index] /= 4; }   // 11223
        if (i == j && j != k && k != l && l == m) { fcp_data.phi5[index] /= 4; }   // 11233
        if (i != j && j == k && k != l && l == m) { fcp_data.phi5[index] /= 4; }   // 12233
        if (i == j && j != k && k == l && l == m) { fcp_data.phi5[index] /= 12; }  // 11222
        if (i == j && j == k && k != l && l == m) { fcp_data.phi5[index] /= 12; }  // 11122
    }

    fclose(fid);
    printf("    Data in fc5.in (%d entries) has been read in.\n", number5);
}


void FCP::read_fc6(char *input_dir, Atom *atom)
{
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/fc6.in");
    FILE *fid = my_fopen(file, "r");

    int count = fscanf(fid, "%d", &number6);
    if (count != 1) { print_error("reading error for fc6.in\n"); }

    CHECK(cudaMallocManaged(&fcp_data.ia6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.jb6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.kc6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.ld6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.me6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.nf6, sizeof(int) * number6));
    CHECK(cudaMallocManaged(&fcp_data.phi6, sizeof(float) * number6));

    for (int index = 0; index < number6; index++)
    {
        int i, j, k, l, m, n, a, b, c, d, e, f;
        count = fscanf
        (
            fid, "%d%d%d%d%d%d%d%d%d%d%d%d%f", &i, &j, &k, &l, &m, &n,
            &a, &b, &c, &d, &e, &f, &fcp_data.phi6[index]
        );
        if (count != 13) { print_error("reading error for fc6.in\n"); }
        fcp_data.ia6[index] = a * atom->N + i;
        fcp_data.jb6[index] = b * atom->N + j;
        fcp_data.kc6[index] = c * atom->N + k;
        fcp_data.ld6[index] = d * atom->N + l;
        fcp_data.me6[index] = e * atom->N + m;
        fcp_data.nf6[index] = f * atom->N + n;
        
        // 2^5-1 = 31 cases:
        if (i == j && j != k && k != l && l != m && m != n) { fcp_data.phi6[index] /= 2; }   // 112345
        if (i != j && j == k && k != l && l != m && m != n) { fcp_data.phi6[index] /= 2; }   // 122345
        if (i != j && j != k && k == l && l != m && m != n) { fcp_data.phi6[index] /= 2; }   // 123345
        if (i != j && j != k && k != l && l == m && m != n) { fcp_data.phi6[index] /= 2; }   // 123445
        if (i != j && j != k && k != l && l != m && m == n) { fcp_data.phi6[index] /= 2; }   // 123455
        if (i == j && j == k && k != l && l != m && m != n) { fcp_data.phi6[index] /= 6; }   // 111234
        if (i != j && j == k && k == l && l != m && m != n) { fcp_data.phi6[index] /= 6; }   // 122234
        if (i != j && j != k && k == l && l == m && m != n) { fcp_data.phi6[index] /= 6; }   // 123334
        if (i != j && j != k && k != l && l == m && m == n) { fcp_data.phi6[index] /= 6; }   // 123444
        if (i == j && j == k && k == l && l != m && m != n) { fcp_data.phi6[index] /= 24; }  // 111123
        if (i != j && j == k && k == l && l == m && m != n) { fcp_data.phi6[index] /= 24; }  // 122223
        if (i != j && j != k && k == l && l == m && m == n) { fcp_data.phi6[index] /= 24; }  // 122223
        if (i == j && j == k && k == l && l == m && m != n) { fcp_data.phi6[index] /= 120; } // 111112
        if (i != j && j == k && k == l && l == m && m == n) { fcp_data.phi6[index] /= 120; } // 122222
        if (i == j && j == k && k == l && l == m && m == n) { fcp_data.phi6[index] /= 720; } // 111111
        if (i == j && j != k && k == l && l != m && m != n) { fcp_data.phi6[index] /= 4; }   // 112234
        if (i == j && j != k && k != l && l == m && m != n) { fcp_data.phi6[index] /= 4; }   // 112334
        if (i == j && j != k && k != l && l != m && m == n) { fcp_data.phi6[index] /= 4; }   // 112344
        if (i != j && j == k && k != l && l == m && m != n) { fcp_data.phi6[index] /= 4; }   // 122334
        if (i != j && j == k && k != l && l != m && m == n) { fcp_data.phi6[index] /= 4; }   // 122344
        if (i != j && j != k && k == l && l != m && m == n) { fcp_data.phi6[index] /= 4; }   // 123344
        if (i == j && j != k && k == l && l == m && m != n) { fcp_data.phi6[index] /= 12; }  // 112223
        if (i == j && j != k && k != l && l == m && m == n) { fcp_data.phi6[index] /= 12; }  // 112333
        if (i != j && j == k && k != l && l == m && m == n) { fcp_data.phi6[index] /= 12; }  // 122333
        if (i == j && j == k && k != l && l == m && m != n) { fcp_data.phi6[index] /= 12; }  // 111223
        if (i == j && j == k && k != l && l != m && m == n) { fcp_data.phi6[index] /= 12; }  // 111233
        if (i != j && j == k && k == l && l != m && m == n) { fcp_data.phi6[index] /= 12; }  // 122233
        if (i == j && j != k && k == l && l == m && m == n) { fcp_data.phi6[index] /= 48; }  // 112222
        if (i == j && j == k && k == l && l != m && m == n) { fcp_data.phi6[index] /= 48; }  // 111122
        if (i == j && j == k && k != l && l == m && m == n) { fcp_data.phi6[index] /= 36; }  // 111222
        if (i == j && j != k && k == l && l != m && m == n) { fcp_data.phi6[index] /= 8; }   // 112233
    }

    fclose(fid);
    printf("    Data in fc6.in (%d entries) has been read in.\n", number6);
}


// potential and force from the second-order force constants
static __global__ void gpu_find_force_fcp2
(
    int N, int number2, int *g_ia2, int *g_jb2, float *g_phi2,
    const float* __restrict__ g_uv, 
    float *g_xij2, float *g_yij2, float *g_zij2, float *g_pfj
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number2)
    {
        int ia = g_ia2[n]; 
        int jb = g_jb2[n];
        float phi = g_phi2[n];
        float xij2 = g_xij2[n];
        float yij2 = g_yij2[n];
        float zij2 = g_zij2[n];

        float uia = LDG(g_uv, ia); 
        float ujb = LDG(g_uv, jb);
        float via = LDG(g_uv, ia + N * 3);
        float vjb = LDG(g_uv, jb + N * 3);

        int atom_id = ia % N;
        atomicAdd(&g_pfj[atom_id], phi * uia * ujb);
        atomicAdd(&g_pfj[ia + N], - phi * ujb);
        atomicAdd(&g_pfj[jb + N], - phi * uia);

        float uvij = via * ujb - uia * vjb;
        atomicAdd(&g_pfj[atom_id + N * 4], phi * xij2 * uvij);
        atomicAdd(&g_pfj[atom_id + N * 5], phi * yij2 * uvij);
        atomicAdd(&g_pfj[atom_id + N * 6], phi * zij2 * uvij);
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


// potential and force from the fourth-order force constants
static __global__ void gpu_find_force_fcp5
(
    int N, int number5, int *g_ia5, int *g_jb5, int *g_kc5, int *g_ld5,
    int *g_me5, float *g_phi5, const float* __restrict__ g_u, float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number5)
    {
        int ia = g_ia5[n];
        int jb = g_jb5[n]; 
        int kc = g_kc5[n];
        int ld = g_ld5[n];
        int me = g_me5[n];
        float phi = g_phi5[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        float uld = LDG(g_u, ld);
        float ume = LDG(g_u, me);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc * uld * ume);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc * uld * ume);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc * uld * ume);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb * uld * ume);
        atomicAdd(&g_pf[ld + N], - phi * uia * ujb * ukc * ume);
        atomicAdd(&g_pf[me + N], - phi * uia * ujb * ukc * uld);
    }
}


// potential and force from the fourth-order force constants
static __global__ void gpu_find_force_fcp6
(
    int N, int number6, int *g_ia6, int *g_jb6, int *g_kc6, int *g_ld6,
    int *g_me6, int *g_nf6, float *g_phi6, const float* __restrict__ g_u, 
    float *g_pf
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < number6)
    {
        int ia = g_ia6[n];
        int jb = g_jb6[n]; 
        int kc = g_kc6[n];
        int ld = g_ld6[n];
        int me = g_me6[n];
        int nf = g_nf6[n];
        float phi = g_phi6[n];
        float uia = LDG(g_u, ia); 
        float ujb = LDG(g_u, jb);
        float ukc = LDG(g_u, kc);
        float uld = LDG(g_u, ld);
        float ume = LDG(g_u, me);
        float unf = LDG(g_u, nf);
        atomicAdd(&g_pf[ia % N], phi * uia * ujb * ukc * uld * ume * unf);
        atomicAdd(&g_pf[ia + N], - phi * ujb * ukc * uld * ume * unf);
        atomicAdd(&g_pf[jb + N], - phi * uia * ukc * uld * ume * unf);
        atomicAdd(&g_pf[kc + N], - phi * uia * ujb * uld * ume * unf);
        atomicAdd(&g_pf[ld + N], - phi * uia * ujb * ukc * ume * unf);
        atomicAdd(&g_pf[me + N], - phi * uia * ujb * ukc * uld * unf);
        atomicAdd(&g_pf[nf + N], - phi * uia * ujb * ukc * uld * ume);
    }
}


// initialize the local potential (p), force (f), and heat current (j)
static __global__ void gpu_initialize_pfj(int N, float *pfj)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        pfj[n] = 0.0;
        pfj[n + N] = 0.0;
        pfj[n + N * 2] = 0.0;
        pfj[n + N * 3] = 0.0;
        pfj[n + N * 4] = 0.0;
        pfj[n + N * 5] = 0.0;
        pfj[n + N * 6] = 0.0;
    }
}


// get the displacement (u=r-r0) and velocity (v)
static __global__ void gpu_get_uv
(
    int N, double *x, double *y, double *z, 
    double *vx, double *vy, double *vz, float *r0, float *uv
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        uv[n]         = x[n] - r0[n];
        uv[n + N]     = y[n] - r0[n + N];
        uv[n + N * 2] = z[n] - r0[n + N + N];
        uv[n + N * 3] = vx[n];
        uv[n + N * 4] = vy[n];
        uv[n + N * 5] = vz[n];
    }
}


// save potential (p), force (f), and heat current (j)
static __global__ void gpu_save_pfj
(
    int N, float *pfj, double *p, double *fx, double *fy, double *fz, double *j
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N)
    {
        p[n]  = pfj[n];                // potential energy
        fx[n] = pfj[n + N];            // fx
        fy[n] = pfj[n + N * 2];        // fy
        fz[n] = pfj[n + N * 3];        // fz
        j[n]  = pfj[n + N * 4];        // jx_in (jx_out = 0)
        j[n + N * 2] = pfj[n + N * 5]; // jy_in (jy_out = 0)
        j[n + N * 4] = pfj[n + N * 6]; // jz
    }
}


// Wrapper of the above kernels
void FCP::compute(Atom *atom, Measure *measure, int potential_number)
{
    const int block_size = 1024;

    gpu_get_uv<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N, atom->x, atom->y, atom->z, atom->vx, atom->vy, atom->vz, 
        fcp_data.r0, fcp_data.uv
    );

    gpu_initialize_pfj<<<(atom->N - 1) / block_size + 1, block_size>>>
    (atom->N, fcp_data.pfj);

    gpu_find_force_fcp2<<<(number2 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number2, fcp_data.ia2, fcp_data.jb2, fcp_data.phi2,
        fcp_data.uv, fcp_data.xij2, fcp_data.yij2, fcp_data.zij2, fcp_data.pfj
    );

    if (order >= 3)
    gpu_find_force_fcp3<<<(number3 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number3, fcp_data.ia3, fcp_data.jb3, fcp_data.kc3,
        fcp_data.phi3, fcp_data.uv, fcp_data.pfj
    );

    if (order >= 4)
    gpu_find_force_fcp4<<<(number4 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number4, fcp_data.ia4, fcp_data.jb4, fcp_data.kc4,
        fcp_data.ld4, fcp_data.phi4, fcp_data.uv, fcp_data.pfj
    );

    if (order >= 5)
    gpu_find_force_fcp5<<<(number5 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number5, fcp_data.ia5, fcp_data.jb5, fcp_data.kc5,
        fcp_data.ld5, fcp_data.me5, fcp_data.phi5, fcp_data.uv, fcp_data.pfj
    );

    if (order >= 6)
    gpu_find_force_fcp6<<<(number6 - 1) / block_size + 1, block_size>>>
    (
        atom->N, number6, fcp_data.ia6, fcp_data.jb6, fcp_data.kc6,
        fcp_data.ld6, fcp_data.me6, fcp_data.nf6, fcp_data.phi6, 
        fcp_data.uv, fcp_data.pfj
    );

    gpu_save_pfj<<<(atom->N - 1) / block_size + 1, block_size>>>
    (
        atom->N, fcp_data.pfj, atom->potential_per_atom,
        atom->fx, atom->fy, atom->fz, atom->heat_per_atom
    );

    CUDA_CHECK_KERNEL
}



