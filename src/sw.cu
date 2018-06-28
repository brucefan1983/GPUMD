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




#include "common.cuh"
#include "mic.cuh"
#include "sw.cuh"




/*----------------------------------------------------------------------------80
    This file implements the three-element Stillinger-Weber (SW) potential.
------------------------------------------------------------------------------*/




// best block size here: 64 or 128
#define BLOCK_SIZE_SW 64




SW2::SW2(FILE *fid, Parameters *para, int num_of_types)
{
    if (num_of_types == 1) { initialize_sw_1985_1(fid); }
    if (num_of_types == 2) { initialize_sw_1985_2(fid); }
    if (num_of_types == 3) { initialize_sw_1985_3(fid); }

    // memory for the partial forces dU_i/dr_ij
    int num_of_neighbors = (para->neighbor.MN < 20) ? para->neighbor.MN : 20;
    int memory = sizeof(real) * para->N * num_of_neighbors; 
    CHECK(cudaMalloc((void**)&sw2_data.f12x, memory));
    CHECK(cudaMalloc((void**)&sw2_data.f12y, memory));
    CHECK(cudaMalloc((void**)&sw2_data.f12z, memory));
}




void SW2::initialize_sw_1985_1(FILE *fid)
{
    printf("INPUT: use single-element Stillinger-Weber potential.\n");
    int count;

    double epsilon, lambda, A, B, a, gamma, sigma, cos0;

    count = fscanf
    (
        fid, "%lf%lf%lf%lf%lf%lf%lf%lf", 
        &epsilon, &lambda, &A, &B, &a, &gamma, &sigma, &cos0
    );
    if (count!=8) {print_error("reading error for potential.in.\n"); exit(1);}

    sw2_para.A[0][0] = epsilon * A;
    sw2_para.B[0][0] = B;
    sw2_para.a[0][0] = a;
    sw2_para.sigma[0][0] = sigma;
    sw2_para.gamma[0][0] = gamma;
    sw2_para.rc[0][0] = sigma * a;
    rc = sw2_para.rc[0][0];

    sw2_para.lambda[0][0][0] = epsilon * lambda;
    sw2_para.cos0[0][0][0] = cos0;
}




void SW2::initialize_sw_1985_2(FILE *fid)
{
    printf("INPUT: use two-element Stillinger-Weber potential.\n");
    int count;

    // 2-body parameters and the force cutoff
    double A[3], B[3], a[3], sigma[3], gamma[3];
    rc = 0.0;
    for (int n = 0; n < 3; n++)
    {  
        count = fscanf
        (fid, "%lf%lf%lf%lf%lf", &A[n], &B[n], &a[n], &sigma[n], &gamma[n]);
        if (count != 5) print_error("reading error for potential file.\n");
    }
    for (int n1 = 0; n1 < 2; n1++)
    for (int n2 = 0; n2 < 2; n2++)
    {
        sw2_para.A[n1][n2] = A[n1+n2];
        sw2_para.B[n1][n2] = B[n1+n2];
        sw2_para.a[n1][n2] = a[n1+n2];
        sw2_para.sigma[n1][n2] = sigma[n1+n2];
        sw2_para.gamma[n1][n2] = gamma[n1+n2];
        sw2_para.rc[n1][n2] = sigma[n1+n2] * a[n1+n2];
        if (rc < sw2_para.rc[n1][n2]) rc = sw2_para.rc[n1][n2];
    }

    // 3-body parameters
    double lambda, cos0;
    for (int n1 = 0; n1 < 2; n1++)
    for (int n2 = 0; n2 < 2; n2++)
    for (int n3 = 0; n3 < 2; n3++)
    {  
        count = fscanf(fid, "%lf%lf", &lambda, &cos0);
        if (count != 2) print_error("reading error for potential file.\n");
        sw2_para.lambda[n1][n2][n3] = lambda;
        sw2_para.cos0[n1][n2][n3] = cos0;
    }
}  




void SW2::initialize_sw_1985_3(FILE *fid)
{
    printf("INPUT: use three-element Stillinger-Weber potential.\n");
    int count;

    // 2-body parameters and the force cutoff
    double A, B, a, sigma, gamma;
    rc = 0.0;
    for (int n1 = 0; n1 < 3; n1++)
    for (int n2 = 0; n2 < 3; n2++)
    {  
        count = fscanf(fid, "%lf%lf%lf%lf%lf", &A, &B, &a, &sigma, &gamma);
        if (count != 5) print_error("reading error for potential file.\n");
        sw2_para.A[n1][n2] = A;
        sw2_para.B[n1][n2] = B;
        sw2_para.a[n1][n2] = a;
        sw2_para.sigma[n1][n2] = sigma;
        sw2_para.gamma[n1][n2] = gamma;
        sw2_para.rc[n1][n2] = sigma * a;
        if (rc < sw2_para.rc[n1][n2]) rc = sw2_para.rc[n1][n2];
    }

    // 3-body parameters
    double lambda, cos0;
    for (int n1 = 0; n1 < 3; n1++)
    for (int n2 = 0; n2 < 3; n2++)
    for (int n3 = 0; n3 < 3; n3++)
    {  
        count = fscanf
        (fid, "%lf%lf", &lambda, &cos0);
        if (count != 2) print_error("reading error for potential file.\n");
        sw2_para.lambda[n1][n2][n3] = lambda;
        sw2_para.cos0[n1][n2][n3] = cos0;
    }
}




SW2::~SW2(void)
{
    cudaFree(sw2_data.f12x);
    cudaFree(sw2_data.f12y);
    cudaFree(sw2_data.f12z);
}




// two-body part of the SW potential
static __device__ void find_p2_and_f2
(real sigma, real a, real B, real epsilon_times_A, real d12, real &p2, real &f2)
{ 
    real r12 = d12 / sigma;
    real B_over_r12power4 = B / (r12*r12*r12*r12);
    real exp_factor = epsilon_times_A * exp(ONE/(r12 - a));
    p2 = exp_factor * (B_over_r12power4 - ONE); 
    f2 = -p2/((r12-a)*(r12-a))-exp_factor*FOUR*B_over_r12power4/r12;
    f2 /= (sigma * d12); 
}




// find the partial forces dU_i/dr_ij
#ifndef MOS2_JIANG
template <int cal_p>
static __global__ void gpu_find_force_sw3_partial 
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z, SW2_Para sw3,
    int *g_neighbor_number, int *g_neighbor_list, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
#else
    real *g_x,  real *g_y,  real *g_z, 
#endif
    real *g_box_length, 
    real *g_potential, real *g_f12x, real *g_f12y, real *g_f12z 
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real lx = g_box_length[0]; 
        real ly = g_box_length[1]; 
        real lz = g_box_length[2];
        real potential_energy = ZERO;

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
            if (d12 >= sw3.rc[type1][type2]) {continue;} 

            real gamma12 = sw3.gamma[type1][type2];
            real sigma12 = sw3.sigma[type1][type2];
            real a12     = sw3.a[type1][type2];
            real tmp = gamma12 / (sigma12*(d12/sigma12-a12)*(d12/sigma12-a12));
            real p2, f2;
            find_p2_and_f2
            (sigma12, a12, sw3.B[type1][type2], sw3.A[type1][type2], d12, p2, f2);

            // treat the two-body part in the same way as the many-body part
            real f12x = f2 * x12 * HALF; 
            real f12y = f2 * y12 * HALF; 
            real f12z = f2 * z12 * HALF; 
            if (cal_p) // accumulate potential energy
            {
                potential_energy += p2 * HALF;
            }

            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];  
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                if (d13 >= sw3.rc[type1][type3]) {continue;}
                
                real cos0   = sw3.cos0[type1][type2][type3];
                real lambda = sw3.lambda[type1][type2][type3];
                real exp123 = d13 / sw3.sigma[type1][type3] - sw3.a[type1][type3];
                exp123 = sw3.gamma[type1][type3] / exp123;
                exp123 = exp(gamma12 / (d12 / sigma12 - a12) + exp123);
                real one_over_d12d13 = ONE / (d12 * d13);
                real cos123 = (x12*x13 + y12*y13 + z12*z13) * one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv;

                real tmp1 = exp123 * (cos123 - cos0) * lambda;
                real tmp2 = tmp * (cos123 - cos0) * d12inv;

                if (cal_p) // accumulate potential energy
                {
                    potential_energy += (cos123 - cos0) * tmp1 * HALF;
                }

                real cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12; 
                f12x += tmp1 * (TWO * cos_d - tmp2 * x12);

                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += tmp1 * (TWO * cos_d - tmp2 * y12);

                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += tmp1 * (TWO * cos_d - tmp2 * z12);
            }

            g_f12x[index] = f12x;
            g_f12y[index] = f12y;
            g_f12z[index] = f12z;
        }
        if (cal_p) // save potential
        {
            g_potential[n1] = potential_energy;
        }
    }
}    
 



#else // [J.-W. Jiang, Nanotechnology 26, 315706 (2015)]
template <int cal_p>
static __global__ void gpu_find_force_sw3_partial 
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z, SW2_Para sw3,
    int *g_neighbor_number, int *g_neighbor_list, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
#else
    real *g_x,  real *g_y,  real *g_z, 
#endif
    real *g_box_length, 
    real *g_potential, real *g_f12x, real *g_f12y, real *g_f12z 
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real lx = g_box_length[0]; 
        real ly = g_box_length[1]; 
        real lz = g_box_length[2];
        real potential_energy = ZERO;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int index = i1 * number_of_particles + n1;
            int n2 = g_neighbor_list[index];
            int type2 = g_type[n2];
            real x2  = LDG(g_x, n2);
            real y2  = LDG(g_y, n2);
            real z2  = LDG(g_z, n2) ; 
            real x12  = x2 - x1;
            real y12  = y2 - y1;
            real z12  = z2 - z1; 
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real d12inv = ONE / d12;
            if (d12 >= sw3.rc[type1][type2]) {continue;} 

            real gamma12 = sw3.gamma[type1][type2];
            real sigma12 = sw3.sigma[type1][type2];
            real a12     = sw3.a[type1][type2];
            real tmp = gamma12 / (sigma12*(d12/sigma12-a12)*(d12/sigma12-a12));
            real p2, f2;
            find_p2_and_f2
            (sigma12, a12, sw3.B[type1][type2], sw3.A[type1][type2], d12, p2, f2);

            // treat the two-body part in the same way as the many-body part
            real f12x = f2 * x12 * HALF; 
            real f12y = f2 * y12 * HALF; 
            real f12z = f2 * z12 * HALF; 
            if (cal_p) // accumulate potential energy
            {
                potential_energy += p2 * HALF;
            }

            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];  
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                real x3 = LDG(g_x, n3); 
                real y3 = LDG(g_y, n3); 
                real z3 = LDG(g_z, n3);
                real x23 = x3 - x2;
                real y23 = y3 - y2;
                real z23 = z3 - z2;
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x23, y23, z23, lx, ly, lz);
                real d23sq = x23 * x23 + y23 * y23 + z23* z23;
                if (d23sq > 14.5924) { continue; }
                real x13 = x3 - x1;
                real y13 = y3 - y1;
                real z13 = z3 - z1;
                dev_apply_mic(pbc_x, pbc_y, pbc_z, x13, y13, z13, lx, ly, lz);
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                if (d13 >= sw3.rc[type1][type3]) {continue;}
                
                real cos0   = sw3.cos0[type1][type2][type3];
                real lambda = sw3.lambda[type1][type2][type3];
                
                real exp123 = d13 / sw3.sigma[type1][type3] - sw3.a[type1][type3];
                exp123 = sw3.gamma[type1][type3] / exp123;
                exp123 = exp(gamma12 / (d12 / sigma12 - a12) + exp123);
                real one_over_d12d13 = ONE / (d12 * d13);
                real cos123 = (x12*x13 + y12*y13 + z12*z13) * one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv;

                real tmp1 = exp123 * (cos123 - cos0) * lambda;
                real tmp2 = tmp * (cos123 - cos0) * d12inv;

                if (cal_p) // accumulate potential energy
                {
                    potential_energy += (cos123 - cos0) * tmp1 * HALF;
                }

                real cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12; 
                f12x += tmp1 * (TWO * cos_d - tmp2 * x12);

                cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
                f12y += tmp1 * (TWO * cos_d - tmp2 * y12);

                cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
                f12z += tmp1 * (TWO * cos_d - tmp2 * z12);
            }

            g_f12x[index] = f12x;
            g_f12y[index] = f12y;
            g_f12z[index] = f12z;
        }
        if (cal_p) // save potential
        {
            g_potential[n1] = potential_energy;
        }
    }
}   
#endif




// force evaluation kernel for the SW potential
template <int cal_p, int cal_j, int cal_q>
static __global__ void gpu_find_force_sw3 
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z, SW2_Para sw3,
    int *g_neighbor_number, int *g_neighbor_list, int *g_type,
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
#else
    real* g_f12x, real* g_f12y, real* g_f12z, 
    real *g_x,  real *g_y,  real *g_z, real *g_vx, real *g_vy, real *g_vz,
#endif
    real *g_box_length, real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz,
    real *g_h, int *g_label, int *g_fv_index, real *g_fv 
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index

    real s_fx = ZERO;
    real s_fy = ZERO;
    real s_fz = ZERO;

    // if cal_p, then s1~s4 = px, py, pz, U; if cal_j, then s1~s5 = j1~j5
    real s1 = ZERO;
    real s2 = ZERO;
    real s3 = ZERO;
    real s4 = ZERO;
    real s5 = ZERO;

    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real vx1; 
        real vy1; 
        real vz1;
        if (!cal_p)
        {
            vx1 = LDG(g_vx, n1); 
            vy1 = LDG(g_vy, n1); 
            vz1 = LDG(g_vz, n1);
        }
        real lx = g_box_length[0]; 
        real ly = g_box_length[1]; 
        real lz = g_box_length[2];

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int index = i1 * number_of_particles + n1; 
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int neighbor_number_2 = g_neighbor_number[n2];
            int type2 = g_type[n2];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;  
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            if (d12 >= sw3.rc[type1][type2]) {continue;} 

            real f12x = LDG(g_f12x, index); 
            real f12y = LDG(g_f12y, index);
            real f12z = LDG(g_f12z, index);
            int offset = 0;
            for (int k = 0; k < neighbor_number_2; ++k)
            {
                if (n1 == g_neighbor_list[n2 + number_of_particles * k]) 
                { 
                    offset = k; break; 
                }
            }
            index = offset * number_of_particles + n2; 
            real f21x = LDG(g_f12x, index);
            real f21y = LDG(g_f12y, index);
            real f21z = LDG(g_f12z, index);

            s_fx += f12x - f21x; // accumulate force
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z; 
            
            if (cal_p) // accumulate potential energy and virial
            {
                s1 -= x12 * (f12x - f21x) * HALF; 
                s2 -= y12 * (f12y - f21y) * HALF; 
                s3 -= z12 * (f12z - f21z) * HALF;
            }
            
            if (cal_j) // heat current (EMD)
            {
                s1 += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s2 += (f21z * vz1) * x12;               // x-out
                s3 += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s4 += (f21z * vz1) * y12;               // y-out
                s5 += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            } 

            if (cal_q) // heat across some section (NEMD)
            {
                int index_12 = g_fv_index[n1] * 12;
                if (index_12 >= 0 && g_fv_index[n1 + number_of_particles] == n2)
                {
                    g_fv[index_12 + 0]  = f12x;
                    g_fv[index_12 + 1]  = f12y;
                    g_fv[index_12 + 2]  = f12z;
                    g_fv[index_12 + 3]  = f21x;
                    g_fv[index_12 + 4]  = f21y;
                    g_fv[index_12 + 5]  = f21z;
                    g_fv[index_12 + 6]  = vx1;
                    g_fv[index_12 + 7]  = vy1;
                    g_fv[index_12 + 8]  = vz1;
                    g_fv[index_12 + 9]  = LDG(g_vx, n2);
                    g_fv[index_12 + 10] = LDG(g_vy, n2);
                    g_fv[index_12 + 11] = LDG(g_vz, n2);
                }  
            }
        }

        g_fx[n1] = s_fx; 
        g_fy[n1] = s_fy; 
        g_fz[n1] = s_fz;  
        if (cal_p) // save stress and potential
        {
            g_sx[n1] = s1; 
            g_sy[n1] = s2; 
            g_sz[n1] = s3;
        }
        if (cal_j) // save heat current
        {
            g_h[n1 + 0 * number_of_particles] = s1;
            g_h[n1 + 1 * number_of_particles] = s2;
            g_h[n1 + 2 * number_of_particles] = s3;
            g_h[n1 + 3 * number_of_particles] = s4;
            g_h[n1 + 4 * number_of_particles] = s5;
        }
    }
}    
 



// Find force and related quantities for the SW potential (A wrapper)
void SW2::compute(Parameters *para, GPU_Data *gpu_data)
{
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE_SW + 1;
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    int *NN = gpu_data->NN_local; 
    int *NL = gpu_data->NL_local;
    int *type = gpu_data->type;
    real *x = gpu_data->x; 
    real *y = gpu_data->y; 
    real *z = gpu_data->z;
    real *vx = gpu_data->vx; 
    real *vy = gpu_data->vy; 
    real *vz = gpu_data->vz;
    real *fx = gpu_data->fx; 
    real *fy = gpu_data->fy; 
    real *fz = gpu_data->fz;
    real *box_length = gpu_data->box_length;
    real *sx = gpu_data->virial_per_atom_x; 
    real *sy = gpu_data->virial_per_atom_y; 
    real *sz = gpu_data->virial_per_atom_z; 
    real *pe = gpu_data->potential_per_atom;
    real *h = gpu_data->heat_per_atom; 
    
    int *label = gpu_data->label;
    int *fv_index = gpu_data->fv_index;
    real *fv = gpu_data->fv;

    real *f12x = sw2_data.f12x; 
    real *f12y = sw2_data.f12y; 
    real *f12z = sw2_data.f12z;
           
    if (para->hac.compute)    
    {
        gpu_find_force_sw3_partial<0><<<grid_size, BLOCK_SIZE_SW>>> 
        (
            N, pbc_x, pbc_y, pbc_z, sw2_para, NN, NL, type, x, y, z, 
            box_length, pe, f12x, f12y, f12z 
        );

        gpu_find_force_sw3<0, 1, 0><<<grid_size, BLOCK_SIZE_SW>>>
        (
            N, pbc_x, pbc_y, pbc_z, sw2_para, NN, NL, type, 
            f12x, f12y, f12z, x, y, z, vx, vy, vz, 
            box_length, fx, fy, fz, sx, sy, sz, h, label, fv_index, fv
        );
    }
    else if (para->shc.compute)
    {
        gpu_find_force_sw3_partial<0><<<grid_size, BLOCK_SIZE_SW>>> 
        (
            N, pbc_x, pbc_y, pbc_z, sw2_para, NN, NL, type, x, y, z, 
            box_length, pe, f12x, f12y, f12z 
        );

        gpu_find_force_sw3<0, 0, 1><<<grid_size, BLOCK_SIZE_SW>>>
        (
            N, pbc_x, pbc_y, pbc_z, sw2_para, NN, NL, type, 
            f12x, f12y, f12z, x, y, z, vx, vy, vz, 
            box_length, fx, fy, fz, sx, sy, sz, h, label, fv_index, fv
        );
    }
    else
    {
        gpu_find_force_sw3_partial<1><<<grid_size, BLOCK_SIZE_SW>>> 
        (
            N, pbc_x, pbc_y, pbc_z, sw2_para, NN, NL, type, x, y, z, 
            box_length, pe, f12x, f12y, f12z 
        );

        gpu_find_force_sw3<1, 0, 0><<<grid_size, BLOCK_SIZE_SW>>>
        (
            N, pbc_x, pbc_y, pbc_z, sw2_para, NN, NL, type, 
            f12x, f12y, f12z, x, y, z, vx, vy, vz, 
            box_length, fx, fy, fz, sx, sy, sz, h, label, fv_index, fv
        );
    }

    #ifdef DEBUG
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    #endif
}




