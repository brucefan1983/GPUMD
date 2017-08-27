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




#include "common.h"
#include "sw_bp.h"




// best block size here: 64 or 128
#define BLOCK_SIZE_SW 64

#ifdef USE_DP
    #define SQRT2INV     0.707106781186548
    #define BP_rc        2.8
    #define BP_sigma1    0.2103
    #define BP_sigma2    0.1559
    #define BP_a1        13.3143
    #define BP_a2        17.9602
    #define BP_A1        4.3807
    #define BP_A2        4.0936 
    #define BP_B1        3045.2
    #define BP_B2        10164.1
    #define BP_gamma1    2.1707
    #define BP_gamma2    2.9282    
    #define BP_lambda1   9.2660
    #define BP_lambda2   11.4510
    #define BP_mcos1     0.1045
    #define BP_mcos2     0.2419
#else
    #define SQRT2INV     0.7071068f
    #define BP_rc        2.8f
    #define BP_sigma1    0.2103f
    #define BP_sigma2    0.1559f
    #define BP_a1        13.3143f
    #define BP_a2        17.9602f
    #define BP_A1        4.3807f
    #define BP_A2        4.0936f 
    #define BP_B1        3045.2f
    #define BP_B2        10164.1f
    #define BP_gamma1    2.1707f
    #define BP_gamma2    2.9282f    
    #define BP_lambda1   9.2660f
    #define BP_lambda2   11.4510f
    #define BP_mcos1     0.1045f
    #define BP_mcos2     0.2419f
#endif




/*----------------------------------------------------------------------------80
If this is 1, calculate the full conductivity tensor but without the in-out 
decomposition. If it is 0, only calculate the diagonal elements but with the 
in-out decomposition.
------------------------------------------------------------------------------*/
#define FULL_TENSOR   1 




/*----------------------------------------------------------------------------80
    This is a special SW potential and I thus hard coded the
    potential parameters. 
    Reference: 
        Wen Xu et al., J. Appl. Phys. 117, 214308 (2015).
------------------------------------------------------------------------------*/




// apply the mininum image convention (This is the fastest version I found)
static __device__ void dev_apply_mic
(
    real *x12, real *y12, real *z12,
    real lx, real ly, real lz
)
{
    if      (*x12 < - lx * HALF) {*x12 += lx;}
    else if (*x12 > + lx * HALF) {*x12 -= lx;}
    if      (*y12 < - ly * HALF) {*y12 += ly;}
    else if (*y12 > + ly * HALF) {*y12 -= ly;}
    if      (*z12 < - lz * HALF) {*z12 += lz;}
    else if (*z12 > + lz * HALF) {*z12 -= lz;}
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




// force evaluation kernel for the SW potential
template <int cal_p, int cal_j, int cal_q>
static __global__ void gpu_find_force_sw
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z,
    int *g_neighbor_number, int *g_neighbor_list, int *g_type,
#ifdef USE_LDG
    const real* __restrict__ g_x, 
    const real* __restrict__ g_y, 
    const real* __restrict__ g_z, 
    const real* __restrict__ g_vx, 
    const real* __restrict__ g_vy, 
    const real* __restrict__ g_vz,
#else
    real *g_x,  real *g_y,  real *g_z, real *g_vx, real *g_vy, real *g_vz,
#endif
    real *g_box_length, real *g_fx, real *g_fy, real *g_fz,
    real *g_sx, real *g_sy, real *g_sz, real *g_potential, 
    real *g_h, int *g_label, int *g_fv_index, real *g_fv 
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index

    __shared__ real s_fx[BLOCK_SIZE_SW];
    __shared__ real s_fy[BLOCK_SIZE_SW];
    __shared__ real s_fz[BLOCK_SIZE_SW];
    // if cal_p, then s1~s4 = px, py, pz, U; if cal_j, then s1~s5 = j1~j5
    __shared__ real s1[BLOCK_SIZE_SW];
    __shared__ real s2[BLOCK_SIZE_SW];
    __shared__ real s3[BLOCK_SIZE_SW];
    __shared__ real s4[BLOCK_SIZE_SW];
    __shared__ real s5[BLOCK_SIZE_SW];

    s_fx[threadIdx.x] = ZERO; 
    s_fy[threadIdx.x] = ZERO; 
    s_fz[threadIdx.x] = ZERO;  
    s1[threadIdx.x] = ZERO; 
    s2[threadIdx.x] = ZERO; 
    s3[threadIdx.x] = ZERO;
    s4[threadIdx.x] = ZERO;
    s5[threadIdx.x] = ZERO;

    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];
        int type1 = g_type[n1];
        real x1 = LDG(g_x, n1); real y1 = LDG(g_y, n1); real z1 = LDG(g_z, n1);
        real vx1 = LDG(g_vx, n1); 
        real vy1 = LDG(g_vy, n1); 
        real vz1 = LDG(g_vz, n1);

        // This trick can make the code 2% faster
        real lx = g_box_length[0] * pbc_x; 
        real ly = g_box_length[1] * pbc_y; 
        real lz = g_box_length[2] * pbc_z;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            int neighbor_number_2 = g_neighbor_number[n2];
            int type2 = g_type[n2];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            
            dev_apply_mic(&x12, &y12, &z12, lx, ly, lz);

            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            if (d12 >= BP_rc) {continue;}
        
            // P1 = P in one layer; P2 = P in the other layer
            real p2, f2;
            if (type1 == type2) // P1-P1 or P2-P2
            {
                find_p2_and_f2(BP_sigma1, BP_a1, BP_B1, BP_A1, d12, p2, f2);
            }
            else // P1-P2 or P2-P1
            {
                find_p2_and_f2(BP_sigma2, BP_a2, BP_B2, BP_A2, d12, p2, f2);
            }

            // treat the two-body part in the same way as the many-body part
            real f12x = f2 * x12 * HALF; 
            real f12y = f2 * y12 * HALF; 
            real f12z = f2 * z12 * HALF; 
            real f21x = -f12x; 
            real f21y = -f12y; 
            real f21z = -f12z; 
             
            // accumulate_force_123
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {       
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];  
                if (n3 == n2) { continue; }
                int type3 = g_type[n3];
                //if ( (type1 != type2) && (type3 == type2) ) { continue; }

                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                
                dev_apply_mic(&x13,&y13,&z13,lx,ly,lz);

                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                if (d13 >= BP_rc) {continue;}
                real cos123 = (x12*x13 + y12*y13 + z12*z13) / (d12*d13);

                real exp123, gamma2;
                if (type1 == type2)
                { 
                    exp123=exp(BP_gamma1/(d12/BP_sigma1-BP_a1)
                              +BP_gamma1/(d13/BP_sigma1-BP_a1));
                    gamma2=BP_gamma1
                        /(BP_sigma1*(d12/BP_sigma1-BP_a1)*(d12/BP_sigma1-BP_a1));
                }
                else
                {
                    exp123 = exp(BP_gamma2/(d12/BP_sigma2-BP_a2) 
                               + BP_gamma2/(d13/BP_sigma2-BP_a2));
                    gamma2=BP_gamma2
                        /(BP_sigma2*(d12/BP_sigma2-BP_a2)*(d12/BP_sigma2-BP_a2));
                }

                 
                real cos_d;
                if (type1 == type2 && type1 == type3) // theta_in
                {

                    if (cal_p) // accumulate potential energy
                    {
                        s4[threadIdx.x] += (cos123+BP_mcos1)*(cos123+BP_mcos1)
                                         * exp123*BP_lambda1*HALF;
                    }

                    cos_d = x13 / (d12 * d13) - x12 * cos123 / (d12 * d12); 
                    f12x += exp123 * (cos123+BP_mcos1) * BP_lambda1
                          * (TWO*cos_d-gamma2*(cos123+BP_mcos1)*x12/d12);
                    cos_d = y13 / (d12 * d13) - y12 * cos123 / (d12 * d12);
                    f12y += exp123 * (cos123+BP_mcos1) * BP_lambda1
                          * (TWO*cos_d-gamma2*(cos123+BP_mcos1)*y12/d12);
                    cos_d = z13 / (d12 * d13) - z12 * cos123 / (d12 * d12);
                    f12z += exp123 * (cos123+BP_mcos1) * BP_lambda1
                          * (TWO*cos_d-gamma2*(cos123+BP_mcos1)*z12/d12);
                }
                else // theta_out
                {

                    if (cal_p) // accumulate potential energy
                    {
                        s4[threadIdx.x] += (cos123+BP_mcos2)*(cos123+BP_mcos2)
                                         * exp123*BP_lambda2*HALF;
                    }

                    cos_d = x13 / (d12 * d13) - x12 * cos123 / (d12 * d12);
                    f12x += exp123 * (cos123+BP_mcos2) * BP_lambda2
                          * (TWO*cos_d-gamma2*(cos123+BP_mcos2)*x12/d12);
                    cos_d = y13 / (d12 * d13) - y12 * cos123 / (d12 * d12);
                    f12y += exp123 * (cos123+BP_mcos2) * BP_lambda2
                          * (TWO*cos_d-gamma2*(cos123+BP_mcos2)*y12/d12);
                    cos_d = z13 / (d12 * d13) - z12 * cos123 / (d12 * d12);
                    f12z += exp123 * (cos123+BP_mcos2) * BP_lambda2
                          * (TWO*cos_d-gamma2*(cos123+BP_mcos2)*z12/d12);
                }
            }

            // accumulate_force_213
            for (int i2 = 0; i2 < neighbor_number_2; ++i2)
            {
                int n3 = g_neighbor_list[n2 + number_of_particles * i2];        
                if (n3 == n1) { continue; } 
                int type3 = g_type[n3];
                //if ( (type1 != type2) && (type3 == type1) ) { continue; }

                real x23 = LDG(g_x, n3) - LDG(g_x, n2);
                real y23 = LDG(g_y, n3) - LDG(g_y, n2);
                real z23 = LDG(g_z, n3) - LDG(g_z, n2);
                
                dev_apply_mic(&x23,&y23,&z23,lx,ly,lz);

                real d23 = sqrt(x23 * x23 + y23 * y23 + z23 * z23);
                if (d23 >= BP_rc) {continue;} 
                real cos213 = -(x12*x23 + y12*y23 + z12*z23) / (d12*d23); 

                real exp213, gamma2;
                if (type1 == type2)
                { 
                    exp213=exp(BP_gamma1/(d12/BP_sigma1-BP_a1)
                              +BP_gamma1/(d23/BP_sigma1-BP_a1));
                    gamma2=BP_gamma1
                        /(BP_sigma1*(d12/BP_sigma1-BP_a1)*(d12/BP_sigma1-BP_a1));
                }
                else
                {
                    exp213 = exp(BP_gamma2/(d12/BP_sigma2-BP_a2) 
                               + BP_gamma2/(d23/BP_sigma2-BP_a2));
                    gamma2=BP_gamma2
                        /(BP_sigma2*(d12/BP_sigma2-BP_a2)*(d12/BP_sigma2-BP_a2));
                }

                real cos_d;
                if (type1 == type2 && type1 == type3) // theta_in
                {
                    cos_d = x23 / (d12 * d23) + x12 * cos213 / (d12 * d12);
                    f21x += exp213 * (cos213+BP_mcos1) * BP_lambda1
                          * (TWO*cos_d+gamma2*(cos213+BP_mcos1)*x12/d12);
                    cos_d = y23 / (d12 * d23) + y12 * cos213 / (d12 * d12);
                    f21y += exp213 * (cos213+BP_mcos1) * BP_lambda1
                          * (TWO*cos_d+gamma2*(cos213+BP_mcos1)*y12/d12);
                    cos_d = z23 / (d12 * d23) + z12 * cos213 / (d12 * d12);
                    f21z += exp213 * (cos213+BP_mcos1) * BP_lambda1
                          * (TWO*cos_d+gamma2*(cos213+BP_mcos1)*z12/d12);
                }
                else // theta_out
                {
                    cos_d = x23 / (d12 * d23) + x12 * cos213 / (d12 * d12);
                    f21x += exp213 * (cos213+BP_mcos2) * BP_lambda2
                          * (TWO*cos_d+gamma2*(cos213+BP_mcos2)*x12/d12);
                    cos_d = y23 / (d12 * d23) + y12 * cos213 / (d12 * d12);
                    f21y += exp213 * (cos213+BP_mcos2) * BP_lambda2
                          * (TWO*cos_d+gamma2*(cos213+BP_mcos2)*y12/d12);
                    cos_d = z23 / (d12 * d23) + z12 * cos213 / (d12 * d12);
                    f21z += exp213 * (cos213+BP_mcos2) * BP_lambda2
                          * (TWO*cos_d+gamma2*(cos213+BP_mcos2)*z12/d12);
                }
            }  
            
            s_fx[threadIdx.x] += f12x - f21x; // accumulate force
            s_fy[threadIdx.x] += f12y - f21y; 
            s_fz[threadIdx.x] += f12z - f21z; 
            
            if (cal_p) // accumulate potential energy and virial
            {
                s4[threadIdx.x] += p2 * HALF; // two-body potential
                s1[threadIdx.x] -= x12 * (f12x - f21x) * HALF; 
                s2[threadIdx.x] -= y12 * (f12y - f21y) * HALF; 
                s3[threadIdx.x] -= z12 * (f12z - f21z) * HALF;
            }
            
            if (cal_j) // heat current (EMD)
            {
#if FULL_TENSOR
                real f21_dot_v1 = f21x*vx1+f21y*vy1+f21z*vz1;
                s1[threadIdx.x] += f21_dot_v1 * x12;                    // x-0
                s2[threadIdx.x] += f21_dot_v1 * y12;                    // y-0
                s3[threadIdx.x] += f21_dot_v1 * (x12 + y12) * SQRT2INV; // x-45
                s4[threadIdx.x] += f21_dot_v1 * (y12 - x12) * SQRT2INV; // y-45
                s5[threadIdx.x] += f21_dot_v1 * z12;                    // z
#else
                s1[threadIdx.x] += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s2[threadIdx.x] += (f21z * vz1) * x12;               // x-out
                s3[threadIdx.x] += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s4[threadIdx.x] += (f21z * vz1) * y12;               // y-out
                s5[threadIdx.x] += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
#endif
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

        g_fx[n1] = s_fx[threadIdx.x]; 
        g_fy[n1] = s_fy[threadIdx.x]; 
        g_fz[n1] = s_fz[threadIdx.x];  
        if (cal_p) // save stress and potential
        {
            g_sx[n1] = s1[threadIdx.x]; 
            g_sy[n1] = s2[threadIdx.x]; 
            g_sz[n1] = s3[threadIdx.x];
            g_potential[n1] = s4[threadIdx.x];
        }
        if (cal_j) // save heat current
        {
            g_h[n1 + 0 * number_of_particles] = s1[threadIdx.x];
            g_h[n1 + 1 * number_of_particles] = s2[threadIdx.x];
            g_h[n1 + 2 * number_of_particles] = s3[threadIdx.x];
            g_h[n1 + 3 * number_of_particles] = s4[threadIdx.x];
            g_h[n1 + 4 * number_of_particles] = s5[threadIdx.x];
        }
    }
}    
 



// Find force and related quantities for the SW potential (A wrapper)
void gpu_find_force_sw_bp(Parameters *para, GPU_Data *gpu_data)
{
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE_SW + 1;
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    int *NN = gpu_data->NN; 
    int *NL = gpu_data->NL;
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
           
    if (para->hac.compute)    
    {
        gpu_find_force_sw<0, 1, 0><<<grid_size, BLOCK_SIZE_SW>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, x, y, z, vx, vy, vz, 
            box_length, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
    }
    else if (para->shc.compute)
    {
        gpu_find_force_sw<0, 0, 1><<<grid_size, BLOCK_SIZE_SW>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, x, y, z, vx, vy, vz, 
            box_length, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
    }
    else
    {
        gpu_find_force_sw<1, 0, 0><<<grid_size, BLOCK_SIZE_SW>>>
        (
            N, pbc_x, pbc_y, pbc_z, NN, NL, type, x, y, z, vx, vy, vz, 
            box_length, fx, fy, fz, sx, sy, sz, pe, h, label, fv_index, fv
        );
    }

    #ifdef DEBUG
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    #endif
}




