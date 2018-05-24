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
#include "mic.cu" // static __device__ dev_apply_mic(...)
#include "tersoff_1989_1.h"


/*----------------------------------------------------------------------------80
The single-element version of the Tersoff potential as described in  
    [1] J. Tersoff, Modeling solid-state chemistry: Interatomic potentials 
        for multicomponent systems, PRB 39, 5566 (1989).
------------------------------------------------------------------------------*/


// best block size here: 64 or 128
#define BLOCK_SIZE_FORCE 64


/*------------------------------------------------------------------------------
    Some simple functions and their derivatives
------------------------------------------------------------------------------*/


static __device__ void find_fr_and_frp
(Tersoff ters0, real d12, real &fr, real &frp)
{   
    fr  = ters0.a * exp(- ters0.lambda * d12);    
    frp = - ters0.lambda * fr;
}


static __device__ void find_fa_and_fap
(Tersoff ters0, real d12, real &fa, real &fap)
{    
    fa  = ters0.b * exp(- ters0.mu * d12);    
    fap = - ters0.mu * fa; 
}


static __device__ void find_fa(Tersoff ters0, real d12, real &fa)
{   
    fa  = ters0.b * exp(- ters0.mu * d12);   
}


static __device__ void find_fc_and_fcp
(
    Tersoff ters0, 
    real d12, real &fc, real &fcp
)
{
    if (d12 < ters0.r1) {fc = ONE; fcp = ZERO;}
    else if (d12 < ters0.r2)
    {              
        fc  =  cos(ters0.pi_factor * (d12 - ters0.r1)) * HALF + HALF;
        fcp = -sin(ters0.pi_factor * (d12 - ters0.r1))*ters0.pi_factor*HALF;
    }
    else {fc  = ZERO; fcp = ZERO;}
}


static __device__ void find_fc(Tersoff ters0, real d12, real &fc)
{
    if (d12 < ters0.r1) {fc  = ONE;}
    else if (d12 < ters0.r2) 
    {fc = cos(ters0.pi_factor * (d12 - ters0.r1)) * HALF + HALF;}
    else {fc  = ZERO;}
}


static __device__ void find_g_and_gp(Tersoff ters0, real cos, real &g, real &gp)
{  
    real temp = ters0.d2 + (cos - ters0.h) * (cos - ters0.h);
    g  = ters0.one_plus_c2overd2 - ters0.c2 / temp;    
    gp = TWO * ters0.c2 * (cos - ters0.h) / (temp * temp); 
}


static __device__ void find_g(Tersoff ters0, real cos, real &g)
{ 
    real temp = ters0.d2 + (cos - ters0.h) * (cos - ters0.h);
    g  = ters0.one_plus_c2overd2 - ters0.c2 / temp;  
}
 

/*------------------------------------------------------------------------------
    Find the bond-order functions and their derivatives first.
    This is an efficient approach.
------------------------------------------------------------------------------*/
static __global__ void find_force_tersoff_step1
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z,
    Tersoff ters0, 
    int* g_neighbor_number, int* g_neighbor_list,
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
    //<<<(number_of_particles - 1) / MAX_THREAD + 1, MAX_THREAD>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];

        real x1 = LDG(g_x, n1); real y1 = LDG(g_y, n1); real z1 = LDG(g_z, n1);

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {      
            int n2 = g_neighbor_list[n1 + number_of_particles * i1];
            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic
            (
                pbc_x, pbc_y, pbc_z, x12, y12, z12, LDG(g_box_length, 0), 
                LDG(g_box_length, 1), LDG(g_box_length, 2)
            );
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real zeta = ZERO;
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int n3 = g_neighbor_list[n1 + number_of_particles * i2];  
                if (n3 == n2) { continue; } // ensure that n3 != n2

                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;         
                dev_apply_mic
                (
                    pbc_x, pbc_y, pbc_z, x13, y13, z13, LDG(g_box_length, 0), 
                    LDG(g_box_length, 1), LDG(g_box_length, 2)
                );
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
                real cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
                real fc13, g123; 

                find_fc(ters0, d13, fc13);
                find_g(ters0, cos123, g123);

                zeta += fc13 * g123;
            } 
            real bzn, b12;

            bzn = pow(ters0.beta * zeta, ters0.n);
            b12 = pow(ONE + bzn, ters0.minus_half_over_n);

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




// second step: calculate the partial forces dU_i/dr_ij
template <int cal_p>
static __global__ void find_force_tersoff_step2
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z,
    Tersoff ters0, int *g_neighbor_number, int *g_neighbor_list,
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
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];

        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);
        real potential_energy = ZERO;

        for (int i1 = 0; i1 < neighbor_number; ++i1)
        { 

            int index = i1 * number_of_particles + n1;   
            int n2 = g_neighbor_list[index];

            real x12  = LDG(g_x, n2) - x1;
            real y12  = LDG(g_y, n2) - y1;
            real z12  = LDG(g_z, n2) - z1;
            dev_apply_mic
            (
                pbc_x, pbc_y, pbc_z, x12, y12, z12, LDG(g_box_length, 0), 
                LDG(g_box_length, 1), LDG(g_box_length, 2)
            );
            real d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
            real d12inv = ONE / d12;
            real fc12, fcp12, fa12, fap12, fr12, frp12;

            find_fc_and_fcp(ters0, d12, fc12, fcp12);
            find_fa_and_fap(ters0, d12, fa12, fap12);
            find_fr_and_frp(ters0, d12, fr12, frp12);
   
            // accumulate_force_12 
            real b12 = LDG(g_b, index);    
            real factor3=(fcp12*(fr12-b12*fa12)+fc12*(frp12-b12*fap12))*d12inv;   
            real f12x = x12 * factor3 * HALF; 
            real f12y = y12 * factor3 * HALF;
            real f12z = z12 * factor3 * HALF;

            if (cal_p) // accumulate potential energy
            {
                potential_energy += fc12 * (fr12 - b12 * fa12) * HALF;
            }

            // accumulate_force_123
            real bp12 = LDG(g_bp, index);
            for (int i2 = 0; i2 < neighbor_number; ++i2)
            {
                int index_2 = n1 + number_of_particles * i2;        
                int n3 = g_neighbor_list[index_2];   
                if (n3 == n2) { continue; } 
                real x13 = LDG(g_x, n3) - x1;
                real y13 = LDG(g_y, n3) - y1;
                real z13 = LDG(g_z, n3) - z1;
                dev_apply_mic
                (
                    pbc_x, pbc_y, pbc_z, x13, y13, z13, LDG(g_box_length, 0), 
                    LDG(g_box_length, 1), LDG(g_box_length, 2)
                );
                real d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);   
                real fc13, fa13;
                find_fc(ters0, d13, fc13);
                find_fa(ters0, d13, fa13); 

                real bp13 = LDG(g_bp, index_2);
                real one_over_d12d13 = ONE / (d12 * d13);   
                real cos123 = (x12*x13 + y12*y13 + z12*z13)*one_over_d12d13;
                real cos123_over_d12d12 = cos123*d12inv*d12inv; 
                real g123, gp123; 
                find_g_and_gp(ters0, cos123, g123, gp123);

                real temp123a=(-bp12*fc12*fa12*fc13-bp13*fc13*fa13*fc12)*gp123;
                real temp123b= - bp13 * fc13 * fa13 * fcp12 * g123 * d12inv;
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
        if (cal_p) // save potential
        {
            g_potential[n1] = potential_energy;
        }
    }
}   


            
/*----------------------------------------------------------------------------80
    Calculate forces, potential energy, and virial stress
------------------------------------------------------------------------------*/
template <int cal_p, int cal_j, int cal_q>
static __global__ void find_force_tersoff_step3
(
    int number_of_particles, int pbc_x, int pbc_y, int pbc_z,
    Tersoff ters0, 
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
    real *g_h, int *g_label, int *g_fv_index, real *g_fv 
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    real s_fx = ZERO;
    real s_fy = ZERO;
    real s_fz = ZERO;

    // if cal_p, then s1~s4 = px, py, pz, U; if cal_j, then s1~s5 = j1~j5
    __shared__ real s1[BLOCK_SIZE_FORCE];
    __shared__ real s2[BLOCK_SIZE_FORCE];
    __shared__ real s3[BLOCK_SIZE_FORCE];
    __shared__ real s4[BLOCK_SIZE_FORCE];
    __shared__ real s5[BLOCK_SIZE_FORCE];
    s1[threadIdx.x] = ZERO; 
    s2[threadIdx.x] = ZERO; 
    s3[threadIdx.x] = ZERO;
    s4[threadIdx.x] = ZERO;
    s5[threadIdx.x] = ZERO;

    if (n1 < number_of_particles)
    {
        int neighbor_number = g_neighbor_number[n1];

        real x1 = LDG(g_x, n1); 
        real y1 = LDG(g_y, n1); 
        real z1 = LDG(g_z, n1);

        real vx1, vy1, vz1;
        if (cal_j || cal_q)
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
            dev_apply_mic
            (
                pbc_x, pbc_y, pbc_z, x12, y12, z12, LDG(g_box_length, 0), 
                LDG(g_box_length, 1), LDG(g_box_length, 2)
            );

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
    
            // per atom force 
            s_fx += f12x - f21x; 
            s_fy += f12y - f21y; 
            s_fz += f12z - f21z; 

            // per-atom stress
            if (cal_p)
            {
                s1[threadIdx.x] -= x12 * (f12x - f21x) * HALF; 
                s2[threadIdx.x] -= y12 * (f12y - f21y) * HALF; 
                s3[threadIdx.x] -= z12 * (f12z - f21z) * HALF;
            }

            // per-atom heat current
            if (cal_j)
            {
                s1[threadIdx.x] += (f21x * vx1 + f21y * vy1) * x12;  // x-in
                s2[threadIdx.x] += (f21z * vz1) * x12;               // x-out
                s3[threadIdx.x] += (f21x * vx1 + f21y * vy1) * y12;  // y-in
                s4[threadIdx.x] += (f21z * vz1) * y12;               // y-out
                s5[threadIdx.x] += (f21x*vx1+f21y*vy1+f21z*vz1)*z12; // z-all
            }
 
            // accumulate heat across some sections (for NEMD)
            if (cal_q)
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

        // save force
        g_fx[n1] = s_fx; 
        g_fy[n1] = s_fy; 
        g_fz[n1] = s_fz;

        if (cal_p) // save stress and potential
        {
            g_sx[n1] = s1[threadIdx.x]; 
            g_sy[n1] = s2[threadIdx.x]; 
            g_sz[n1] = s3[threadIdx.x];
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


            

/*
    Force evaluation for the Tersoff potential (a wrapper)
*/
void gpu_find_force_tersoff1
(Parameters *para, Force_Model *force_model, GPU_Data *gpu_data)
{
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE_FORCE + 1;
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    int *NN = gpu_data->NN_local; 
    int *NL = gpu_data->NL_local;
    real *x = gpu_data->x; 
    real *y = gpu_data->y; 
    real *z = gpu_data->z;
    real *vx = gpu_data->vx; 
    real *vy = gpu_data->vy; 
    real *vz = gpu_data->vz;
    real *fx = gpu_data->fx; 
    real *fy = gpu_data->fy; 
    real *fz = gpu_data->fz;
    real *b = gpu_data->b; 
    real *bp = gpu_data->bp; 
    real *box_length = gpu_data->box_length;
    real *sx = gpu_data->virial_per_atom_x; 
    real *sy = gpu_data->virial_per_atom_y; 
    real *sz = gpu_data->virial_per_atom_z; 
    real *pe = gpu_data->potential_per_atom;
    real *h = gpu_data->heat_per_atom;   
    
    int *label = gpu_data->label;
    int *fv_index = gpu_data->fv_index;
    real *fv = gpu_data->fv;

    real *f12x = gpu_data->f12x; 
    real *f12y = gpu_data->f12y; 
    real *f12z = gpu_data->f12z; 
    
    find_force_tersoff_step1<<<grid_size, BLOCK_SIZE_FORCE>>>
    (       
        N, pbc_x, pbc_y, pbc_z, force_model->ters0, 
        NN, NL, x, y, z, box_length, b, bp
    );

    if (para->hac.compute)
    {
        find_force_tersoff_step2<0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, para->pbc_x, para->pbc_y, para->pbc_z, force_model->ters0, 
            NN, NL, b, bp, x, y, z, box_length, pe, f12x, f12y, f12z
        );
        find_force_tersoff_step3<0, 1, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, para->pbc_x, para->pbc_y, para->pbc_z,
            force_model->ters0, NN, NL, 
            f12x, f12y, f12z, x, y, z, vx, vy, vz, box_length, fx, fy, fz, 
            sx, sy, sz, h, label, fv_index, fv
        );
    }
    else if (para->shc.compute)
    {
        find_force_tersoff_step2<0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, para->pbc_x, para->pbc_y, para->pbc_z, force_model->ters0, 
            NN, NL, b, bp, x, y, z, box_length, pe, f12x, f12y, f12z
        );
        find_force_tersoff_step3<0, 0, 1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, para->pbc_x, para->pbc_y, para->pbc_z,
            force_model->ters0, NN, NL, 
            f12x, f12y, f12z, x, y, z, vx, vy, vz, box_length, fx, fy, fz, 
            sx, sy, sz, h, label, fv_index, fv
        );
    }
    else
    {
        find_force_tersoff_step2<1><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, para->pbc_x, para->pbc_y, para->pbc_z, force_model->ters0, 
            NN, NL, b, bp, x, y, z, box_length, pe, f12x, f12y, f12z
        );
        find_force_tersoff_step3<1, 0, 0><<<grid_size, BLOCK_SIZE_FORCE>>>
        (
            N, para->pbc_x, para->pbc_y, para->pbc_z,
            force_model->ters0, NN, NL, 
            f12x, f12y, f12z, x, y, z, vx, vy, vz, box_length, fx, fy, fz, 
            sx, sy, sz, h, label, fv_index, fv
        );
    }

}



