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
#include "mic_template.cu" // static __device__ void dev_apply_mic(...)
#include "force.h"
#include "lj1.h"
#include "ri.h"
#include "eam_zhou_2004.h"
#include "eam_dai_2006.h"
#include "sw_1985.h"
#include "sw_1985_2.h"
#include "vashishta.h"
#include "vashishta_table.h"
#include "tersoff_1989_1.h"
#include "tersoff_1989_2.h"
#include "rebo_mos2.h"



#ifndef FIXED_NL


// Construct the local neighbor list from the global one (Kernel)
template <int pbc_x, int pbc_y, int pbc_z>
static __global__ void gpu_find_neighbor_local
(
    int N, real cutoff_square, real *box_length,
    int *NN, int *NL, int *NN_local, int *NL_local, 
#ifdef USE_LDG
    const real* __restrict__ x, 
    const real* __restrict__ y, 
    const real* __restrict__ z
#else
    real *x, real *y, real *z
#endif
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    real lx = box_length[0];
    real ly = box_length[1];
    real lz = box_length[2];
    if (n1 < N)
    {  
        int neighbor_number = NN[n1];
        real x1 = LDG(x, n1);   
        real y1 = LDG(y, n1);
        real z1 = LDG(z, n1);  
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = NL[n1 + N * i1];
            real x12  = LDG(x, n2) - x1;
            real y12  = LDG(y, n2) - y1;
            real z12  = LDG(z, n2) - z1;
            dev_apply_mic<pbc_x, pbc_y, pbc_z>(lx, ly, lz, &x12, &y12, &z12);
            real distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < cutoff_square)
            {        
                NL_local[count * N + n1] = n2;
                ++count;
            }
        }
        NN_local[n1] = count;
    }
}


// Construct the local neighbor list from the global one (Wrapper)
static void find_neighbor_local(Parameters *para, GPU_Data *gpu_data, real rc2)
{
     
    int N = para->N;
    int grid_size = (N - 1) / BLOCK_SIZE + 1; 
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    int *NN = gpu_data->NN;
    int *NL = gpu_data->NL;
    int *NN_local = gpu_data->NN_local;
    int *NL_local = gpu_data->NL_local;
    real *x = gpu_data->x;
    real *y = gpu_data->y;
    real *z = gpu_data->z;
    real *box = gpu_data->box_length;
      
    if (pbc_x && pbc_y && pbc_z)
        gpu_find_neighbor_local<1,1,1><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (pbc_x && pbc_y && !pbc_z)
        gpu_find_neighbor_local<1,1,0><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (pbc_x && !pbc_y && pbc_z)
        gpu_find_neighbor_local<1,0,1><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (!pbc_x && pbc_y && pbc_z)
        gpu_find_neighbor_local<0,1,1><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (pbc_x && !pbc_y && !pbc_z)
        gpu_find_neighbor_local<1,0,0><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (!pbc_x && pbc_y && !pbc_z)
        gpu_find_neighbor_local<0,1,0><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (!pbc_x && !pbc_y && pbc_z)
        gpu_find_neighbor_local<0,0,1><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);
    if (!pbc_x && !pbc_y && !pbc_z)
        gpu_find_neighbor_local<0,0,0><<<grid_size, BLOCK_SIZE>>>
        (N, rc2, box, NN, NL, NN_local, NL_local, x, y, z);  
}

#endif




// calculate force and related quantities 
void gpu_find_force
(Force_Model *force_model, Parameters *para, GPU_Data *gpu_data)
{     
    switch (force_model->type)
    {
        case 0:
            gpu_find_force_lj1(para, force_model->lj1, gpu_data);
            break;
        case 10: 
            gpu_find_force_ri(para, force_model->ri, gpu_data);
            break;
        case 20:
            gpu_find_force_eam(para, force_model, gpu_data);
            break;
        case 21:
            gpu_find_force_fs(para, force_model, gpu_data);
            break;
        case 30:  
            find_neighbor_local(para, gpu_data, force_model->rc * force_model->rc); 
            gpu_find_force_sw(para, force_model->sw, gpu_data);
            break;
        case 32:  
            gpu_find_force_vashishta(para, force_model->vas, gpu_data);
            break;
        case 33:  
            find_neighbor_local(para, gpu_data, force_model->rc * force_model->rc); 
            gpu_find_force_sw2(para, force_model->sw2, gpu_data);
            break;
        case 34:  
            gpu_find_force_vashishta_table
            (para, force_model->vas_table, gpu_data);
            break;
        case 40:
            find_neighbor_local(para, gpu_data, force_model->rc * force_model->rc); 
            gpu_find_force_tersoff1(para, force_model, gpu_data);        
            break;
        case 41:
            find_neighbor_local(para, gpu_data, force_model->rc * force_model->rc);
            gpu_find_force_tersoff_1989_2(para, force_model, gpu_data);
            break;
        case 42:
            gpu_find_force_rebo_mos2(para, gpu_data);
            break;
        default: 
            printf("illegal force model\n");
            exit(EXIT_FAILURE);
            break;
    }  
}

