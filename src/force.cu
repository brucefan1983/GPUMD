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
#include "potential.cuh"
#include "force.cuh"
#include "tersoff.cuh"
#include "rebo_mos2.cuh"
#include "vashishta.cuh"
#include "sw.cuh"
#include "pair.cuh"
#include "eam.cuh"
#include "mic_template.cuh"




Force::Force(void)
{
    potential = NULL;
    build_local_neighbor = false;
}




Force::~Force(void)
{
    if(potential) delete potential;
}




void Force::initialize(char *potential_file, Parameters *para)
{
    printf("INFO:  read in potential parameters.\n");
    FILE *fid_potential = my_fopen(potential_file, "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential.in.\n");
        exit(1);
    }
    
    // determine the potential
    if (strcmp(potential_name, "tersoff_1989_1") == 0) 
    { 
         potential = new Tersoff2(fid_potential, para, 1);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "tersoff_1989_2") == 0) 
    { 
         potential = new Tersoff2(fid_potential, para, 2);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "sw_1985") == 0) 
    { 
         potential = new SW2(fid_potential, para, 1);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "sw_1985_2") == 0) 
    { 
         potential = new SW2(fid_potential, para, 2);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "sw_1985_3") == 0) 
    { 
         potential = new SW2(fid_potential, para, 3);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "rebo_mos2") == 0) 
    { 
         potential = new REBO_MOS(para);
         build_local_neighbor = false;
    }
    else if (strcmp(potential_name, "lj1") == 0) 
    { 
         potential = new Pair(fid_potential, para, 0);
         build_local_neighbor = false;
    }
    else if (strcmp(potential_name, "ri") == 0) 
    { 
         potential = new Pair(fid_potential, para, 1);
         build_local_neighbor = false;
    }
    else if (strcmp(potential_name, "eam_zhou_2004_1") == 0) 
    { 
         potential = new EAM_Analytical(fid_potential, para, potential_name);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "eam_dai_2006") == 0) 
    { 
         potential = new EAM_Analytical(fid_potential, para, potential_name);
         build_local_neighbor = true;
    }
    else if (strcmp(potential_name, "vashishta") == 0) 
    { 
         potential = new Vashishta(fid_potential, para, 0);
         build_local_neighbor = false;
    }
    else if (strcmp(potential_name, "vashishta_table") == 0) 
    { 
         potential = new Vashishta(fid_potential, para, 1);
         build_local_neighbor = false;
    }
    else    
    { 
        print_error("illegal potential model.\n"); 
        exit(1); 
    }

    fclose(fid_potential);
    printf("INFO:  potential parameters initialized.\n\n");
}




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




void Force::compute(Parameters *para, GPU_Data *gpu_data)
{
    if (build_local_neighbor) 
    { 
        real cutoff_square = potential->rc * potential->rc;
        find_neighbor_local(para, gpu_data, cutoff_square); 
    }
    potential->compute(para, gpu_data);
}




