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
#include "mic.inc"




Force::Force(void)
{
    for (int m = 0; m < MAX_NUM_OF_POTENTIALS; m++)
    {
        potential[m] = NULL;
    }
    num_of_potentials = 0;
    interlayer_only = 0;
    rc_max = ZERO;
}




Force::~Force(void)
{
    for (int m = 0; m < num_of_potentials; m++)
    {
        delete potential[m];
        potential[m] = NULL;
    }

    if (interlayer_only) 
    {
        cudaFree(layer_label); 
    }
}




void Force::initialize_one_potential(Parameters *para, int m)
{
    FILE *fid_potential = my_fopen(file_potential[m], "r");
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
        potential[m] = new Tersoff2(fid_potential, para, 1);
        if (para->number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "tersoff_1989_2") == 0) 
    { 
        potential[m] = new Tersoff2(fid_potential, para, 2);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "sw_1985") == 0) 
    { 
        potential[m] = new SW2(fid_potential, para, 1);
        if (para->number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "sw_1985_2") == 0) 
    { 
        potential[m] = new SW2(fid_potential, para, 2);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "sw_1985_3") == 0) 
    { 
        potential[m] = new SW2(fid_potential, para, 3);
        if (para->number_of_types != 3) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "rebo_mos2") == 0) 
    { 
        potential[m] = new REBO_MOS(para);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj1") == 0)
    { 
        potential[m] = new Pair(fid_potential, para, 1);
        if (para->number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj2") == 0)
    { 
        potential[m] = new Pair(fid_potential, para, 2);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj3") == 0)
    { 
        potential[m] = new Pair(fid_potential, para, 3);
        if (para->number_of_types != 3) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj4") == 0)
    { 
        potential[m] = new Pair(fid_potential, para, 4);
        if (para->number_of_types != 4) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj5") == 0)
    { 
        potential[m] = new Pair(fid_potential, para, 5);
        if (para->number_of_types != 5) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "ri") == 0)
    { 
        potential[m] = new Pair(fid_potential, para, 0);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "eam_zhou_2004_1") == 0) 
    { 
        potential[m] = new EAM(fid_potential, para, potential_name);
        if (para->number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "eam_dai_2006") == 0) 
    { 
        potential[m] = new EAM(fid_potential, para, potential_name);
        if (para->number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "vashishta") == 0) 
    { 
        potential[m] = new Vashishta(fid_potential, para, 0);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "vashishta_table") == 0) 
    { 
        potential[m] = new Vashishta(fid_potential, para, 1);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else    
    { 
        print_error("illegal potential model.\n"); 
        exit(1); 
    }

    potential[m]->N1 = 0;
    potential[m]->N2 = para->N;

    fclose(fid_potential);
}




void Force::initialize_two_body_potential(Parameters *para)
{
    FILE *fid_potential = my_fopen(file_potential[0], "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential file.\n");
        exit(1);
    }
    
    // determine the potential
    if (strcmp(potential_name, "lj1") == 0)
    { 
        potential[0] = new Pair(fid_potential, para, 1);
        if (para->number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj2") == 0)
    { 
        potential[0] = new Pair(fid_potential, para, 2);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj3") == 0)
    { 
        potential[0] = new Pair(fid_potential, para, 3);
        if (para->number_of_types != 3) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj4") == 0)
    { 
        potential[0] = new Pair(fid_potential, para, 4);
        if (para->number_of_types != 4) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "lj5") == 0)
    { 
        potential[0] = new Pair(fid_potential, para, 5);
        if (para->number_of_types != 5) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "ri") == 0)
    { 
        potential[0] = new Pair(fid_potential, para, 0);
        if (para->number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else    
    { 
        print_error("illegal two-body potential model.\n"); 
        exit(1); 
    }

    potential[0]->N1 = 0;
    potential[0]->N2 = para->N;

    fclose(fid_potential);
}




void Force::initialize_many_body_potential
(Parameters *para, CPU_Data *cpu_data, int m)
{
    FILE *fid_potential = my_fopen(file_potential[m], "r");
    char potential_name[20];
    int count = fscanf(fid_potential, "%s", potential_name);
    if (count != 1) 
    {
        print_error("reading error for potential file.\n");
        exit(1);
    }
    
    int number_of_types = type_end[m] - type_begin[m] + 1;
    // determine the potential
    if (strcmp(potential_name, "tersoff_1989_1") == 0) 
    { 
        potential[m] = new Tersoff2(fid_potential, para, 1);
        if (number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "tersoff_1989_2") == 0) 
    { 
        potential[m] = new Tersoff2(fid_potential, para, 2);
        if (number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "sw_1985") == 0) 
    { 
        potential[m] = new SW2(fid_potential, para, 1);
        if (number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "sw_1985_2") == 0) 
    { 
        potential[m] = new SW2(fid_potential, para, 2);
        if (number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "sw_1985_3") == 0) 
    { 
        potential[m] = new SW2(fid_potential, para, 3);
        if (number_of_types != 3) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "rebo_mos2") == 0) 
    { 
        potential[m] = new REBO_MOS(para);
        if (number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "eam_zhou_2004_1") == 0) 
    { 
        potential[m] = new EAM(fid_potential, para, potential_name);
        if (number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "eam_dai_2006") == 0) 
    { 
        potential[m] = new EAM(fid_potential, para, potential_name);
        if (number_of_types != 1) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "vashishta") == 0) 
    { 
        potential[m] = new Vashishta(fid_potential, para, 0);
        if (number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else if (strcmp(potential_name, "vashishta_table") == 0) 
    { 
        potential[m] = new Vashishta(fid_potential, para, 1);
        if (number_of_types != 2) 
            print_error("number of types does not match potential file.\n");
    }
    else    
    { 
        print_error("illegal many-body potential model.\n"); 
        exit(1); 
    }

    potential[m]->N1 = 0;
    potential[m]->N2 = 0;
    for (int n = 0; n < type_begin[m]; ++n)
    {
        potential[m]->N1 += cpu_data->type_size[n];
    }
    for (int n = 0; n <= type_end[m]; ++n)
    {
        potential[m]->N2 += cpu_data->type_size[n];
    }
    printf
    (
        "       applies to atoms [%d, %d) from type %d to type %d.\n",
        potential[m]->N1, potential[m]->N2, type_begin[m], type_end[m]
    );

    fclose(fid_potential);
}




void Force::initialize
(char *input_dir, Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    // a single potential
    if (num_of_potentials == 1) 
    {
        initialize_one_potential(para, 0);
        rc_max = potential[0]->rc;
    }
    else // hybrid potentials
    {
        // the two-body part
        initialize_two_body_potential(para);
        rc_max = potential[0]->rc;

        // if the intralayer interactions are to be excluded
        if (interlayer_only) 
        {
            int *layer_label_cpu;
            MY_MALLOC(layer_label_cpu, int, para->N); 
            char file_layer_label[FILE_NAME_LENGTH];
            strcpy(file_layer_label, input_dir);
            strcat(file_layer_label, "/layer.in");
            FILE *fid = my_fopen(file_layer_label, "r");
            for (int n = 0; n < para->N; ++n)
            {
                int count = fscanf(fid, "%d", &layer_label_cpu[n]);
                if (count != 1) print_error("reading error for layer.in");
            }
            fclose(fid);

            int memory = sizeof(int)*para->N;
            cudaMalloc((void**)&layer_label, memory);
            cudaMemcpy(layer_label, layer_label_cpu, memory, cudaMemcpyHostToDevice);
            MY_FREE(layer_label_cpu); 
        }

        // the many-body part
        for (int m = 1; m < num_of_potentials; m++)
        {
            initialize_many_body_potential(para, cpu_data, m);
            if (rc_max < potential[m]->rc) rc_max = potential[m]->rc;

            // check the atom types in xyz.in
            for (int n = potential[m]->N1; n < potential[m]->N2; ++n)
            {
                if (cpu_data->type[n] < type_begin[m] || cpu_data->type[n] > type_end[m])
                {
                    printf("ERROR: ");
                    printf
                    (
                        "atom type for many-body potential # %d not from %d to %d.", 
                        m, type_begin[m], type_begin[m]
                    );
                    exit(1);
                }

                // the local type always starts from 0
                cpu_data->type_local[n] -= type_begin[m];
            }
        }
        
        // copy the local atom type to the GPU
        cudaMemcpy
        (
            gpu_data->type_local, cpu_data->type_local, 
            sizeof(int) * para->N, cudaMemcpyHostToDevice
        );
    }
}




// Construct the local neighbor list from the global one (Kernel)
template<int check_layer_label, int check_type>
static __global__ void gpu_find_neighbor_local
(
    int pbc_x, int pbc_y, int pbc_z, int type_begin, int type_end, int *type,
    int N, int N1, int N2, real cutoff_square, real *box_length,
    int *NN, int *NL, int *NN_local, int *NL_local, int *layer_label,
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
    int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
    int count = 0;
    real lx = box_length[0];
    real ly = box_length[1];
    real lz = box_length[2];

    int layer_n1;

    if (n1 >= N1 && n1 < N2)
    {  
        int neighbor_number = NN[n1];

        if (check_layer_label) layer_n1 = layer_label[n1];

        real x1 = LDG(x, n1);   
        real y1 = LDG(y, n1);
        real z1 = LDG(z, n1);  
        for (int i1 = 0; i1 < neighbor_number; ++i1)
        {   
            int n2 = NL[n1 + N * i1];

            // exclude intralayer interactions if needed
            if (check_layer_label) 
            {
                if (layer_n1 == layer_label[n2]) continue;
            }

            // only include neighors with the correct types
            if (check_type)
            {
                int type_n2 = type[n2];
                if (type_n2 < type_begin || type_n2 > type_end) continue;
            }

            real x12  = LDG(x, n2) - x1;
            real y12  = LDG(y, n2) - y1;
            real z12  = LDG(z, n2) - z1;
            dev_apply_mic(pbc_x, pbc_y, pbc_z, x12, y12, z12, lx, ly, lz);
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
void Force::find_neighbor_local(Parameters *para, GPU_Data *gpu_data, int m)
{  
    int type1 = type_begin[m];
    int type2 = type_end[m];
    int N = para->N;
    int N1 = potential[m]->N1;
    int N2 = potential[m]->N2;
    int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1; 
    int pbc_x = para->pbc_x;
    int pbc_y = para->pbc_y;
    int pbc_z = para->pbc_z;
    int *NN = gpu_data->NN;
    int *NL = gpu_data->NL;
    int *NN_local = gpu_data->NN_local;
    int *NL_local = gpu_data->NL_local;
    int *type = gpu_data->type; // global type
    real rc2 = potential[m]->rc * potential[m]->rc;
    real *x = gpu_data->x;
    real *y = gpu_data->y;
    real *z = gpu_data->z;
    real *box = gpu_data->box_length;
      
    if (0 == m)
    {
        if (interlayer_only)
        {
            gpu_find_neighbor_local<1, 0><<<grid_size, BLOCK_SIZE>>>
            (
                pbc_x, pbc_y, pbc_z, type1, type2, type, N, N1, N2, 
                rc2, box, NN, NL, NN_local, NL_local, layer_label, x, y, z
            );
        }
        else
        {
            gpu_find_neighbor_local<0, 0><<<grid_size, BLOCK_SIZE>>>
            (
                pbc_x, pbc_y, pbc_z, type1, type2, type, N, N1, N2, 
                rc2, box, NN, NL, NN_local, NL_local, layer_label, x, y, z
            );
        }
    }
    else
    {
        gpu_find_neighbor_local<0, 1><<<grid_size, BLOCK_SIZE>>>
        (
            pbc_x, pbc_y, pbc_z, type1, type2, type, N, N1, N2, 
            rc2, box, NN, NL, NN_local, NL_local, layer_label, x, y, z
        );
    }
}




static __global__ void initialize_properties
(
    int compute_shc,
    int N, int M, real *g_fx, real *g_fy, real *g_fz, real *g_pe,
    real *g_sx, real *g_sy, real *g_sz, real *g_h, real *g_fv
)
{
    //<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {  
        g_fx[n1] = ZERO;
        g_fy[n1] = ZERO;
        g_fz[n1] = ZERO;
        g_sx[n1] = ZERO;
        g_sy[n1] = ZERO;
        g_sz[n1] = ZERO;
        g_pe[n1] = ZERO;
        g_h[n1 + 0 * N] = ZERO;
        g_h[n1 + 1 * N] = ZERO;
        g_h[n1 + 2 * N] = ZERO;
        g_h[n1 + 3 * N] = ZERO;
        g_h[n1 + 4 * N] = ZERO;
    }
    if (compute_shc && n1 < M)
    {  
        g_fv[n1] = ZERO;
    }
}




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}




// get the total force
static __global__ void gpu_sum_force
(int N, real *g_fx, real *g_fy, real *g_fz, real *g_f)
{
    //<<<3, MAX_THREAD>>>

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int patch, n;
    int number_of_patches = (N - 1) / 1024 + 1; 

    switch (bid)
    {
        case 0:
            __shared__ real s_fx[1024];
            s_fx[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N) s_fx[tid] += g_fx[n]; 
            }
            __syncthreads();
            if (tid < 512) s_fx[tid] += s_fx[tid + 512]; __syncthreads();
            if (tid < 256) s_fx[tid] += s_fx[tid + 256]; __syncthreads();
            if (tid < 128) s_fx[tid] += s_fx[tid + 128]; __syncthreads();
            if (tid <  64) s_fx[tid] += s_fx[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_fx, tid); 
            if (tid ==  0) { g_f[0] = s_fx[0]; }                  
            break;
        case 1:
            __shared__ real s_fy[1024];
            s_fy[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N) s_fy[tid] += g_fy[n]; 
            }
            __syncthreads();
            if (tid < 512) s_fy[tid] += s_fy[tid + 512]; __syncthreads();
            if (tid < 256) s_fy[tid] += s_fy[tid + 256]; __syncthreads();
            if (tid < 128) s_fy[tid] += s_fy[tid + 128]; __syncthreads();
            if (tid <  64) s_fy[tid] += s_fy[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_fy, tid); 
            if (tid ==  0) { g_f[1] = s_fy[0]; }                  
            break;
        case 2:
            __shared__ real s_fz[1024];
            s_fz[tid] = ZERO;
            for (patch = 0; patch < number_of_patches; ++patch)
            { 
                n = tid + patch * 1024;
                if (n < N) s_fz[tid] += g_fz[n]; 
            }
            __syncthreads();
            if (tid < 512) s_fz[tid] += s_fz[tid + 512]; __syncthreads();
            if (tid < 256) s_fz[tid] += s_fz[tid + 256]; __syncthreads();
            if (tid < 128) s_fz[tid] += s_fz[tid + 128]; __syncthreads();
            if (tid <  64) s_fz[tid] += s_fz[tid + 64];  __syncthreads();
            if (tid <  32) warp_reduce(s_fz, tid); 
            if (tid ==  0) { g_f[2] = s_fz[0]; }                  
            break;
    }
}




// correct the total force
static __global__ void gpu_correct_force
(int N, real *g_fx, real *g_fy, real *g_fz, real *g_f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {  
        g_fx[i] -= g_f[0] / N;
        g_fy[i] -= g_f[1] / N;
        g_fz[i] -= g_f[2] / N;
    }
}




void Force::compute(Parameters *para, GPU_Data *gpu_data)
{
    initialize_properties<<<(para->N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>
    (
        para->shc.compute,
        para->N, para->shc.number_of_pairs * 12,
        gpu_data->fx, gpu_data->fy, gpu_data->fz, 
        gpu_data->potential_per_atom,  
        gpu_data->virial_per_atom_x,
        gpu_data->virial_per_atom_y,
        gpu_data->virial_per_atom_z,
        gpu_data->heat_per_atom, gpu_data->fv
    );

    for (int m = 0; m < num_of_potentials; m++)
    {
        // first build a local neighbor list
        find_neighbor_local(para, gpu_data, m);
        // and then calculate the forces and related quantities
        potential[m]->compute(para, gpu_data);
    }

    // correct the force when using the HNEMD method
    if (para->hnemd.compute)
    {
        real *ftot; // total force vector of the system
        cudaMalloc((void**)&ftot, sizeof(real) * 3);
        gpu_sum_force<<<3, 1024>>>(para->N, gpu_data->fx, gpu_data->fy, gpu_data->fz, ftot);

        int grid_size = (para->N - 1) / BLOCK_SIZE + 1;
        gpu_correct_force<<<grid_size, BLOCK_SIZE>>>
        (para->N, gpu_data->fx, gpu_data->fy, gpu_data->fz, ftot);

        cudaFree(ftot);
    }

}




