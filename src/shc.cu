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
#include "shc.cuh"




//build the look-up table used for recording force and velocity data
static void build_fv_table
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    para->shc.number_of_sections = 1; 
    int count = 0;
    for (int n1 = 0; n1 < para->N; ++n1)
    {
        cpu_data->fv_index[n1] = -1;
        cpu_data->fv_index[n1 + para->N] = -1; 
        int label1 = cpu_data->label[n1];
        for (int i1 = 0; i1 < cpu_data->NN[n1]; ++i1)   
        {       
            int n2 = cpu_data->NL[n1 * para->neighbor.MN + i1]; 
            int label2 = cpu_data->label[n2];
                 
            // old correct version
            if (label1 == para->shc.block_A && label2 == para->shc.block_B)
            {           
                cpu_data->fv_index[n1] = count;
                cpu_data->fv_index[n1 + para->N] = n2;
                count++;
            }
            
        }
    }
}


// allocate memory and initialize for calculating SHC
void preprocess_shc(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{      
    if (para->shc.compute)
    {
        // 2*N data for indexing purpose 
        MY_MALLOC(cpu_data->fv_index, int,  2 * para->N); 
        build_fv_table(para, cpu_data, gpu_data);
        para->shc.number_of_pairs = 0;
        for (int n = 0; n < para->N; n++)
        {
            if (cpu_data->fv_index[n] >= 0) { para->shc.number_of_pairs++; }
        }
        //printf("number_of_pairs = %d\n", para->shc.number_of_pairs);

        // there are 12 data for each pair
        int num1 = para->shc.number_of_pairs * 12; 
        int num2 = num1 * para->shc.M;
        cudaMalloc((void**)&gpu_data->fv_index, sizeof(int)  * 2 * para->N);
        cudaMalloc((void**)&gpu_data->fv,       sizeof(real) * num1);
        cudaMalloc((void**)&gpu_data->fv_all,   sizeof(real) * num2);
        CHECK(cudaMemcpy(gpu_data->fv_index, cpu_data->fv_index, 
            sizeof(int) * 2 * para->N, cudaMemcpyHostToDevice));
    }
}


// Find the time correlation function K(t); GPU version.
static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}

static __global__ void gpu_find_k_time
(
    int Nc, int Nd, int M,int number_of_sections, int number_of_pairs, 
    real *g_fv_all, real *g_k_time_i, real *g_k_time_o
)
{
    //<<<Nc, 128>>>
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (M - 1) / 128 + 1; 

    __shared__ real s_k_time_i[128];
    __shared__ real s_k_time_o[128];
    s_k_time_i[tid] = ZERO;  
    s_k_time_o[tid] = ZERO;

    for (int patch = 0; patch < number_of_patches; ++patch)
    {  
        int m = tid + patch * 128;
        if (m < M)
        {
            int index_0 = (m +   0) * number_of_pairs * 12;
            int index_t = (m + bid) * number_of_pairs * 12;

            for (int np = 0; np < number_of_pairs; np++) // pairs
            {
                real f12x = g_fv_all[index_0 + np * 12 + 0];
                real f12y = g_fv_all[index_0 + np * 12 + 1];
                real f12z = g_fv_all[index_0 + np * 12 + 2];
                real f21x = g_fv_all[index_0 + np * 12 + 3];
                real f21y = g_fv_all[index_0 + np * 12 + 4];
                real f21z = g_fv_all[index_0 + np * 12 + 5];
                real  v1x = g_fv_all[index_t + np * 12 + 6];
                real  v1y = g_fv_all[index_t + np * 12 + 7];
                real  v1z = g_fv_all[index_t + np * 12 + 8];
                real  v2x = g_fv_all[index_t + np * 12 + 9];
                real  v2y = g_fv_all[index_t + np * 12 + 10];
                real  v2z = g_fv_all[index_t + np * 12 + 11];
                real f_dot_v_x = f12x * v2x - f21x * v1x;
                real f_dot_v_y = f12y * v2y - f21y * v1y;
                real f_dot_v_z = f12z * v2z - f21z * v1z;

                s_k_time_i[tid] -= f_dot_v_x + f_dot_v_y;
                s_k_time_o[tid] -= f_dot_v_z;
            }
        }
    }
    __syncthreads();

    if (tid < 64)
    {
        s_k_time_i[tid] += s_k_time_i[tid + 64];
        s_k_time_o[tid] += s_k_time_o[tid + 64];
    }
    __syncthreads();
 
    if (tid < 32)
    {
        warp_reduce(s_k_time_i, tid);
        warp_reduce(s_k_time_o, tid);
    }
   
    if (tid == 0)
    {
        g_k_time_i[bid] = s_k_time_i[0] / (number_of_sections * M);
        g_k_time_o[bid] = s_k_time_o[0] / (number_of_sections * M);
    }
}


// calculate the correlation function K(t)
static void find_k_time
(Files *files, Parameters *para, CPU_Data *cpu_data,GPU_Data *gpu_data)
{
    int number_of_sections = para->shc.number_of_sections;
    int number_of_pairs = para->shc.number_of_pairs;

    int Nd = para->shc.M;
    int Nc = para->shc.Nc;
    int M  = Nd - Nc; 

    // allocate memory for K(t)
    real *k_time_i;
    real *k_time_o;
    MY_MALLOC(k_time_i, real, Nc);
    MY_MALLOC(k_time_o, real, Nc);

    // calculate K(t)
    real *g_k_time_i;
    real *g_k_time_o;
    CHECK(cudaMalloc((void**)&g_k_time_i, sizeof(real) * Nc));
    CHECK(cudaMalloc((void**)&g_k_time_o, sizeof(real) * Nc));
    gpu_find_k_time<<<Nc, 128>>>
    (
        Nc, Nd, M, number_of_sections, number_of_pairs, 
        gpu_data->fv_all, g_k_time_i, g_k_time_o
    );
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(k_time_i, g_k_time_i, 
        sizeof(real) * Nc, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(k_time_o, g_k_time_o, 
        sizeof(real) * Nc, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(g_k_time_i));
    CHECK(cudaFree(g_k_time_o)); 

    // output the results
    FILE *fid;
    fid = my_fopen(files->shc, "a");
    for (int nc = 0; nc < Nc; nc++)
    { 
        fprintf(fid, "%25.15e%25.15e\n", k_time_i[nc], k_time_o[nc]);
    }
    fflush(fid);
    fclose(fid);

    // free memory
    MY_FREE(k_time_i);
    MY_FREE(k_time_o);
} 


void process_shc
(
    int step, Files *files, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    if (para->shc.compute)
    { 
        int step_ref = para->shc.sample_interval * para->shc.M;
        int fv_size = para->shc.number_of_pairs * 12;
        int fv_memo = fv_size * sizeof(real);
        
        // sample fv data every "sample_interval" steps
        if ((step + 1) % para->shc.sample_interval == 0)
        {
            int offset = 
                (
                    (step - (step/step_ref)*step_ref + 1) 
                    / para->shc.sample_interval - 1
                ) * fv_size;
            CHECK(cudaMemcpy(gpu_data->fv_all + offset, 
            gpu_data->fv, fv_memo, cudaMemcpyDeviceToDevice));
        }

        // calculate the correlation function every "sample_interval * M" steps
        if ((step + 1) % step_ref == 0)
        {       
            find_k_time(files, para, cpu_data, gpu_data);
        }
    }
}


void postprocess_shc(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    if (para->shc.compute)
    {
        MY_FREE(cpu_data->fv_index);
        CHECK(cudaFree(gpu_data->fv_index));
        CHECK(cudaFree(gpu_data->fv));
        CHECK(cudaFree(gpu_data->fv_all));  
    }
}

