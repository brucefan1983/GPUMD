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
#include "memory.cuh"
#define FILE_NAME_LENGTH      200

typedef unsigned long long uint64;




static FILE *my_fopen(const char *filename, const char *mode)
{
    FILE *fid = fopen(filename, mode);
    if (fid == NULL) 
    {
        printf ("Failed to open %s!\n", filename);
        printf ("%s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    return fid;
}




//build the look-up table used for recording force and velocity data
static void build_fv_table
(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    para->shc.number_of_sections = 1; 
    para->shc.number_of_pairs = 0;
    for (int n1 = 0; n1 < para->N; ++n1)
    {
    	if (cpu_data->a_map[n1] != -1)
    	{
    		// need loop to initialize all fv_table elements to -1
    		for (int n2 = 0; n2 <  para->N; ++n2)
    		{
    			if (cpu_data->b_map[n2] != -1)
    			{
    				cpu_data->fv_index[cpu_data->a_map[n1] * *(cpu_data->count_b) + cpu_data->b_map[n2]] = -1;
    			}
    		}
    		// Now set neighbors to correct value
    		for (int i1 = 0; i1 < cpu_data->NN[n1]; ++i1)
    		{
    			int n2 = cpu_data->NL[n1 * para->neighbor.MN + i1];
    			if (cpu_data->b_map[n2] != -1)
    			{
    				cpu_data->fv_index[cpu_data->a_map[n1] * *(cpu_data->count_b) + cpu_data->b_map[n2]] =
    				    						para->shc.number_of_pairs;
    				para->shc.number_of_pairs++;
    			}
    		}
    	}
    }
}

// allocate memory and initialize for calculating SHC
void preprocess_shc(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{      
    if (para->shc.compute)
    {
    	//build map from N atoms to A and B labeled atoms
    	int c_a = 0; int c_b = 0;
    	MY_MALLOC(cpu_data->a_map, int, para->N);
    	MY_MALLOC(cpu_data->b_map, int, para->N);
    	for (int n = 0; n < para->N; n++)
    	{
    		cpu_data->a_map[n] = -1;
    		cpu_data->b_map[n] = -1;
    		if (cpu_data->label[n] == para->shc.block_A)
    		{
    			cpu_data->a_map[n] = c_a;
    			c_a++;
    		}
    		else if (cpu_data->label[n] == para->shc.block_B)
    		{
    			cpu_data->b_map[n] = c_b;
    			c_b++;
    		}
    	}
    	MY_MALLOC(cpu_data->count_a, int, 1); // count of atoms in group A
		MY_MALLOC(cpu_data->count_b, int, 1);
    	*(cpu_data->count_a) = c_a;
		*(cpu_data->count_b) = c_b;
		MY_MALLOC(cpu_data->fv_index, int,  *(cpu_data->count_a) * *(cpu_data->count_b));
		build_fv_table(para, cpu_data, gpu_data);

		// there are 12 data for each pair
		uint64 num1 = para->shc.number_of_pairs * 12;
		uint64 num2 = num1 * para->shc.M;

		cudaMalloc((void**)&gpu_data->a_map, sizeof(int) * para->N);
		cudaMalloc((void**)&gpu_data->b_map, sizeof(int) * para->N);
		cudaMalloc((void**)&gpu_data->count_a, sizeof(int));
		cudaMalloc((void**)&gpu_data->count_b, sizeof(int));
		cudaMalloc((void**)&gpu_data->fv_index, sizeof(int)  * *(cpu_data->count_a) * *(cpu_data->count_b));
		cudaMalloc((void**)&gpu_data->fv,       sizeof(real) * num1);
		cudaMalloc((void**)&gpu_data->fv_all,   sizeof(real) * num2);

		CHECK(cudaMemcpy(gpu_data->fv_index, cpu_data->fv_index,
				sizeof(int)  * *(cpu_data->count_a) * *(cpu_data->count_b), cudaMemcpyHostToDevice));

		CHECK(cudaMemcpy(gpu_data->a_map, cpu_data->a_map,
				sizeof(int) * para->N, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(gpu_data->b_map, cpu_data->b_map,
				sizeof(int) * para->N, cudaMemcpyHostToDevice));

		CHECK(cudaMemcpy(gpu_data->count_a, cpu_data->count_a,
				sizeof(int), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(gpu_data->count_b, cpu_data->count_b,
				sizeof(int), cudaMemcpyHostToDevice));

    }
}


// Find the time correlation function K(t); GPU version.
static __device__ void warp_reduce(volatile real *s, uint64 t)
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
    uint64 tid = threadIdx.x;
    uint64 bid = blockIdx.x;
    uint64 number_of_patches = (M - 1) / 128 + 1;

    __shared__ real s_k_time_i[128];
    __shared__ real s_k_time_o[128];
    s_k_time_i[tid] = ZERO;  
    s_k_time_o[tid] = ZERO;

    for (uint64 patch = 0; patch < number_of_patches; ++patch)
    {  
        uint64 m = tid + patch * 128;
        if (m < M)
        {
            uint64 index_0 = (m +   0) * number_of_pairs * 12;
            uint64 index_t = (m + bid) * number_of_pairs * 12;

            for (uint64 np = 0; np < number_of_pairs; np++) // pairs
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
(char *input_dir, Parameters *para, CPU_Data *cpu_data,GPU_Data *gpu_data)
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
    char file_shc[FILE_NAME_LENGTH];
    strcpy(file_shc, input_dir);
    strcat(file_shc, "/shc.out");
    FILE *fid = my_fopen(file_shc, "a");
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
    int step, char *input_dir, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    if (para->shc.compute)
    { 
        uint64 step_ref = para->shc.sample_interval * para->shc.M;
        uint64 fv_size = para->shc.number_of_pairs * 12;
        uint64 fv_memo = fv_size * sizeof(real);
        
        // sample fv data every "sample_interval" steps
        if ((step + 1) % para->shc.sample_interval == 0)
        {
            uint64 offset =
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
            find_k_time(input_dir, para, cpu_data, gpu_data);
        }
    }
}


void postprocess_shc(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    if (para->shc.compute)
    {
        MY_FREE(cpu_data->fv_index);
        MY_FREE(cpu_data->a_map);
        MY_FREE(cpu_data->b_map);
        MY_FREE(cpu_data->count_a);
        MY_FREE(cpu_data->count_b);
        CHECK(cudaFree(gpu_data->fv_index));
        CHECK(cudaFree(gpu_data->a_map));
        CHECK(cudaFree(gpu_data->b_map));
        CHECK(cudaFree(gpu_data->count_a));
        CHECK(cudaFree(gpu_data->count_b));
        CHECK(cudaFree(gpu_data->fv));
        CHECK(cudaFree(gpu_data->fv_all));
    }
}

