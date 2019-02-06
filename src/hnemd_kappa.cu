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
Calculate the thermal conductivity using the HNEMD method.
Reference:
[1] arXiv:1805.00277
------------------------------------------------------------------------------*/


#include "hnemd_kappa.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "atom.cuh"
#include "error.cuh"

#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH       200


static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}


void HNEMD::preprocess(Atom *atom)
{
    if (!compute) return;
    int num = NUM_OF_HEAT_COMPONENTS * output_interval;
    CHECK(cudaMalloc((void**)&heat_all, sizeof(real) * num));
}


static __global__ void gpu_sum_heat
(int N, int step, real *g_heat, real *g_heat_sum)
{
    // <<<5, 1024>>> 
    int tid = threadIdx.x; 
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1;
    __shared__ real s_data[1024];  
    s_data[tid] = ZERO;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024; 
        if (n < N) { s_data[tid] += g_heat[n + N * bid]; }
    }
    __syncthreads();
    if (tid < 512) { s_data[tid] += s_data[tid + 512]; } __syncthreads();
    if (tid < 256) { s_data[tid] += s_data[tid + 256]; } __syncthreads();
    if (tid < 128) { s_data[tid] += s_data[tid + 128]; } __syncthreads();
    if (tid <  64) { s_data[tid] += s_data[tid +  64]; } __syncthreads();
    if (tid <  32) { warp_reduce(s_data, tid);         } 
    if (tid ==  0) { g_heat_sum[step*NUM_OF_HEAT_COMPONENTS+bid] = s_data[0]; }
}


void HNEMD::process(int step, char *input_dir, Atom *atom, Integrate *integrate)
{
    if (!compute) return;
    int output_flag = ((step+1) % output_interval == 0);
    step %= output_interval;
    gpu_sum_heat<<<5, 1024>>>(atom->N, step, atom->heat_per_atom, heat_all);
    CUDA_CHECK_KERNEL
    if (output_flag)
    {
        int num = NUM_OF_HEAT_COMPONENTS * output_interval;
        int mem = sizeof(real) * num;
        real volume = atom->box.get_volume();
        real *heat_cpu;
        MY_MALLOC(heat_cpu, real, num);
        CHECK(cudaMemcpy(heat_cpu, heat_all, mem, cudaMemcpyDeviceToHost));
        real kappa[NUM_OF_HEAT_COMPONENTS];
        for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) 
        {
            kappa[n] = ZERO;
        }
        for (int m = 0; m < output_interval; m++)
        {
            for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++)
            {
                kappa[n] += heat_cpu[m * NUM_OF_HEAT_COMPONENTS + n];
            }
        }
        real factor = KAPPA_UNIT_CONVERSION / output_interval;
        factor /= (volume * integrate->ensemble->temperature * fe);

        char file_kappa[FILE_NAME_LENGTH];
        strcpy(file_kappa, input_dir);
        strcat(file_kappa, "/kappa.out");
        FILE *fid = fopen(file_kappa, "a");
        for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++)
        {
            fprintf(fid, "%25.15f", kappa[n] * factor);
        }
        fprintf(fid, "\n");
        fflush(fid);  
        fclose(fid);
        MY_FREE(heat_cpu);
    }
}


void HNEMD::postprocess(Atom *atom)
{
    if (compute) { CHECK(cudaFree(heat_all)); }
}


