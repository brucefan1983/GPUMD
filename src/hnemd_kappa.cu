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





#include "hnemd_kappa.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"
#include "memory.cuh"
#include "atom.cuh"
#include "error.cuh"


#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH       200
#ifdef USE_DP
    #define ZERO  0.0
    #define KAPPA_UNIT_CONVERSION    1.573769e+5
#else
    #define ZERO  0.0f
    #define KAPPA_UNIT_CONVERSION    1.573769e+5f
#endif




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}




void HNEMD::preprocess_hnemd_kappa(Atom *atom)
{
    if (compute)
    {
        int num = NUM_OF_HEAT_COMPONENTS * output_interval;
        CHECK(cudaMalloc((void**)&heat_all, sizeof(real) * num));
    }
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




static real get_volume(real *box_gpu)
{
    real *box_cpu;
    MY_MALLOC(box_cpu, real, 3);
    cudaMemcpy(box_cpu, box_gpu, sizeof(real) * 3, cudaMemcpyDeviceToHost);
    real volume = box_cpu[0] * box_cpu[1] * box_cpu[2];
    MY_FREE(box_cpu);
    return volume;
}




void HNEMD::process_hnemd_kappa
(
    int step, char *input_dir,
    Atom *atom, Integrate *integrate
)
{
    if (compute)
    {
        int output_flag = ((step+1) % output_interval == 0);
        step %= output_interval;
        gpu_sum_heat<<<5, 1024>>>
        (atom->N, step, atom->heat_per_atom, heat_all);
        if (output_flag)
        {
            int num = NUM_OF_HEAT_COMPONENTS * output_interval;
            int mem = sizeof(real) * num;
            real volume = get_volume(atom->box_length);
            real *heat_cpu;
            MY_MALLOC(heat_cpu, real, num);
            cudaMemcpy(heat_cpu, heat_all, mem, cudaMemcpyDeviceToHost);
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
}




void HNEMD::postprocess_hnemd_kappa(Atom *atom)
{
    if (compute) { cudaFree(heat_all); }
}




