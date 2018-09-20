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
#include "hac.cuh"
#include "integrate.cuh"
#include "ensemble.cuh"




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}




//Allocate memory for recording heat current data
void preprocess_hac(Parameters *para, CPU_Data *cpu_data, GPU_Data *gpu_data)
{
    if (para->hac.compute)
    {
        int num = NUM_OF_HEAT_COMPONENTS * para->number_of_steps 
                / para->hac.sample_interval;
        CHECK(cudaMalloc((void**)&gpu_data->heat_all, sizeof(real) * num));
    }
}




// sum up the per-atom heat current to get the total heat current
static __global__ void gpu_sum_heat
(
    int N, int Nd, int nd, real *g_vx, real *g_vy, real *g_vz, 
    real *g_mass, real *g_potential, real *g_heat, 
    real *g_heat_all, real *g_heat_tmp
)
{
    // <<<8, 1024>>> 

    int tid = threadIdx.x; 
    int number_of_patches = (N - 1) / 1024 + 1;

    __shared__ real s_data[1024];  
    s_data[tid] = ZERO;
 
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024; 
        if (n < N) 
        { 
            if (blockIdx.x < NUM_OF_HEAT_COMPONENTS)
            {
                s_data[tid] += g_heat[n + N * blockIdx.x];
            }
            else
            {
                real vx = g_vx[n];
                real vy = g_vy[n];
                real vz = g_vz[n];
                real v_square = vx * vx + vy * vy + vz * vz;
                real energy = g_mass[n] * v_square * HALF + g_potential[n];
                if (blockIdx.x == 5) s_data[tid] += vx * energy;
                if (blockIdx.x == 6) s_data[tid] += vy * energy;
                if (blockIdx.x == 7) s_data[tid] += vz * energy;
            }
        }
    }

    __syncthreads();
    if (tid < 512) { s_data[tid] += s_data[tid + 512]; } __syncthreads();
    if (tid < 256) { s_data[tid] += s_data[tid + 256]; } __syncthreads();
    if (tid < 128) { s_data[tid] += s_data[tid + 128]; } __syncthreads();
    if (tid <  64) { s_data[tid] += s_data[tid +  64]; } __syncthreads();
    if (tid <  32) { warp_reduce(s_data, tid);         } 
    if (tid ==  0) 
    { 
        g_heat_tmp[blockIdx.x] = s_data[0];
        if (blockIdx.x < NUM_OF_HEAT_COMPONENTS)
            g_heat_all[nd + Nd * blockIdx.x] = s_data[0];
    }
}




// sample heat current data for HAC calculations.
void sample_hac
(
    int step, char *input_dir, Parameters *para, 
    CPU_Data *cpu_data, GPU_Data *gpu_data
)
{
    if (para->hac.compute)
    { 
        if (step % para->hac.sample_interval == 0)
        {   
            // get the total heat current from the per-atom heat current
            int nd = step / para->hac.sample_interval;
            int Nd = para->number_of_steps / para->hac.sample_interval;
            int M = NUM_OF_HEAT_COMPONENTS + DIM;
            real *gpu_heat;
            CHECK(cudaMalloc((void**)&gpu_heat, sizeof(real) * M));
            gpu_sum_heat<<<M, 1024>>>
            (
                para->N, Nd, nd, gpu_data->vx, gpu_data->vy, gpu_data->vz,
                gpu_data->mass, gpu_data->potential_per_atom,
                gpu_data->heat_per_atom, gpu_data->heat_all, gpu_heat
            );
#ifdef HEAT_CURRENT
            // dump the heat current components
            char file_heat[FILE_NAME_LENGTH];
            strcpy(file_heat, input_dir);
            strcat(file_heat, "/heat_current.out");
            FILE *fid = fopen(file_heat, "a");
            real *cpu_heat;
            MY_MALLOC(cpu_heat, real, M);
            CHECK(cudaMemcpy(cpu_heat, gpu_heat, sizeof(real) * M, 
                cudaMemcpyDeviceToHost));
            for (int m = 0; m < M; ++m)
                fprintf(fid, "%25.15e", cpu_heat[m]);
            fprintf(fid, "\n");
            fflush(fid);  
            fclose(fid);
            MY_FREE(cpu_heat);
#endif
            cudaFree(gpu_heat);
        }
    }
}




// Calculate the Heat current Auto-Correlation function (HAC) 
__global__ void gpu_find_hac(int Nc, int Nd, real *g_heat, real *g_hac)
{
    //<<<Nc, 128>>>

    __shared__ real s_hac_xi[128];
    __shared__ real s_hac_xo[128];
    __shared__ real s_hac_yi[128];
    __shared__ real s_hac_yo[128];
    __shared__ real s_hac_z[128];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int M = Nd - Nc;
    int number_of_patches = M / 128; 
    int number_of_data = number_of_patches * 128; 

    s_hac_xi[tid] = ZERO;
    s_hac_xo[tid] = ZERO;
    s_hac_yi[tid] = ZERO;
    s_hac_yo[tid] = ZERO;
    s_hac_z[tid]  = ZERO;

    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int index = tid + patch * 128;
        s_hac_xi[tid] += g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 0]
                       + g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 1];
        s_hac_xo[tid] += g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 1]
                       + g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 0];
        s_hac_yi[tid] += g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 2]
                       + g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 3];
        s_hac_yo[tid] += g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 3]
                       + g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 2];
        s_hac_z[tid]  += g_heat[index + Nd * 4] * g_heat[index + bid + Nd * 4];
    }
    __syncthreads();

    if (tid < 64)
    {
        s_hac_xi[tid] += s_hac_xi[tid + 64];
        s_hac_xo[tid] += s_hac_xo[tid + 64];
        s_hac_yi[tid] += s_hac_yi[tid + 64];
        s_hac_yo[tid] += s_hac_yo[tid + 64];
        s_hac_z[tid]  += s_hac_z[tid  + 64];
    }
    __syncthreads();
 
    if (tid < 32)
    {
        warp_reduce(s_hac_xi, tid); 
        warp_reduce(s_hac_xo, tid);  
        warp_reduce(s_hac_yi, tid); 
        warp_reduce(s_hac_yo, tid);  
        warp_reduce(s_hac_z,  tid);
    }
   
    if (tid == 0)
    {
        g_hac[bid + Nc * 0] = s_hac_xi[0] / number_of_data;
        g_hac[bid + Nc * 1] = s_hac_xo[0] / number_of_data;
        g_hac[bid + Nc * 2] = s_hac_yi[0] / number_of_data;
        g_hac[bid + Nc * 3] = s_hac_yo[0] / number_of_data;
        g_hac[bid + Nc * 4] = s_hac_z[0]  / number_of_data;
    }
}




// Calculate the Running Thermal Conductivity (RTC) from the HAC
static void find_rtc(int Nc, real factor, real *hac, real *rtc)
{
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++)
    {
        for (int nc = 1; nc < Nc; nc++)  
        {
            int index = Nc * k + nc;
            rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index])*factor;
        }
    }
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




// Calculate 
// (1) HAC = Heat current Auto-Correlation and 
// (2) RTC = Running Thermal Conductivity
static void find_hac_kappa
(
    char *input_dir, Parameters *para, CPU_Data *cpu_data, 
    GPU_Data *gpu_data, Integrate *integrate
)
{
    // rename variables
    int number_of_steps = para->number_of_steps;
    int sample_interval = para->hac.sample_interval;
    int Nc = para->hac.Nc;
    real temperature = para->temperature2;
    real time_step = para->time_step;

    // other parameters
    int Nd = number_of_steps / sample_interval;
    real dt = time_step * sample_interval;
    real dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

    // major data
    real *hac;
    real *rtc;
    MY_MALLOC(hac, real, Nc * NUM_OF_HEAT_COMPONENTS);
    MY_MALLOC(rtc, real, Nc * NUM_OF_HEAT_COMPONENTS);
    
    for (int nc = 0; nc < Nc * NUM_OF_HEAT_COMPONENTS; nc++) 
    { hac[nc] = rtc[nc] = 0.0; }

    real *g_hac;
    CHECK
    (cudaMalloc((void**)&g_hac, sizeof(real) * Nc * NUM_OF_HEAT_COMPONENTS));

    // Here, the block size is fixed to 128, which is a good choice
    gpu_find_hac<<<Nc, 128>>>(Nc, Nd, gpu_data->heat_all, g_hac);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(hac, g_hac, sizeof(real) * Nc * NUM_OF_HEAT_COMPONENTS, 
        cudaMemcpyDeviceToHost));
    CHECK(cudaFree(g_hac));

    real volume = get_volume(gpu_data->box_length);
    real factor = dt * 0.5 / (K_B * temperature * temperature * volume);
    factor *= KAPPA_UNIT_CONVERSION;
 
    find_rtc(Nc, factor, hac, rtc);

    char file_hac[FILE_NAME_LENGTH];
    strcpy(file_hac, input_dir);
    strcat(file_hac, "/hac.out");
    FILE *fid = fopen(file_hac, "a");
    int number_of_output_data = Nc / para->hac.output_interval;
    for (int nd = 0; nd < number_of_output_data; nd++)
    {
        int nc = nd * para->hac.output_interval;
        real hac_ave[NUM_OF_HEAT_COMPONENTS] = {ZERO};
        real rtc_ave[NUM_OF_HEAT_COMPONENTS] = {ZERO};
        for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++)
        {
            for (int m = 0; m < para->hac.output_interval; m++)
            {
                int count = Nc * k + nc + m;
                hac_ave[k] += hac[count];
                rtc_ave[k] += rtc[count];
            }
        }
        for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++)
        {
            hac_ave[m] /= para->hac.output_interval;
            rtc_ave[m] /= para->hac.output_interval;
        }
        fprintf
        (fid, "%25.15e", (nc + para->hac.output_interval * 0.5) * dt_in_ps);
        for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) 
        { fprintf(fid, "%25.15e", hac_ave[m]); }
        for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) 
        { fprintf(fid, "%25.15e", rtc_ave[m]); }
        fprintf(fid, "\n");
    }  
    fflush(fid);  
    fclose(fid);
    MY_FREE(hac);
    MY_FREE(rtc);    
}




// Calculate HAC (heat currant auto-correlation function) 
// and RTC (running thermal conductivity)
void postprocess_hac
(
    char *input_dir, Parameters *para, CPU_Data *cpu_data,
    GPU_Data *gpu_data, Integrate *integrate
)
{
    if (para->hac.compute) 
    {
        printf("INFO:  start to calculate HAC and related quantities.\n");
        find_hac_kappa(input_dir, para, cpu_data, gpu_data, integrate);
        CHECK(cudaFree(gpu_data->heat_all));
        printf("INFO:  HAC and related quantities are calculated.\n\n");
    }
}




