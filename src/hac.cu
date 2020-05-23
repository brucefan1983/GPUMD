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
Calculate the heat current autocorrelation (HAC) function.
------------------------------------------------------------------------------*/


#include "hac.cuh"
#include "ensemble.cuh"
#include "atom.cuh"
#include "error.cuh"

#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH 200
#define DIM 3


//Allocate memory for recording heat current data
void HAC::preprocess(const int number_of_steps)
{
    if (compute)
    {
        int number_of_frames = number_of_steps / sample_interval;
        heat_all.resize(NUM_OF_HEAT_COMPONENTS * number_of_frames);
    }
}


// calculate the per-atom heat current 
static __global__ void gpu_get_peratom_heat
(
    const int N,
    const double *sxx,
    const double *sxy,
    const double *sxz,
    const double *syx,
    const double *syy,
    const double *syz,
    const double *szx,
    const double *szy,
    const double *szz,
    const double *vx,
    const double *vy,
    const double *vz,
    double *jx_in,
    double *jx_out,
    double *jy_in,
    double *jy_out,
    double *jz
)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N)
    {
        jx_in[n] = sxx[n] * vx[n] + sxy[n] * vy[n];
        jx_out[n] = sxz[n] * vz[n];
        jy_in[n] = syx[n] * vx[n] + syy[n] * vy[n];
        jy_out[n] = syz[n] * vz[n];
        jz[n] = szx[n] * vx[n] + szy[n] * vy[n] + szz[n] * vz[n];
    }
}


// sum up the per-atom heat current to get the total heat current
static __global__ void gpu_sum_heat
(
    const int N,
    const int Nd,
    const int nd,
    const double *g_heat,
    double *g_heat_all
)
{
    // <<<NUM_OF_HEAT_COMPONENTS, 1024>>> 
    const int tid = threadIdx.x;
    const int number_of_patches = (N - 1) / 1024 + 1;

    __shared__ double s_data[1024];  
    s_data[tid] = 0.0;
 
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        const int n = tid + patch * 1024;
        if (n < N) { s_data[tid] += g_heat[n + N * blockIdx.x]; }
    }

    __syncthreads();
    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_data[tid] += s_data[tid + offset]; }
        __syncthreads();
    }
    if (tid ==  0) { g_heat_all[nd + Nd * blockIdx.x] = s_data[0]; }
}


// sample heat current data for HAC calculations.
void HAC::process
(
    const int number_of_steps,
    const int step,
    const char *input_dir,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& heat_per_atom
)
{
    if (!compute) return; 
    if ((step + 1) % sample_interval != 0) return;

    const int N = velocity_per_atom.size() / 3;

    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_get_peratom_heat<<<(N - 1) / 128 + 1, 128>>>
    (
        N,
        virial_per_atom.data(),
        virial_per_atom.data() + N * 3,
        virial_per_atom.data() + N * 4,
        virial_per_atom.data() + N * 6,
        virial_per_atom.data() + N * 1,
        virial_per_atom.data() + N * 5,
        virial_per_atom.data() + N * 7,
        virial_per_atom.data() + N * 8,
        virial_per_atom.data() + N * 2,
        velocity_per_atom.data(),
        velocity_per_atom.data() + N,
        velocity_per_atom.data() + 2 * N,
        heat_per_atom.data(),
        heat_per_atom.data() + N,
        heat_per_atom.data() + N * 2,
        heat_per_atom.data() + N * 3,
        heat_per_atom.data() + N * 4
    );
    CUDA_CHECK_KERNEL
 
    int nd = (step + 1) / sample_interval - 1;
    int Nd = number_of_steps / sample_interval;
    gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>
    (
        N,
        Nd,
        nd,
        heat_per_atom.data(),
        heat_all.data()
    );
    CUDA_CHECK_KERNEL
}


// Calculate the Heat current Auto-Correlation function (HAC) 
__global__ void gpu_find_hac
(
    const int Nc,
    const int Nd,
    const double *g_heat,
    double *g_hac
)
{
    //<<<Nc, 128>>>

    __shared__ double s_hac_xi[128];
    __shared__ double s_hac_xo[128];
    __shared__ double s_hac_yi[128];
    __shared__ double s_hac_yo[128];
    __shared__ double s_hac_z[128];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (Nd - 1) / 128 + 1;
    int number_of_data = Nd - bid;

    s_hac_xi[tid] = 0.0;
    s_hac_xo[tid] = 0.0;
    s_hac_yi[tid] = 0.0;
    s_hac_yo[tid] = 0.0;
    s_hac_z[tid]  = 0.0;

    for (int patch = 0; patch < number_of_patches; ++patch)
    { 
        int index = tid + patch * 128;
        if (index + bid < Nd)
        {
            s_hac_xi[tid] += g_heat[index + Nd * 0]
                           * g_heat[index + bid + Nd * 0]
                           + g_heat[index + Nd * 0]
                           * g_heat[index + bid + Nd * 1];
            s_hac_xo[tid] += g_heat[index + Nd * 1]
                           * g_heat[index + bid + Nd * 1]
                           + g_heat[index + Nd * 1]
                           * g_heat[index + bid + Nd * 0];
            s_hac_yi[tid] += g_heat[index + Nd * 2]
                           * g_heat[index + bid + Nd * 2]
                           + g_heat[index + Nd * 2]
                           * g_heat[index + bid + Nd * 3];
            s_hac_yo[tid] += g_heat[index + Nd * 3]
                           * g_heat[index + bid + Nd * 3]
                           + g_heat[index + Nd * 3]
                           * g_heat[index + bid + Nd * 2];
            s_hac_z[tid]  += g_heat[index + Nd * 4]
                           * g_heat[index + bid + Nd * 4];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_hac_xi[tid] += s_hac_xi[tid + offset];
            s_hac_xo[tid] += s_hac_xo[tid + offset];
            s_hac_yi[tid] += s_hac_yi[tid + offset];
            s_hac_yo[tid] += s_hac_yo[tid + offset];
            s_hac_z[tid]  += s_hac_z[tid  + offset];
        }
        __syncthreads();
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
static void find_rtc
(
    const int Nc,
    const double factor,
    const double *hac,
    double *rtc
)
{
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++)
    {
        for (int nc = 1; nc < Nc; nc++)  
        {
            const int index = Nc * k + nc;
            rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index])*factor;
        }
    }
}


// Calculate HAC (heat currant auto-correlation function) 
// and RTC (running thermal conductivity)
void HAC::postprocess
(
    const int number_of_steps,
    const char *input_dir,
    const double temperature,
    const double time_step,
    const double volume
)
{
    if (!compute) return;
    print_line_1();
    printf("Start to calculate HAC and related quantities.\n");

    const int Nd = number_of_steps / sample_interval;
    const double dt = time_step * sample_interval;
    const double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

    // major data
    std::vector<double> rtc(Nc * NUM_OF_HEAT_COMPONENTS, 0.0);
    GPU_Vector<double> hac(Nc * NUM_OF_HEAT_COMPONENTS, Memory_Type::managed);

    // Here, the block size is fixed to 128, which is a good choice
    gpu_find_hac<<<Nc, 128>>>(Nc, Nd, heat_all.data(), hac.data());
    CUDA_CHECK_KERNEL

    CHECK(cudaDeviceSynchronize()); // Needed for Windows

    //const double volume = atom->box.get_volume();
    double factor = dt * 0.5 / (K_B * temperature * temperature * volume);
    factor *= KAPPA_UNIT_CONVERSION;

    find_rtc(Nc, factor, hac.data(), rtc.data());

    char file_hac[FILE_NAME_LENGTH];
    strcpy(file_hac, input_dir);
    strcat(file_hac, "/hac.out");
    FILE *fid = fopen(file_hac, "a");
    const int number_of_output_data = Nc / output_interval;
    for (int nd = 0; nd < number_of_output_data; nd++)
    {
        const int nc = nd * output_interval;
        double hac_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
        double rtc_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
        for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++)
        {
            for (int m = 0; m < output_interval; m++)
            {
                const int count = Nc * k + nc + m;
                hac_ave[k] += hac[count];
                rtc_ave[k] += rtc[count];
            }
        }
        for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++)
        {
            hac_ave[m] /= output_interval;
            rtc_ave[m] /= output_interval;
        }
        fprintf
        (
            fid,
            "%25.15e",
            (nc + output_interval * 0.5) * dt_in_ps
        );
        for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++)
        {
            fprintf(fid, "%25.15e", hac_ave[m]);
        }
        for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++)
        {
            fprintf(fid, "%25.15e", rtc_ave[m]);
        }
        fprintf(fid, "\n");
    }
    fflush(fid);
    fclose(fid);

    printf("HAC and related quantities are calculated.\n");
    print_line_2();
}


