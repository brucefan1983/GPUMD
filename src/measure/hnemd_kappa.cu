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
[1] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium
molecular dynamics method for heat transport and spectral decomposition
with many-body potentials, Phys. Rev. B 99, 064308 (2019).
------------------------------------------------------------------------------*/


#include "hnemd_kappa.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH       200


void HNEMD::preprocess()
{
    if (!compute) return;
    heat_all.resize(NUM_OF_HEAT_COMPONENTS * output_interval);
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


static __global__ void gpu_sum_heat
(
    const int N,
    const int step,
    const double *g_heat,
    double *g_heat_sum
)
{
    // <<<5, 1024>>> 
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int number_of_patches = (N - 1) / 1024 + 1;
    __shared__ double s_data[1024];  
    s_data[tid] = 0.0;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        const int n = tid + patch * 1024;
        if (n < N) { s_data[tid] += g_heat[n + N * bid]; }
    }
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_data[tid] += s_data[tid + offset]; }
        __syncthreads();
    }

    if (tid ==  0)
    {
        g_heat_sum[step*NUM_OF_HEAT_COMPONENTS+bid] = s_data[0];
    }
}


void HNEMD::process
(
    int step,
    const char *input_dir,
    const double temperature,
    const double volume,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& heat_per_atom
)
{
    if (!compute) return;
    const int output_flag = ((step+1) % output_interval == 0);
    step %= output_interval;

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

    gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>
    (
        N,
        step,
        heat_per_atom.data(),
        heat_all.data()
    );
    CUDA_CHECK_KERNEL

    if (output_flag)
    {
        const int num = NUM_OF_HEAT_COMPONENTS * output_interval;
        std::vector<double> heat_cpu(num);
        heat_all.copy_to_host(heat_cpu.data());
        double kappa[NUM_OF_HEAT_COMPONENTS];
        for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) 
        {
            kappa[n] = 0.0;
        }
        for (int m = 0; m < output_interval; m++)
        {
            for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++)
            {
                kappa[n] += heat_cpu[m * NUM_OF_HEAT_COMPONENTS + n];
            }
        }
        double factor = KAPPA_UNIT_CONVERSION / output_interval;
        factor /= (volume * temperature * fe);

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
    }
}


void HNEMD::postprocess()
{
    if (compute)
    {
        // nothing now
    }
}


