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
Spectral heat current (SHC) calculations. Referene:
[1] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium 
molecular dynamics method for heat transport and spectral decomposition 
with many-body potentials, Phys. Rev. B 99, 064308 (2019).
------------------------------------------------------------------------------*/


#include "shc.cuh"
#include "atom.cuh"
#include "error.cuh"

const int BLOCK_SIZE_SHC = 128;


static __global__ void gpu_initialize_k(int Nc, double *g_ki, double *g_ko)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < Nc)
    {
        g_ki[n] = ZERO;
        g_ko[n] = ZERO;
    }
}


void SHC::preprocess(Atom *atom)
{
    if (!compute) { return; }

    num_time_origins = 0;
    if (-1 == group_method)
    {
        group_size = atom->N;
    }
    else
    {
        group_size = atom->group[group_method].cpu_size[group_id];
    }

    vx.resize(group_size * Nc);
    vy.resize(group_size * Nc);
    vz.resize(group_size * Nc);
    sx.resize(group_size);
    sy.resize(group_size);
    sz.resize(group_size);
    ki.resize(Nc, Memory_Type::managed);
    ko.resize(Nc, Memory_Type::managed);

    gpu_initialize_k<<<(Nc - 1) / BLOCK_SIZE_SHC + 1, BLOCK_SIZE_SHC>>>
    (Nc, ki.data(), ko.data());
    CUDA_CHECK_KERNEL
}


static __global__ void gpu_find_k
(
    int group_size, int correlation_step,
    double *g_sx, double *g_sy, double *g_sz,
    double *g_vx, double *g_vy, double *g_vz,
    double *g_ki, double *g_ko
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int size_sum = bid * group_size;
    int number_of_rounds = (group_size - 1) / BLOCK_SIZE_SHC + 1;
    __shared__ double s_ki[BLOCK_SIZE_SHC];
    __shared__ double s_ko[BLOCK_SIZE_SHC];
    double ki = ZERO;
    double ko = ZERO;

    for (int round = 0; round < number_of_rounds; ++round)
    {
        int n = tid + round * BLOCK_SIZE_SHC;
        if (n < group_size)
        {
            ki += g_sx[n] * g_vx[size_sum + n] + g_sy[n] * g_vy[size_sum + n];
            ko += g_sz[n] * g_vz[size_sum + n];
        }
    }
    s_ki[tid] = ki;
    s_ko[tid] = ko;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_ki[tid] += s_ki[tid + offset];
            s_ko[tid] += s_ko[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        if (bid <= correlation_step)
        {
            g_ki[correlation_step - bid] += s_ki[0];
            g_ko[correlation_step - bid] += s_ko[0];
        }
        else
        {
            g_ki[correlation_step + gridDim.x - bid] += s_ki[0];
            g_ko[correlation_step + gridDim.x - bid] += s_ko[0]; 
        }
    }
}


static __global__ void gpu_copy_data
(
    int group_size, int offset, int *g_group_contents,
    double *g_sx_o, double *g_sy_o, double *g_sz_o,
    double *g_vx_o, double *g_vy_o, double *g_vz_o,
    double *g_sx_i, double *g_sy_i, double *g_sz_i,
    double *g_vx_i, double *g_vy_i, double *g_vz_i
)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n < group_size)
    {
        int m = g_group_contents[offset + n];
        g_sx_o[n] = g_sx_i[m];
        g_sy_o[n] = g_sy_i[m];
        g_sz_o[n] = g_sz_i[m];
        g_vx_o[n] = g_vx_i[m];
        g_vy_o[n] = g_vy_i[m];
        g_vz_o[n] = g_vz_i[m];
    }
}


void SHC::process(int step, Atom *atom)
{
    if (!compute) { return; }
    if ((step + 1) % sample_interval != 0) { return; }
    int sample_step = step / sample_interval; // 0, 1, ..., Nc-1, Nc, Nc+1, ...
    int correlation_step = sample_step % Nc;  // 0, 1, ..., Nc-1, 0, 1, ...
    int offset = correlation_step * group_size;

    const int tensor[3][3] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
    double *sx_tmp = atom->virial_per_atom + atom->N * tensor[direction][0];
    double *sy_tmp = atom->virial_per_atom + atom->N * tensor[direction][1];
    double *sz_tmp = atom->virial_per_atom + atom->N * tensor[direction][2];

    if (-1 == group_method)
    {
        sx.copy_from_device(sx_tmp);
        sy.copy_from_device(sy_tmp);
        sz.copy_from_device(sz_tmp);
        CHECK(cudaMemcpy(vx.data() + offset, atom->vx, group_size * sizeof(double),
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(vy.data() + offset, atom->vy, group_size * sizeof(double),
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(vz.data() + offset, atom->vz, group_size * sizeof(double),
            cudaMemcpyDeviceToDevice));
    }
    else
    {
        gpu_copy_data<<<(group_size - 1) / BLOCK_SIZE_SHC + 1, BLOCK_SIZE_SHC>>>
        (
            group_size, atom->group[group_method].cpu_size_sum[group_id],
            atom->group[group_method].contents,
            sx.data(), sy.data(), sz.data(), vx.data() + offset, vy.data() + offset, vz.data() + offset,
            sx_tmp, sy_tmp, sz_tmp, atom->vx , atom->vy, atom->vz
        );
        CUDA_CHECK_KERNEL 
    }

    if (sample_step >= Nc - 1)
    {
        ++num_time_origins;
        
        gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>
        (group_size, correlation_step, sx.data(), sy.data(), sz.data(), vx.data(), vy.data(), vz.data(), ki.data(), ko.data());
        CUDA_CHECK_KERNEL 
    }
}


void SHC::postprocess(char *input_dir)
{
    if (!compute) { return; }

    CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU
    char file_shc[FILE_NAME_LENGTH];
    strcpy(file_shc, input_dir);
    strcat(file_shc, "/shc.out");
    FILE *fid = my_fopen(file_shc, "a");

    for (int nc = 0; nc < Nc; ++nc)
    {
        fprintf
        (
            fid, "%25.15e%25.15e\n", 
            ki[nc] / num_time_origins, ko[nc] / num_time_origins
        );
    }
    fflush(fid);
    fclose(fid);
}


