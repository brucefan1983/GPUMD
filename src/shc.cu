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


static __global__ void gpu_initialize_k(int Nc, real *g_ki, real *g_ko)
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

    CHECK(cudaMalloc((void**)&vx, sizeof(real) * group_size * Nc));
    CHECK(cudaMalloc((void**)&vy, sizeof(real) * group_size * Nc));
    CHECK(cudaMalloc((void**)&vz, sizeof(real) * group_size * Nc));
    CHECK(cudaMalloc((void**)&sx, sizeof(real) * group_size));
    CHECK(cudaMalloc((void**)&sy, sizeof(real) * group_size));
    CHECK(cudaMalloc((void**)&sz, sizeof(real) * group_size));
    CHECK(cudaMallocManaged((void**)&ki, sizeof(real) * Nc));
    CHECK(cudaMallocManaged((void**)&ko, sizeof(real) * Nc));

    gpu_initialize_k<<<(Nc - 1) / BLOCK_SIZE_SHC + 1, BLOCK_SIZE_SHC>>>
    (Nc, ki, ko);
    CUDA_CHECK_KERNEL
}


static __global__ void gpu_find_k
(
    int group_size, int correlation_step,
    real *g_sx, real *g_sy, real *g_sz,
    real *g_vx, real *g_vy, real *g_vz,
    real *g_ki, real *g_ko
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int size_sum = bid * group_size;
    int number_of_rounds = (group_size - 1) / BLOCK_SIZE_SHC + 1;
    __shared__ real s_ki[BLOCK_SIZE_SHC];
    __shared__ real s_ko[BLOCK_SIZE_SHC];
    real ki = ZERO;
    real ko = ZERO;

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
    real *g_sx_o, real *g_sy_o, real *g_sz_o,
    real *g_vx_o, real *g_vy_o, real *g_vz_o,
    real *g_sx_i, real *g_sy_i, real *g_sz_i,
    real *g_vx_i, real *g_vy_i, real *g_vz_i
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
    real *sx_tmp = atom->virial_per_atom + atom->N * tensor[direction][0];
    real *sy_tmp = atom->virial_per_atom + atom->N * tensor[direction][1];
    real *sz_tmp = atom->virial_per_atom + atom->N * tensor[direction][2];

    if (-1 == group_method)
    {
        CHECK(cudaMemcpy(sx, sx_tmp, group_size * sizeof(real), 
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(sy, sy_tmp, group_size * sizeof(real), 
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(sz, sz_tmp, group_size * sizeof(real), 
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(vx + offset, atom->vx, group_size * sizeof(real), 
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(vy + offset, atom->vy, group_size * sizeof(real), 
            cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(vz + offset, atom->vz, group_size * sizeof(real), 
            cudaMemcpyDeviceToDevice));
    }
    else
    {
        gpu_copy_data<<<(group_size - 1) / BLOCK_SIZE_SHC + 1, BLOCK_SIZE_SHC>>>
        (
            group_size, atom->group[group_method].cpu_size_sum[group_id],
            atom->group[group_method].contents,
            sx, sy, sz, vx + offset, vy + offset, vz + offset,
            sx_tmp, sy_tmp, sz_tmp, atom->vx , atom->vy, atom->vz
        );
        CUDA_CHECK_KERNEL 
    }

    if (sample_step >= Nc - 1)
    {
        ++num_time_origins;
        
        gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>
        (group_size, correlation_step, sx, sy, sz, vx, vy, vz, ki, ko);
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

    CHECK(cudaFree(vx));
    CHECK(cudaFree(vy));
    CHECK(cudaFree(vz));
    CHECK(cudaFree(sx));
    CHECK(cudaFree(sy));
    CHECK(cudaFree(sz));
    CHECK(cudaFree(ki));
    CHECK(cudaFree(ko));
}


