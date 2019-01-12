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




#include "compute.cuh"

#include "integrate.cuh"
#include "ensemble.cuh"
#include "atom.cuh"
#include "error.cuh"

#define DIM 3




void Compute::preprocess(char* input_dir, Atom* atom)
{
    if (compute_force)
    {
        MY_MALLOC(group_fx, real, atom->number_of_groups);
        MY_MALLOC(group_fy, real, atom->number_of_groups);
        MY_MALLOC(group_fz, real, atom->number_of_groups);
        char filename[200];
        strcpy(filename, input_dir);
        strcat(filename, "/force.out");
        fid_force = my_fopen(filename, "a");
    }
    if (compute_temperature)
    {
        MY_MALLOC(group_temperature, real, atom->number_of_groups);
        char filename[200];
        strcpy(filename, input_dir);
        strcat(filename, "/temperature.out");
        fid_temperature = my_fopen(filename, "a");
    }
}




static __device__ void warp_reduce(volatile real *s, int t) 
{
    s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];
    s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];
}




static __global__ void find_group_force
(
    int *g_group_size, int *g_group_size_sum, int *g_group_contents,
    real *g_fx, real *g_fy, real *g_fz, real *g_group_fx, real *g_group_fy,
    real *g_group_fz
)
{
    // <<<number_of_groups, 256>>> (one CUDA block for one group of atoms)
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int group_size = g_group_size[bid];
    int offset = g_group_size_sum[bid];
    int number_of_patches = (group_size - 1) / 256 + 1;
    __shared__ real s_fx[256];
    __shared__ real s_fy[256];
    __shared__ real s_fz[256];
    s_fx[tid] = ZERO; s_fy[tid] = ZERO; s_fz[tid] = ZERO;

    for (int patch = 0; patch < number_of_patches; patch++)
    {
        int k = tid + patch * 256;
        if (k < group_size)
        {
            int n = g_group_contents[offset + k]; // particle index
            s_fx[tid] = g_fx[n]; s_fy[tid] = g_fy[n]; s_fz[tid] = g_fz[n];
        }
    }
    __syncthreads();

    if (tid < 128)
    {
        s_fx[tid] += s_fx[tid + 128];
        s_fy[tid] += s_fy[tid + 128];
        s_fz[tid] += s_fz[tid + 128];
    }
    __syncthreads();
    if (tid < 64)
    {
        s_fx[tid] += s_fx[tid + 64];
        s_fy[tid] += s_fy[tid + 64];
        s_fz[tid] += s_fz[tid + 64];
    }
    __syncthreads();
    if (tid < 32)
    {
        warp_reduce(s_fx, tid);
        warp_reduce(s_fy, tid);
        warp_reduce(s_fz, tid);
    }
    if (tid == 0)
    {
        g_group_fx[bid] = s_fx[0] / (group_size);
        g_group_fy[bid] = s_fy[0] / (group_size);
        g_group_fz[bid] = s_fz[0] / (group_size);
    }
}




static __global__ void find_group_temperature
(
    int  *g_group_size, int  *g_group_size_sum, int  *g_group_contents,
    real *g_mass, real *g_vx, real *g_vy, real *g_vz, real *g_group_temperature
)
{
    // <<<number_of_groups, 256>>> (one CUDA block for one group of atoms)

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int group_size = g_group_size[bid];
    int offset = g_group_size_sum[bid];
    int number_of_patches = (group_size - 1) / 256 + 1;
    __shared__ real s_ke[256];
    s_ke[tid] = ZERO;

    for (int patch = 0; patch < number_of_patches; patch++)
    {
        int k = tid + patch * 256;
        if (k < group_size)
        {
            int n = g_group_contents[offset + k]; // particle index
            real vx = g_vx[n]; real vy = g_vy[n]; real vz = g_vz[n];
            s_ke[tid] += g_mass[n] * (vx * vx + vy * vy + vz * vz);
        }
    }
    __syncthreads();

    if (tid <  128) { s_ke[tid] += s_ke[tid + 128]; }  __syncthreads();
    if (tid <   64) { s_ke[tid] += s_ke[tid + 64];  }  __syncthreads();
    if (tid <   32) { warp_reduce(s_ke, tid);       }
    if (tid ==   0) 
    {g_group_temperature[bid] = s_ke[0] / (DIM * K_B * group_size);}
}




void Compute::process_force(int step, Atom *atom)
{
    if (!compute_force) return;
    if (step % interval_force != 0) return;
    int Ng = atom->number_of_groups;
    real *g_group_fx;
    real *g_group_fy;
    real *g_group_fz;
    CHECK(cudaMalloc((void**)&g_group_fx, sizeof(real) * Ng));
    CHECK(cudaMalloc((void**)&g_group_fy, sizeof(real) * Ng));
    CHECK(cudaMalloc((void**)&g_group_fz, sizeof(real) * Ng));
    find_group_force<<<Ng, 256>>>(atom->group_size, atom->group_size_sum,
        atom->group_contents, atom->fx, atom->fy, atom->fz,
        g_group_fx, g_group_fy, g_group_fz);
    CUDA_CHECK_KERNEL
    CHECK(cudaMemcpy(group_fx, g_group_fx, sizeof(real) * Ng,
        cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(group_fy, g_group_fy, sizeof(real) * Ng,
        cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(group_fz, g_group_fz, sizeof(real) * Ng,
        cudaMemcpyDeviceToHost));
    CHECK(cudaFree(g_group_fx));
    CHECK(cudaFree(g_group_fy));
    CHECK(cudaFree(g_group_fz));

    for (int k = 0; k < Ng; k++)
    {
        fprintf(fid_force, "%15.6e%15.6e%15.6e",
            group_fx[k], group_fy[k], group_fz[k]);
    }
    fprintf(fid_force, "\n");
    fflush(fid_force);
}




void Compute::process_temperature(int step, Atom *atom, Integrate *integrate)
{
    if (!compute_temperature) return;
    if (step % interval_temperature != 0) return;
    int Ng = atom->number_of_groups;

    // calculate the block temperatures
    real *temp_gpu;
    CHECK(cudaMalloc((void**)&temp_gpu, sizeof(real) * Ng));
    find_group_temperature<<<Ng, 256>>>(atom->group_size, atom->group_size_sum,
        atom->group_contents, atom->mass, atom->vx, atom->vy, atom->vz,
        temp_gpu);
    CUDA_CHECK_KERNEL
    CHECK(cudaMemcpy(group_temperature, temp_gpu, sizeof(real) * Ng,
        cudaMemcpyDeviceToHost));
    CHECK(cudaFree(temp_gpu));

    // output
    for (int k = 0; k < Ng; k++)
    {
        fprintf(fid_temperature, "%15.6e", group_temperature[k]);
    }
    fprintf(fid_temperature, "%15.6e",
        integrate->ensemble->energy_transferred[0]);
    fprintf(fid_temperature, "%15.6e",
        integrate->ensemble->energy_transferred[1]);
    fprintf(fid_temperature, "\n");
    fflush(fid_temperature);
}




void Compute::process(int step, Atom *atom, Integrate *integrate)
{
    process_force(step, atom);
    process_temperature(step, atom, integrate);
}




void Compute::postprocess(Atom* atom, Integrate *integrate)
{
    if (compute_force)
    {
        MY_FREE(group_fx);
        MY_FREE(group_fy);
        MY_FREE(group_fz);
        fclose(fid_force);
    }
    if (compute_temperature)
    {
        MY_FREE(group_temperature);
        fclose(fid_temperature);
    }
}




