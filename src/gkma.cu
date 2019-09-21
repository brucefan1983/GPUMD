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
Green-Kubo Modal Analysis (GKMA)
- Currently only supports output of modal heat flux
 -> Green-Kubo integrals must be post-processed

GPUMD Contributing author: Alexander Gabourie (Stanford University)

Some code here and supporting code in 'potential.cu' is based on the LAMMPS
implementation provided by the Henry group at MIT. This code can be found:
https://drive.google.com/open?id=1IHJ7x-bLZISX3I090dW_Y_y-Mqkn07zg
------------------------------------------------------------------------------*/

#include "gkma.cuh"
#include "atom.cuh"
#include <fstream>
#include <string>
#include <iostream>

#define BLOCK_SIZE 128
#define ACCUM_BLOCK 1024


static __global__ void gpu_reset_data
(
        int num_elements, real* data
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        data[n] = ZERO;
    }
}

static __global__ void gpu_average_jm
(
        int num_elements, int samples_per_output, real* jm
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < num_elements)
    {
        jm[n]/=(float)samples_per_output;
    }
}

static __global__ void gpu_reduce_jmn
(
        int N, int num_modes,
        const real* __restrict__ jmn,
        real* jm
)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / ACCUM_BLOCK + 1;

    __shared__ real s_data_x[ACCUM_BLOCK];
    __shared__ real s_data_y[ACCUM_BLOCK];
    __shared__ real s_data_z[ACCUM_BLOCK];
    s_data_x[tid] = ZERO;
    s_data_y[tid] = ZERO;
    s_data_z[tid] = ZERO;

    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * ACCUM_BLOCK;
        if (n < N)
        {
            s_data_x[tid] += jmn[n + bid*N ];
            s_data_y[tid] += jmn[n + (bid + num_modes)*N];
            s_data_z[tid] += jmn[n + (bid + 2*num_modes)*N];
        }
    }

    __syncthreads();
    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_data_x[tid] += s_data_x[tid + offset];
            s_data_y[tid] += s_data_y[tid + offset];
            s_data_z[tid] += s_data_z[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        jm[bid] = s_data_x[0];
        jm[bid + num_modes] = s_data_y[0];
        jm[bid + 2*num_modes] = s_data_z[0];
    }

}


void GKMA::preprocess(char *input_dir, Atom *atom)
{
    num_modes = last_mode-first_mode+1;
    samples_per_output = output_interval/sample_interval;
    num_bins = num_modes/bin_size;

    strcpy(gkma_file_position, input_dir);
    strcat(gkma_file_position, "/heatmode.out");

    // initialize eigenvector data structures
    strcpy(eig_file_position, input_dir);
    strcat(eig_file_position, "/eigenvector.eig");
    std::ifstream eigfile;
    eigfile.open(eig_file_position);
    if (!eigfile)
    {
        print_error("Cannot open eigenvector.eig file.\n");
    }

    int N = atom->N;
    MY_MALLOC(cpu_eig, real, N * num_modes * 3);
    CHECK(cudaMalloc(&eig, sizeof(real) * N * num_modes * 3));

    // Following code snippet is heavily based on MIT LAMMPS code
    std::string val;
    double doubleval;

    for (int i=0; i<=N+3 ; i++){
        getline(eigfile,val);
    }
    for (int i=0; i<first_mode-1; i++){
      for (int j=0; j<N+2; j++) getline(eigfile,val);
    }
    for (int j=0; j<num_modes; j++){
        getline(eigfile,val);
        getline(eigfile,val);
        for (int i=0; i<N; i++){
            eigfile >> doubleval;
            cpu_eig[i + 3*N*j] = doubleval;
            eigfile >> doubleval;
            cpu_eig[i + (1 + 3*j)*N] = doubleval;
            eigfile >> doubleval;
            cpu_eig[i + (2 + 3*j)*N] = doubleval;
        }
        getline(eigfile,val);
    }
    eigfile.close();
    //end snippet

    CHECK(cudaMemcpy(eig, cpu_eig, sizeof(real) * N * num_modes * 3,
                            cudaMemcpyHostToDevice));
    MY_FREE(cpu_eig);

    // Allocate modal variables
    MY_MALLOC(cpu_jm, real, num_modes * 3) //cpu
    CHECK(cudaMalloc(&xdot, sizeof(real) * num_modes * 3));
    CHECK(cudaMalloc(&jm, sizeof(real) * num_modes * 3));
    CHECK(cudaMalloc(&xdotn, sizeof(real) * num_modes * 3 * N));
    CHECK(cudaMalloc(&jmn, sizeof(real) * num_modes * 3 * N));

    int num_elements = num_modes*3;
    gpu_reset_data<<<(num_elements-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
    (
            num_elements, jm
    );
    CUDA_CHECK_KERNEL

    gpu_reset_data<<<(num_elements*N-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
    (
            num_elements*N, jmn
    );
    CUDA_CHECK_KERNEL

}


void GKMA::process(int step, Atom *atom)
{
    if (!compute) return;
    if (!((step+1) % output_interval == 0)) return;

    int N = atom->N;
    gpu_reduce_jmn<<<num_modes, ACCUM_BLOCK>>>
    (
            N, num_modes, jmn, jm
    );
    CUDA_CHECK_KERNEL


    int num_elements = num_modes*3;
    gpu_average_jm<<<(num_elements-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
    (
            num_elements, samples_per_output, jm
    );
    CUDA_CHECK_KERNEL

    // TODO make into a GPU function
    real *bin_out; // bins of heat current modes for output
    ZEROS(bin_out, real, 3*num_bins);

    CHECK(cudaMemcpy(cpu_jm, jm, sizeof(real) * num_modes * 3,
                        cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_bins; i++)
    {
        for (int j = 0; j < bin_size; j++)
        {
            bin_out[i] += cpu_jm[j + i*bin_size];
            bin_out[i + num_bins] += cpu_jm[j + i*bin_size + num_modes];
            bin_out[i + 2*num_bins] += cpu_jm[j + i*bin_size + 2*num_modes];
        }
    }

    FILE *fid = fopen(gkma_file_position, "a");
    for (int i = 0; i < num_bins; i++)
    {
        fprintf(fid, "%25.15e %25.15e %25.15e\n",
                bin_out[i], bin_out[i+num_bins], bin_out[i+2*num_bins]);
    }
    fflush(fid);
    fclose(fid);
    MY_FREE(bin_out);

    gpu_reset_data<<<(num_elements*N-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>
    (
            num_elements*N, jmn
    );
    CUDA_CHECK_KERNEL

}

void GKMA::postprocess()
{
    if (!compute) return;
    CHECK(cudaFree(eig));
    CHECK(cudaFree(xdot));
    CHECK(cudaFree(jm));
    CHECK(cudaFree(jmn));
    MY_FREE(cpu_jm);
}


