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


void HNEMD::preprocess(Atom *atom)
{
    if (!compute) return;
    int num = NUM_OF_HEAT_COMPONENTS * output_interval;
    CHECK(cudaMalloc((void**)&heat_all, sizeof(double) * num));
}


// calculate the per-atom heat current 
static __global__ void gpu_get_peratom_heat
(
    int N, double *sxx, double *sxy, double *sxz, double *syx, double *syy, double *syz,
    double *szx, double *szy, double *szz, double *vx, double *vy, double *vz, 
    double *jx_in, double *jx_out, double *jy_in, double *jy_out, double *jz
)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
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
(int N, int step, double *g_heat, double *g_heat_sum)
{
    // <<<5, 1024>>> 
    int tid = threadIdx.x; 
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / 1024 + 1;
    __shared__ double s_data[1024];  
    s_data[tid] = ZERO;
    for (int patch = 0; patch < number_of_patches; ++patch)
    {
        int n = tid + patch * 1024; 
        if (n < N) { s_data[tid] += g_heat[n + N * bid]; }
    }
    __syncthreads();

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset) { s_data[tid] += s_data[tid + offset]; }
        __syncthreads();
    }

    if (tid ==  0) { g_heat_sum[step*NUM_OF_HEAT_COMPONENTS+bid] = s_data[0]; }
}


void HNEMD::process(int step, char *input_dir, Atom *atom, Integrate *integrate)
{
    if (!compute) return;
    int output_flag = ((step+1) % output_interval == 0);
    step %= output_interval;

    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_get_peratom_heat<<<(atom->N - 1) / 128 + 1, 128>>>
    (
        atom->N, 
        atom->virial_per_atom, 
        atom->virial_per_atom + atom->N * 3,
        atom->virial_per_atom + atom->N * 4,
        atom->virial_per_atom + atom->N * 6,
        atom->virial_per_atom + atom->N * 1,
        atom->virial_per_atom + atom->N * 5,
        atom->virial_per_atom + atom->N * 7,
        atom->virial_per_atom + atom->N * 8,
        atom->virial_per_atom + atom->N * 2,
        atom->vx, atom->vy, atom->vz, 
        atom->heat_per_atom, 
        atom->heat_per_atom + atom->N,
        atom->heat_per_atom + atom->N * 2,
        atom->heat_per_atom + atom->N * 3,
        atom->heat_per_atom + atom->N * 4
    );
    CUDA_CHECK_KERNEL

    gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>
    (atom->N, step, atom->heat_per_atom, heat_all);
    CUDA_CHECK_KERNEL

    if (output_flag)
    {
        int num = NUM_OF_HEAT_COMPONENTS * output_interval;
        int mem = sizeof(double) * num;
        double volume = atom->box.get_volume();
        std::vector<double> heat_cpu(num);
        CHECK(cudaMemcpy(heat_cpu.data(), heat_all, mem, cudaMemcpyDeviceToHost));
        double kappa[NUM_OF_HEAT_COMPONENTS];
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
        double factor = KAPPA_UNIT_CONVERSION / output_interval;
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
    }
}


void HNEMD::postprocess(Atom *atom)
{
    if (compute) { CHECK(cudaFree(heat_all)); }
}


