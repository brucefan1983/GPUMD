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
Calculate the velocity autocorrelation function (VAC)
[1] J. M. Dickey and A. Paskin, 
Computer Simulation of the Lattice Dynamics of Solids, 
Phys. Rev. 188, 1407 (1969).
------------------------------------------------------------------------------*/


#include "vac.cuh"
#include "group.cuh"
#include "atom.cuh"
#include "warp_reduce.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128
#define FILE_NAME_LENGTH      200


// Allocate memory for recording velocity data
void VAC::preprocess(Atom *atom)
{
    if (!compute_dos && !compute_sdc) return;
    if (compute_dos == compute_sdc)
    {
    	print_error("DOS and SDC commands cannot be used simultaneously.\n");
    }
    Group sel_group;  //selected group
    if (grouping_method == -1) { N = atom->N; }
    else
    {
    	sel_group = atom->group[grouping_method];
    	N = sel_group.cpu_size[group];

    	// initialize array that stores atom indices for the group
		int *gindex;
		MY_MALLOC(gindex, int, N);
		int group_index = sel_group.cpu_size_sum[group];
		for (int i = 0; i < N; i++)
		{
			gindex[i] = sel_group.cpu_contents[group_index];
			group_index++;
		}
	    // Copy indices to GPU
	    CHECK(cudaMalloc((void**)&g_gindex, sizeof(int) * N));
	    CHECK(cudaMemcpy(g_gindex, gindex, sizeof(int) * N, cudaMemcpyHostToDevice));
	    MY_FREE(gindex);
    }
    int num = N * (atom->number_of_steps / sample_interval);
    CHECK(cudaMalloc((void**)&vx_all, sizeof(real) * num));
    CHECK(cudaMalloc((void**)&vy_all, sizeof(real) * num));
    CHECK(cudaMalloc((void**)&vz_all, sizeof(real) * num));
}


// Record velocity data (kernel)
static __global__ void gpu_copy_velocity
(
    int N, int nd, int grouped,
    real *g_in_x, real *g_in_y, real *g_in_z,
    real *g_out_x, real *g_out_y, real *g_out_z,
    const int* __restrict__ g_gindex
)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x; // atom index
    if (n < N)
    {
        int m = nd * N + n;
        if (grouped)
        {
        	g_out_x[m] = g_in_x[LDG(g_gindex, n)];
			g_out_y[m] = g_in_y[LDG(g_gindex, n)];
			g_out_z[m] = g_in_z[LDG(g_gindex, n)];
        }
        else
        {
        	g_out_x[m] = g_in_x[n];
			g_out_y[m] = g_in_y[n];
			g_out_z[m] = g_in_z[n];
        }
    }
}


// Record velocity data (wrapper)
void VAC::process(int step, Atom *atom)
{
    if (!(compute_dos || compute_sdc)) return;
    if (step % sample_interval != 0) return;
    int nd = step / sample_interval;  
    int grid_size = (N - 1) / BLOCK_SIZE + 1;
    int grouped = (grouping_method != -1);
    gpu_copy_velocity<<<grid_size, BLOCK_SIZE>>>
    (N, nd, grouped,
    		atom->vx, atom->vy, atom->vz, vx_all, vy_all, vz_all, g_gindex);
    CUDA_CHECK_KERNEL
}


static __global__ void gpu_find_vac
(
    int N, int M, int compute_dos,
    const real* __restrict__ g_mass,
    const real* __restrict__ g_vx,
    const real* __restrict__ g_vy,
    const real* __restrict__ g_vz,
    real *g_vac_x, real *g_vac_y, real *g_vac_z,
    const int* __restrict__ g_gindex,
    int grouping_method
)
{
    //<<<Nc, 128>>>

    __shared__ real s_vac_x[128];
    __shared__ real s_vac_y[128];
    __shared__ real s_vac_z[128];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int number_of_patches = (N - 1) / 128 + 1;  

    s_vac_x[tid] = 0.0;  
    s_vac_y[tid] = 0.0;
    s_vac_z[tid] = 0.0;

    for (int m = 0; m < M; ++m)
    {
        int index_1 = m * N;
        int index_2 = (m + bid) * N;
        for (int patch = 0; patch < number_of_patches; ++patch)
        { 
            int n = tid + patch * 128;
            if (n < N)
            {
            	if (compute_dos)
            	{
            		real mass;
            		if (grouping_method != -1){ mass = LDG(g_mass, LDG(g_gindex,n));}
            		else {mass = LDG(g_mass, n);}
					s_vac_x[tid] += mass * LDG(g_vx, index_1 + n) *
							LDG(g_vx, index_2 + n);
					s_vac_y[tid] += mass * LDG(g_vy, index_1 + n) *
							LDG(g_vy, index_2 + n);
					s_vac_z[tid] += mass * LDG(g_vz, index_1 + n) *
							LDG(g_vz, index_2 + n);
            	}
            	else
            	{
            		s_vac_x[tid] += LDG(g_vx, index_1 + n) * LDG(g_vx, index_2 + n);
					s_vac_y[tid] += LDG(g_vy, index_1 + n) * LDG(g_vy, index_2 + n);
					s_vac_z[tid] += LDG(g_vz, index_1 + n) * LDG(g_vz, index_2 + n);
            	}
            }
        }
    }
    __syncthreads();

    if (tid < 64)
    {
        s_vac_x[tid] += s_vac_x[tid + 64];
        s_vac_y[tid] += s_vac_y[tid + 64];
        s_vac_z[tid] += s_vac_z[tid + 64];
    }
    __syncthreads();
 
    if (tid < 32)
    {
        warp_reduce(s_vac_x, tid);
        warp_reduce(s_vac_y, tid);
        warp_reduce(s_vac_z, tid); 
    }
   
    if (tid == 0)
    {
        int number_of_data = M * N;
        g_vac_x[bid] = s_vac_x[0] / number_of_data;
        g_vac_y[bid] = s_vac_y[0] / number_of_data;
        g_vac_z[bid] = s_vac_z[0] / number_of_data;
    }
}

// Calculate VAC
void VAC::find_vac(char *input_dir, Atom *atom)
{
    // rename variables
    int number_of_steps = atom->number_of_steps;
    real time_step = atom->time_step;
    real *mass = atom->mass;

    // other parameters
    int Nd = number_of_steps / sample_interval;
    int M = Nd - Nc; // number of time origins
    real dt = time_step * sample_interval;
    real dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

    // major data
    MY_MALLOC(vac_x, real, Nc);
    MY_MALLOC(vac_y, real, Nc);
    MY_MALLOC(vac_z, real, Nc);
    MY_MALLOC(vac_x_normalized, real, Nc);
    MY_MALLOC(vac_y_normalized, real, Nc);
    MY_MALLOC(vac_z_normalized, real, Nc);

    for (int nc = 0; nc < Nc; nc++) {vac_x[nc] = vac_y[nc] = vac_z[nc] = 0.0;}

    real *g_vac_x, *g_vac_y, *g_vac_z;
    CHECK(cudaMalloc((void**)&g_vac_x, sizeof(real) * Nc));
    CHECK(cudaMalloc((void**)&g_vac_y, sizeof(real) * Nc));
    CHECK(cudaMalloc((void**)&g_vac_z, sizeof(real) * Nc));

    // Here, the block size is fixed to 128, which is a good choice
    gpu_find_vac<<<Nc, 128>>>
    (
        N, M, compute_dos, mass,
        vx_all, vy_all, vz_all,
        g_vac_x, g_vac_y, g_vac_z,
        g_gindex, grouping_method
    );
    CUDA_CHECK_KERNEL

    CHECK(cudaMemcpy(vac_x, g_vac_x, sizeof(real)*Nc, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vac_y, g_vac_y, sizeof(real)*Nc, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vac_z, g_vac_z, sizeof(real)*Nc, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(g_vac_x));
    CHECK(cudaFree(g_vac_y));
    CHECK(cudaFree(g_vac_z));

    real vac_x_0 = vac_x[0];
    real vac_y_0 = vac_y[0];
    real vac_z_0 = vac_z[0];
    for (int nc = 0; nc < Nc; nc++)
    {
        vac_x_normalized[nc] = vac_x[nc] / vac_x_0;
        vac_y_normalized[nc] = vac_y[nc] / vac_y_0;
        vac_z_normalized[nc] = vac_z[nc] / vac_z_0;
    }

    if (compute_dos)
    {
		char file_vac[FILE_NAME_LENGTH];
		strcpy(file_vac, input_dir);
		strcat(file_vac, "/mvac.out");
		FILE *fid = fopen(file_vac, "a");
		for (int nc = 0; nc < Nc; nc++)
		{
			real t = nc * dt_in_ps;

			// change to A^2/ps^2
			vac_x[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;
			vac_y[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;
			vac_z[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;

			fprintf(fid, "%25.15e",                                             t);
			fprintf(fid, "%25.15e%25.15e%25.15e", vac_x[nc], vac_y[nc], vac_z[nc]);
			fprintf(fid, "\n");
		}
		fflush(fid);
		fclose(fid);
    }

}


// postprocess VAC and related quantities.
void VAC::postprocess(char *input_dir, Atom *atom, DOS *dos, SDC *sdc)
{
    if (!(compute_dos || compute_sdc)) return;
    print_line_1();
    printf("Start to calculate VAC and related quantities.\n");
    find_vac(input_dir, atom);
    if (compute_dos){dos->process(input_dir, atom, this);}
    else{sdc->process(input_dir, atom, this);}
    MY_FREE(vac_x);
    MY_FREE(vac_y);
    MY_FREE(vac_z);
    MY_FREE(vac_x_normalized);
    MY_FREE(vac_y_normalized);
    MY_FREE(vac_z_normalized);
    CHECK(cudaFree(vx_all));
    CHECK(cudaFree(vy_all));
    CHECK(cudaFree(vz_all));
    if (grouping_method != -1)
    {
    	CHECK(cudaFree(g_gindex));
    }
    printf("VAC and related quantities are calculated.\n");
    print_line_2();
}


