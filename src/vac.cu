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
#include "error.cuh"

#define BLOCK_SIZE 128


// Allocate memory for recording velocity data
void VAC::preprocess(Atom *atom)
{
    if (!compute_dos && !compute_sdc) return;
    if (compute_dos == compute_sdc)
    {
        PRINT_INPUT_ERROR("DOS and SDC commands cannot be used simultaneously.");
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

    if (compute_dos)
    {
        // set default number of DOS points
        if (num_dos_points == -1) {num_dos_points = Nc;}
        float sample_frequency = 1000.0/(atom->time_step * sample_interval); // THz
        if (sample_frequency < omega_max/PI)
        {
            printf("WARNING: VAC sampling rate is less than Nyquist frequency.\n");
        }
    }
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

    #pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_vac_x[tid] += s_vac_x[tid + offset];
            s_vac_y[tid] += s_vac_y[tid + offset];
            s_vac_z[tid] += s_vac_z[tid + offset];
        }
        __syncthreads();
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

// Calculate phonon density of states (DOS)
// using the method by Dickey and Paskin
static void perform_dft
(
    int N, int Nc, int num_dos_points,
    real delta_t, real omega_0, real d_omega,
    real *vac_x_normalized, real *vac_y_normalized, real *vac_z_normalized,
    real *dos_x, real *dos_y, real *dos_z
)
{
    // Apply Hann window and normalize by the correct factor
    for (int nc = 0; nc < Nc; nc++)
    {
        real hann_window = (cos((PI * nc) / Nc) + 1.0) * 0.5;

        real multiply_factor = 2.0 * hann_window;
        if (nc == 0)
        {
            multiply_factor = 1.0 * hann_window;
        }

        vac_x_normalized[nc] *= multiply_factor;
        vac_y_normalized[nc] *= multiply_factor;
        vac_z_normalized[nc] *= multiply_factor;
    }

    // Calculate DOS by discrete Fourier transform
    for (int nw = 0; nw < num_dos_points; nw++)
    {
        real omega = omega_0 + nw * d_omega;
        for (int nc = 0; nc < Nc; nc++)
        {
            real cos_factor = cos(omega * nc * delta_t);
            dos_x[nw] += vac_x_normalized[nc] * cos_factor;
            dos_y[nw] += vac_y_normalized[nc] * cos_factor;
            dos_z[nw] += vac_z_normalized[nc] * cos_factor;
        }
        dos_x[nw] *= delta_t*2.0*N;
        dos_y[nw] *= delta_t*2.0*N;
        dos_z[nw] *= delta_t*2.0*N;
    }
}


// Calculate phonon density of states
void VAC::find_dos(char *input_dir, Atom *atom)
{
    // rename variables
    real time_step = atom->time_step;

    // other parameters
    real dt = time_step * sample_interval;
    real dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps
    real d_omega = omega_max / num_dos_points;
    real omega_0 = d_omega;

    // major data
    real *dos_x, *dos_y, *dos_z;
    MY_MALLOC(dos_x, real, num_dos_points);
    MY_MALLOC(dos_y, real, num_dos_points);
    MY_MALLOC(dos_z, real, num_dos_points);

    for (int nw = 0; nw < num_dos_points; nw++)
    {
    	dos_x[nw] = dos_y[nw] = dos_z[nw] = 0.0;
    }
    perform_dft
    (
        N, Nc, num_dos_points, dt_in_ps, omega_0, d_omega,
        vac_x_normalized, vac_y_normalized, vac_z_normalized,
        dos_x, dos_y, dos_z
    );

    char file_dos[FILE_NAME_LENGTH];
    strcpy(file_dos, input_dir);
    strcat(file_dos, "/dos.out");
    FILE *fid = fopen(file_dos, "a");
    for (int nw = 0; nw < num_dos_points; nw++)
    {
        real omega = omega_0 + d_omega * nw;
        fprintf(fid, "%25.15e",                                         omega);
        fprintf(fid, "%25.15e%25.15e%25.15e", dos_x[nw], dos_y[nw], dos_z[nw]);
        fprintf(fid, "\n");
    }
    fflush(fid);
    fclose(fid);
    MY_FREE(dos_x); MY_FREE(dos_y); MY_FREE(dos_z);
}


// Calculate the Self Diffusion Coefficient (SDC)
// from the VAC using the Green-Kubo formula
static void integrate_vac
(
    int Nc, real dt, real *vac_x, real *vac_y, real *vac_z,
    real *sdc_x, real *sdc_y, real *sdc_z
)
{
    real dt2 = dt * 0.5;
    for (int nc = 1; nc < Nc; nc++)
    {
        sdc_x[nc] = sdc_x[nc - 1] + (vac_x[nc - 1] + vac_x[nc]) * dt2;
        sdc_y[nc] = sdc_y[nc - 1] + (vac_y[nc - 1] + vac_y[nc]) * dt2;
        sdc_z[nc] = sdc_z[nc - 1] + (vac_z[nc - 1] + vac_z[nc]) * dt2;
    }
}


void VAC::find_sdc(char *input_dir, Atom *atom)
{
    // rename variables
    real time_step = atom->time_step;

    // other parameters
    real dt = time_step * sample_interval;
    real dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

    // major data
    real *sdc_x, *sdc_y, *sdc_z;
    MY_MALLOC(sdc_x, real, Nc);
    MY_MALLOC(sdc_y, real, Nc);
    MY_MALLOC(sdc_z, real, Nc);

    for (int nc = 0; nc < Nc; nc++) {sdc_x[nc] = sdc_y[nc] = sdc_z[nc] = 0.0;}

    integrate_vac(Nc, dt, vac_x, vac_y, vac_z, sdc_x, sdc_y, sdc_z);

    char file_sdc[FILE_NAME_LENGTH];
    strcpy(file_sdc, input_dir);
    strcat(file_sdc, "/sdc.out");
    FILE *fid = fopen(file_sdc, "a");
    for (int nc = 0; nc < Nc; nc++)
    {
        real t = nc * dt_in_ps;

        // change to A^2/ps^2
        vac_x[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;
        vac_y[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;
        vac_z[nc] *= 1000000.0 / TIME_UNIT_CONVERSION / TIME_UNIT_CONVERSION;

        sdc_x[nc] *= 1000.0 / TIME_UNIT_CONVERSION; // change to A^2/ps
        sdc_y[nc] *= 1000.0 / TIME_UNIT_CONVERSION; // change to A^2/ps
        sdc_z[nc] *= 1000.0 / TIME_UNIT_CONVERSION; // change to A^2/ps

        fprintf(fid, "%25.15e",                                             t);
        fprintf(fid, "%25.15e%25.15e%25.15e", vac_x[nc], vac_y[nc], vac_z[nc]);
        fprintf(fid, "%25.15e%25.15e%25.15e", sdc_x[nc], sdc_y[nc], sdc_z[nc]);
        fprintf(fid, "\n");
    }
    fflush(fid);
    fclose(fid);

    MY_FREE(sdc_x); MY_FREE(sdc_y); MY_FREE(sdc_z);
}


// postprocess VAC and related quantities.
void VAC::postprocess(char *input_dir, Atom *atom)
{
    if (!(compute_dos || compute_sdc)) return;
    print_line_1();
    printf("Start to calculate VAC and related quantities.\n");
    find_vac(input_dir, atom);

    if (compute_dos)
    {
        find_dos(input_dir, atom);
    }
    else
    {
        find_sdc(input_dir, atom);
    }

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


