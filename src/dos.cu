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
Calculates the phonon (vibrational) density of states (DOS) with the mass-
weighted VAC with the integral of the DOS normalized to 3N.
[1] J. M. Dickey and A. Paskin, 
Computer Simulation of the Lattice Dynamics of Solids, 
Phys. Rev. 188, 1407 (1969).
------------------------------------------------------------------------------*/


#include "dos.cuh"
#include "atom.cuh"
#include "warp_reduce.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128
#define FILE_NAME_LENGTH      200


// Allocate memory for recording velocity data
void DOS::preprocess(Atom *atom, VAC *vac)
{
    if (!vac->compute_dos) return;
    // set default number of DOS points
    if (num_dos_points == -1) {num_dos_points = vac->Nc;}
    float sample_frequency =
    		1000.0/(atom->time_step * vac->sample_interval); // THz
	if (sample_frequency < omega_max/PI)
	{
		printf("WARNING: VAC sampling rate is less than Nyquist frequency.\n");
	}
}

// Calculate phonon density of states (DOS)
// using the method by Dickey and Paskin
static void find_dos
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
void DOS::process(char *input_dir, Atom *atom, VAC *vac)
{
    // rename variables
    int N = vac->N;
    real time_step = atom->time_step;
    int Nc = vac->Nc;
    real *vac_x_normalized = vac->vac_x_normalized;
    real *vac_y_normalized = vac->vac_y_normalized;
    real *vac_z_normalized = vac->vac_z_normalized;

    // other parameters
    real dt = time_step * vac->sample_interval;
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
    find_dos
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



