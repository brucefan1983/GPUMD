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
Calculate the self (running) diffusion coefficient (SDC)
[1] J. M. Dickey and A. Paskin, 
Computer Simulation of the Lattice Dynamics of Solids, 
Phys. Rev. 188, 1407 (1969).
------------------------------------------------------------------------------*/


#include "sdc.cuh"
#include "atom.cuh"
#include "warp_reduce.cuh"
#include "error.cuh"

#define BLOCK_SIZE 128

// Calculate the Self Diffusion Coefficient (SDC)
// from the VAC using the Green-Kubo formula
static void find_sdc
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

// Calculate (1) VAC, (2) SDC, and (3) DOS = phonon density of states
void SDC::process(char *input_dir, Atom *atom, VAC *vac)
{
    // rename variables
    real time_step = atom->time_step;
    real *vac_x = vac->vac_x;
	real *vac_y = vac->vac_y;
	real *vac_z = vac->vac_z;
	int Nc = vac->Nc;

    // other parameters
    real dt = time_step * vac->sample_interval;
    real dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

    // major data
    real *sdc_x, *sdc_y, *sdc_z;
    MY_MALLOC(sdc_x, real, Nc);
    MY_MALLOC(sdc_y, real, Nc);
    MY_MALLOC(sdc_z, real, Nc);

    for (int nc = 0; nc < Nc; nc++) {sdc_x[nc] = sdc_y[nc] = sdc_z[nc] = 0.0;}

    find_sdc(Nc, dt, vac_x, vac_y, vac_z, sdc_x, sdc_y, sdc_z);

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



