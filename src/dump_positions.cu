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
Dump atom positions in XYZ compatible format (or redirect to alternative
formatter).
------------------------------------------------------------------------------*/

#include "dump_positions.cuh"
#include "error.cuh"
#include "atom.cuh"


void DUMP_POS::initialize(char *input_dir)
{
	if (output_pos)
	{
		strcpy(file_position, input_dir);
		if (format == 0)
		{
			strcat(file_position, "/movie.xyz");
		}
#ifdef NETCDF
		else if(format == 1)
		{
			strcat(file_position, "/movie.nc");
		}
#endif

		fid_position = my_fopen(file_position, "a");
	}

}

void DUMP_POS::finalize()
{
	if (output_pos)
	{
		fclose(fid_position);
		output_pos = 0;
	}
}

void DUMP_POS::dump(Atom *atom, int step)
{
	if ((step + 1) % interval != 0) return;
	if (format == 0)
	{
		dump_xyz(atom, step);
	}
#ifdef NETCDF
	else if (format == 1)
	{
		dump_netcdf(atom, step);
	}
#endif
}

void DUMP_POS::dump_xyz(Atom *atom, int step)
{

	int memory = sizeof(real) * atom->N;
	CHECK(cudaMemcpy(atom->cpu_x, atom->x, memory, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(atom->cpu_y, atom->y, memory, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(atom->cpu_z, atom->z, memory, cudaMemcpyDeviceToHost));
	fprintf(fid_position, "%d\n", atom->N);
	fprintf(fid_position, "%d\n", (step + 1) / interval - 1);

	// Determine output precision
	if (precision == 1)
	{
		strcpy(precision_str, "%d %f %f %f\n"); // higher precision
	}
	for (int n = 0; n < atom->N; n++)
	{
		fprintf(fid_position, precision_str, atom->cpu_type[n],
			atom->cpu_x[n], atom->cpu_y[n], atom->cpu_z[n]);
	}
	fflush(fid_position);
}

