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
Dump atom positions in XYZ compatible format.
------------------------------------------------------------------------------*/

#include "dump_xyz.cuh"

DUMP_XYZ::DUMP_XYZ()
{

}

void DUMP_XYZ::initialize(char *input_dir)
{
    strcpy(file_position, input_dir);
    strcat(file_position, "/movie.xyz");
    fid_position = my_fopen(file_position, "a");

    if (precision == 0)
        strcpy(precision_str, "%d %g %g %g\n");
    else if (precision == 1) // single
        strcpy(precision_str, "%d %0.9g %0.9g %0.9g\n");
    else if (precision == 2) // double precision
        strcpy(precision_str, "%d %.17f %.17f %.17f\n");
}


void DUMP_XYZ::finalize()
{
    fclose(fid_position);
}

void DUMP_XYZ::dump(Atom *atom, int step)
{
    if ((step + 1) % interval != 0) return;
    atom->x.copy_to_host(atom->cpu_x.data());
    atom->y.copy_to_host(atom->cpu_y.data());
    atom->z.copy_to_host(atom->cpu_z.data());
    fprintf(fid_position, "%d\n", atom->N);
    fprintf(fid_position, "%d\n", (step + 1) / interval - 1);

    for (int n = 0; n < atom->N; n++)
    {
        fprintf(fid_position, precision_str, atom->cpu_type[n],
            atom->cpu_x[n], atom->cpu_y[n], atom->cpu_z[n]);
    }
    fflush(fid_position);
}
