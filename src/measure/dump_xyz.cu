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

DUMP_XYZ::DUMP_XYZ() {}

void DUMP_XYZ::initialize(char* input_dir, const int number_of_atoms)
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

void DUMP_XYZ::finalize() { fclose(fid_position); }

void DUMP_XYZ::dump(
  const int step,
  const double global_time,
  const Box& box,
  const std::vector<int>& cpu_type,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  if ((step + 1) % interval != 0)
    return;

  const int number_of_atoms = cpu_type.size();

  position_per_atom.copy_to_host(cpu_position_per_atom.data());

  fprintf(fid_position, "%d\n", number_of_atoms);
  fprintf(fid_position, "%d\n", (step + 1) / interval - 1);

  for (int n = 0; n < number_of_atoms; n++) {
    fprintf(
      fid_position, precision_str, cpu_type[n], cpu_position_per_atom[n],
      cpu_position_per_atom[n + number_of_atoms], cpu_position_per_atom[n + 2 * number_of_atoms]);
  }
  fflush(fid_position);
}
