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

/*-----------------------------------------------------------------------------------------------100
Dump velocity data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_velocity.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Dump_Velocity::parse(char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_velocity should have 1 parameter.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("velocity dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("velocity dump interval should > 0.");
  }
  dump_ = true;
  printf("Dump velocity every %d steps.\n", dump_interval_);
}

void Dump_Velocity::preprocess(char* input_dir)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/velocity.out");
    fid_ = my_fopen(filename_, "a");
  }
}

void Dump_Velocity::process(
  const int step, GPU_Vector<double>& velocity_per_atom, std::vector<double>& cpu_velocity_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int number_of_atoms = velocity_per_atom.size() / 3;
  velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());
  for (int n = 0; n < number_of_atoms; n++) {
    fprintf(
      fid_, "%g %g %g\n", cpu_velocity_per_atom[n], cpu_velocity_per_atom[n + number_of_atoms],
      cpu_velocity_per_atom[n + 2 * number_of_atoms]);
  }

  fflush(fid_);
}

void Dump_Velocity::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
  }
}
