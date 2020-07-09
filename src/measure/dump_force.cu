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
Dump force data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_force.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Dump_Force::parse(char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_force should have 1 parameter.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("force dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("force dump interval should > 0.");
  }
  dump_ = true;
  printf("Dump force every %d steps.\n", dump_interval_);
}

void Dump_Force::preprocess(char* input_dir, const int number_of_atoms)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/force.out");
    fid_ = my_fopen(filename_, "a");
    cpu_force_per_atom.resize(number_of_atoms * 3);
  }
}

void Dump_Force::process(const int step, GPU_Vector<double>& force_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int number_of_atoms = force_per_atom.size() / 3;
  force_per_atom.copy_to_host(cpu_force_per_atom.data());
  for (int n = 0; n < number_of_atoms; n++) {
    fprintf(
      fid_, "%g %g %g\n", cpu_force_per_atom[n], cpu_force_per_atom[n + number_of_atoms],
      cpu_force_per_atom[n + 2 * number_of_atoms]);
  }

  fflush(fid_);
}

void Dump_Force::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
  }
}
