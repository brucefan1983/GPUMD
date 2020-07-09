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
Dump a restart file
--------------------------------------------------------------------------------------------------*/

#include "dump_restart.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/neighbor.cuh"
//#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Dump_Restart::parse(char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_restart should have 1 parameter.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("restart dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("restart dump interval should > 0.");
  }
  dump_ = true;
  printf("Dump restart every %d steps.\n", dump_interval_);
}

void Dump_Restart::preprocess(char* input_dir)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/restart.out");
    fid_ = my_fopen(filename_, "w");
  }
}

void Dump_Restart::process(
  const int step,
  const Neighbor& neighbor,
  const Box& box,
  const std::vector<Group>& group,
  const std::vector<int>& cpu_type,
  const std::vector<double>& cpu_mass,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  std::vector<double>& cpu_position_per_atom,
  std::vector<double>& cpu_velocity_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int number_of_atoms = cpu_mass.size();

  position_per_atom.copy_to_host(cpu_position_per_atom.data());
  velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());

  fprintf(
    fid_, "%d %d %g %d %d %d\n", number_of_atoms, neighbor.MN, neighbor.rc, box.triclinic, 1,
    int(group.size()));

  if (box.triclinic == 0) {
    fprintf(
      fid_, "%d %d %d %g %g %g\n", box.pbc_x, box.pbc_y, box.pbc_z, box.cpu_h[0], box.cpu_h[1],
      box.cpu_h[2]);
  } else {
    fprintf(
      fid_, "%d %d %d %g %g %g %g %g %g %g %g %g\n", box.pbc_x, box.pbc_y, box.pbc_z, box.cpu_h[0],
      box.cpu_h[3], box.cpu_h[6], box.cpu_h[1], box.cpu_h[4], box.cpu_h[7], box.cpu_h[2],
      box.cpu_h[5], box.cpu_h[8]);
  }

  for (int n = 0; n < number_of_atoms; n++) {
    fprintf(
      fid_, "%d %g %g %g %g %g %g %g ", cpu_type[n], cpu_position_per_atom[n],
      cpu_position_per_atom[n + number_of_atoms], cpu_position_per_atom[n + 2 * number_of_atoms],
      cpu_mass[n], cpu_velocity_per_atom[n], cpu_velocity_per_atom[n + number_of_atoms],
      cpu_velocity_per_atom[n + 2 * number_of_atoms]);

    for (int m = 0; m < group.size(); ++m) {
      fprintf(fid_, "%d ", group[m].cpu_label[n]);
    }

    fprintf(fid_, "\n");
  }

  fflush(fid_);
}

void Dump_Restart::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
  }
}
