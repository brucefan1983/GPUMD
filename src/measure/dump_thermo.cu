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
Dump thermo data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_thermo.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"

void Dump_Thermo::parse(char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_thermo should have 1 parameter.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("thermo dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("thermo dump interval should > 0.");
  }
  dump_ = true;
  printf("Dump thermo every %d steps.\n", dump_interval_);
}

void Dump_Thermo::preprocess(char* input_dir)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/thermo.out");
    fid_ = my_fopen(filename_, "a");
  }
}

void Dump_Thermo::process(
  const int step,
  const int number_of_atoms,
  const int number_of_atoms_fixed,
  const Box& box,
  GPU_Vector<double>& gpu_thermo)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);

  const int number_of_atoms_moving = number_of_atoms - number_of_atoms_fixed;
  double energy_kin = 1.5 * number_of_atoms_moving * K_B * thermo[0];

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(
    fid_, "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e", thermo[0], energy_kin,
    thermo[1], thermo[2] * PRESSURE_UNIT_CONVERSION, thermo[3] * PRESSURE_UNIT_CONVERSION,
    thermo[4] * PRESSURE_UNIT_CONVERSION, thermo[7] * PRESSURE_UNIT_CONVERSION,
    thermo[6] * PRESSURE_UNIT_CONVERSION, thermo[5] * PRESSURE_UNIT_CONVERSION);

  if (box.triclinic == 0) {
    fprintf(fid_, "%20.10e%20.10e%20.10e\n", box.cpu_h[0], box.cpu_h[1], box.cpu_h[2]);
  } else {
    fprintf(
      fid_, "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e\n", box.cpu_h[0],
      box.cpu_h[3], box.cpu_h[6], box.cpu_h[1], box.cpu_h[4], box.cpu_h[7], box.cpu_h[2],
      box.cpu_h[5], box.cpu_h[8]);
  }
  fflush(fid_);
}

void Dump_Thermo::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
  }
}
