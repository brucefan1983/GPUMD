/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include "integrate/integrate.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

Dump_Thermo::Dump_Thermo(const char** param, int num_param) 
{
  parse(param, num_param);
  property_name = "dump_thermo";
}

void Dump_Thermo::parse(const char** param, int num_param)
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
  printf("Dump thermo every %d steps.\n", dump_interval_);
}

void Dump_Thermo::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  fid_ = my_fopen("thermo.out", "a");
  // header
  fprintf(fid_, "# col 1: T K\n");
  fprintf(fid_, "# col 2: K eV\n");
  fprintf(fid_, "# col 3: U eV\n");
  fprintf(fid_, "# col 4: Pxx GPa\n");
  fprintf(fid_, "# col 5: Pyy GPa\n");
  fprintf(fid_, "# col 6: Pzz GPa\n");
  fprintf(fid_, "# col 7: Pyz GPa\n");
  fprintf(fid_, "# col 8: Pzx GPa\n");
  fprintf(fid_, "# col 9: Pxy GPa\n");
  fprintf(fid_, "# col 10: ax A\n");
  fprintf(fid_, "# col 11: ay A\n");
  fprintf(fid_, "# col 12: az A\n");
  fprintf(fid_, "# col 13: bx A\n");
  fprintf(fid_, "# col 14: by A\n");
  fprintf(fid_, "# col 15: bz A\n");
  fprintf(fid_, "# col 16: cx A\n");
  fprintf(fid_, "# col 17: cy A\n");
  fprintf(fid_, "# col 18: cz A\n");
}

void Dump_Thermo::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature_target,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& gpu_thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  int number_of_atoms_fixed = (fixed_group < 0) ? 0 : group[0].cpu_size[fixed_group];

  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);
  double energy_kin, temperature;
  if (integrate.type >= 31) {
    energy_kin = thermo[0];
    temperature = temperature_target;
  } else {
    const int number_of_atoms_moving = atom.number_of_atoms - number_of_atoms_fixed;
    energy_kin = 1.5 * number_of_atoms_moving * K_B * thermo[0];
    temperature = thermo[0];
  }

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(
    fid_,
    "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e",
    temperature,
    energy_kin,
    thermo[1],
    thermo[2] * PRESSURE_UNIT_CONVERSION,
    thermo[3] * PRESSURE_UNIT_CONVERSION,
    thermo[4] * PRESSURE_UNIT_CONVERSION,
    thermo[7] * PRESSURE_UNIT_CONVERSION,
    thermo[6] * PRESSURE_UNIT_CONVERSION,
    thermo[5] * PRESSURE_UNIT_CONVERSION);

  fprintf(
    fid_,
    "%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e\n",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);
  fflush(fid_);
}

void Dump_Thermo::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(fid_);
}
