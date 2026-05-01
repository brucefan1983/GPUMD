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
Calculate the electrostatic energy and forces
--------------------------------------------------------------------------------------------------*/

#include "compute_es.cuh"
#include "force/force.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <iostream>
#include <vector>

void Compute_es::check_ewald_pppm()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  use_pppm = true;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "kspace") {
        if (tokens.size() != 2) {
          std::cout << "kspace must have 1 parameter\n";
          exit(1);
        }
        std::string kspace_method = tokens[1];
        if (kspace_method == "ewald") {
          use_pppm = false;
        } else if (kspace_method == "pppm") {
          use_pppm = true;
        } else {
          std::cout << "kspace method can only be ewald or pppm\n";
          exit(1);
        }
      }
    }
  }

  input_run.close();
}

void Compute_es::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  alpha = float(PI) / 10.0f; // a good value
  check_ewald_pppm();
  if (use_pppm) {
    pppm.initialize(alpha);
  } else {
    ewald.initialize(alpha);
  }
}

void Compute_es::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  const int N = atom.number_of_atoms;
  GPU_Vector<float> D_real(N);

  if (use_pppm) {
    pppm.find_force(
      N,
      0,
      N,
      box,
      atom.charge,
      atom.position_per_atom,
      D_real,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.potential_per_atom);
  } else {
    ewald.find_force(
      N,
      0,
      N,
      box.cpu_h,
      atom.charge,
      atom.position_per_atom,
      D_real,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.potential_per_atom);
  }

  std::vector<double> force_cpu(N * 3);
  atom.force_per_atom.copy_to_host(force_cpu.data());

  FILE* fid = fopen("elactrostatic.out", "a");
  for (int n = 0; n < N; ++n) {
    printf("%16.8e%16.8e%16.8e\n", force_cpu[0 * N + n], force_cpu[1 * N + n], force_cpu[2 * N + n]);
  }
  fclose(fid);
}

void Compute_es::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  // nothing
}

void Compute_es::parse(const char** param, int num_param)
{
  printf("Compute electrostatic energy and force.\n");

  if (num_param != 2) {
    PRINT_INPUT_ERROR("compute_dpdt should have 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for compute_es should be an integer number.\n");
  }
  if (sample_interval != 1) {
    PRINT_INPUT_ERROR("sample interval for compute_es should be 1.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);
}

Compute_es::Compute_es(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "compute_es";
}
