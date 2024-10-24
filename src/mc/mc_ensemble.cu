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

/*----------------------------------------------------------------------------80
The abstract base class (ABC) for the MC_Ensemble classes.
------------------------------------------------------------------------------*/

#include "mc_ensemble.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static std::string get_potential_file_name()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string potential_file_name;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "potential") {
        potential_file_name = tokens[1];
        break;
      }
    }
  }

  input_run.close();
  return potential_file_name;
}

static void check_is_nep(std::string& potential_file_name)
{
  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens[0].substr(0, 3) != "nep") {
    PRINT_INPUT_ERROR("MCMD only supports NEP models.");
  }
  input_potential.close();
}

MC_Ensemble::MC_Ensemble(const char** param, int num_param)
{
  mc_output.open("mcmd.out", std::ios::app);
  mc_output << "# ";
  for (int n = 0; n < num_param; ++n) {
    mc_output << param[n] << " ";
  }
  mc_output << "\n";
  mc_output << "# num_MD_steps  acceptance_ratio [species_concentrations]" << std::endl;

  const int n_max = 1000;
  const int m_max = 1000;
  NN_radial.resize(n_max);
  NN_angular.resize(n_max);
  local_type_before.resize(n_max);
  local_type_after.resize(n_max);
  t2_radial_before.resize(n_max * m_max);
  t2_radial_after.resize(n_max * m_max);
  t2_angular_before.resize(n_max * m_max);
  t2_angular_after.resize(n_max * m_max);
  x12_radial.resize(n_max * m_max);
  y12_radial.resize(n_max * m_max);
  z12_radial.resize(n_max * m_max);
  x12_angular.resize(n_max * m_max);
  y12_angular.resize(n_max * m_max);
  z12_angular.resize(n_max * m_max);
  pe_before.resize(n_max);
  pe_after.resize(n_max);

  std::string potential_file_name = get_potential_file_name();
  check_is_nep(potential_file_name);
  nep_energy.initialize(potential_file_name.c_str());

#ifdef DEBUG
  rng = std::mt19937(13579);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
}

MC_Ensemble::~MC_Ensemble(void)
{
  // nothing now
}

bool MC_Ensemble::check_if_small_box(const double rc, const Box& box)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.0 * rc) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.0 * rc) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.0 * rc) {
    is_small_box = true;
  }
  return is_small_box;
}