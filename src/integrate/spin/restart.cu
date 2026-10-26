/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "restart.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "utilities/read_file.cuh"
#include "utilities/error.cuh"
#include <cmath>

void read_spin_restart(const std::vector<std::string>& tokens, Atom& atom)
{
  bool has_mttk_state = false, has_mttk_time = false;
  for (const auto& token : tokens) {
    if (token.find("mttk_state=") == 0) {
      if (has_mttk_state || token.size() == 11) PRINT_INPUT_ERROR("Invalid or duplicate MTTK restart state.");
      has_mttk_state = true;
      atom.spin_lattice_restart.mttk = token.substr(11);
    } else if (token.find("mttk_time=") == 0) {
      if (has_mttk_time || !is_valid_real(token.substr(10), &atom.spin_lattice_restart.mttk_time) ||
          !std::isfinite(atom.spin_lattice_restart.mttk_time) || atom.spin_lattice_restart.mttk_time < 0.0) {
        PRINT_INPUT_ERROR("Invalid or duplicate MTTK restart time.");
      }
      has_mttk_time = true;
    }
  }
  if (has_mttk_state != has_mttk_time) PRINT_INPUT_ERROR("MTTK restart requires state and time together.");

  bool has_nhc_state = false, has_nhc_time = false;
  for (const auto& token : tokens) {
    if (token.find("nhc_state=") == 0) {
      if (has_nhc_state || token.size() == 10) PRINT_INPUT_ERROR("Invalid or duplicate NHC restart state.");
      has_nhc_state = true;
      atom.spin_lattice_restart.nhc = token.substr(10);
    } else if (token.find("nhc_time=") == 0) {
      if (has_nhc_time || !is_valid_real(token.substr(9), &atom.spin_lattice_restart.nhc_time) ||
          !std::isfinite(atom.spin_lattice_restart.nhc_time) || atom.spin_lattice_restart.nhc_time < 0.0) {
        PRINT_INPUT_ERROR("Invalid or duplicate NHC restart time.");
      }
      has_nhc_time = true;
    }
  }
  if (has_nhc_state != has_nhc_time) PRINT_INPUT_ERROR("NHC restart requires state and time together.");
  if (has_nhc_state && has_mttk_state) PRINT_INPUT_ERROR("Conflicting NHC and MTTK restart states.");

  bool has_glsd_step = false, has_glsd_seed = false;
  for (const auto& token : tokens) {
    const bool is_step = token.find("spin_glsd_step=") == 0;
    const bool is_seed = token.find("spin_glsd_seed=") == 0;
    if (!is_step && !is_seed) continue;
    bool& seen = is_step ? has_glsd_step : has_glsd_seed;
    const std::string value = token.substr(token.find('=') + 1);
    if (seen || value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
      PRINT_INPUT_ERROR("Invalid or duplicate GLSD restart counter/seed.");
    }
    seen = true;
    try {
      const auto parsed = std::stoull(value);
      if (is_step) atom.spin_glsd_step = parsed;
      else atom.spin_glsd_seed = parsed;
    } catch (...) {
      PRINT_INPUT_ERROR("GLSD restart counter/seed is out of range.");
    }
  }
  if (has_glsd_step != has_glsd_seed || (has_glsd_seed && atom.spin_glsd_seed == 0)) {
    PRINT_INPUT_ERROR("GLSD restart requires both spin_glsd_step and a positive spin_glsd_seed.");
  }

}

bool write_spin_restart(FILE* fid, const Atom& atom, const Integrate& integrate, double global_time)
{
  const std::string mttk_state = integrate.spin_mttk_restart_state();
  const std::string nhc_state = integrate.spin_nhc_restart_state();
  if (!mttk_state.empty()) {
    fprintf(fid, "mttk_state=%s mttk_time=%.17g ", mttk_state.c_str(), global_time);
  }
  if (!nhc_state.empty()) {
    fprintf(fid, "nhc_state=%s nhc_time=%.17g ", nhc_state.c_str(), global_time);
  }
  if (atom.spin_glsd_seed != 0) {
    // Spin and thermostat checkpoints refer to the same completed step.
    fprintf(fid, "spin_glsd_step=%llu spin_glsd_seed=%llu ",
      static_cast<unsigned long long>(atom.spin_glsd_step),
      static_cast<unsigned long long>(atom.spin_glsd_seed));
  }
  return atom.spin_glsd_seed || !mttk_state.empty() || !nhc_state.empty();
}
