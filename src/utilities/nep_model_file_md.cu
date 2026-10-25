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

#include "nep_model_file.cuh"
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

bool NEP_Model_File::get_md_input(std::string& input, std::string& error) const
{
  if (name == "nep" || name == "nep_temperature") {
    return get_nep_md_input(input, error);
  }
  if (name == "qnep1" || name == "qnep2") {
    return get_charge_md_input(input, error);
  }
  if (name == "nep_dipole" || name == "nep_polarizability") {
    return get_tnep_md_input(input, error);
  }
  return get_md_input("", input, error);
}

bool NEP_Model_File::get_md_input(
  const std::string& tag, std::string& input, std::string& error) const
{
  input.clear();
  error.clear();
  if (num_neurons2 != 0) {
    error = "Native MD supports only one hidden layer.";
    return false;
  }
  if (zbl_parameters.size() > 550) {
    error = "Native MD supports flexible ZBL for at most 10 active elements.";
    return false;
  }
  if (
    max_radial < 1 || max_radial > 819 || max_angular < 1 ||
    max_angular > std::numeric_limits<int>::max() / 2) {
    error = "Invalid neighbor capacities for native MD.";
    return false;
  }

  if (tag.empty()) {
    error = "This model is not supported by native MD.";
    return false;
  }

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10);
  output << tag << ' ' << num_types;
  for (const std::string& element : elements) {
    output << ' ' << element;
  }
  output << '\n';
  if (enable_zbl) {
    output << "zbl " << zbl_rc_inner << ' ' << zbl_rc_outer;
    if (use_typewise_cutoff_zbl) {
      output << ' ' << typewise_cutoff_zbl_factor;
    }
    output << '\n';
  }
  output << "cutoff";
  const int count = has_multiple_cutoffs ? num_types : 1;
  for (int t = 0; t < count; ++t) {
    output << ' ' << rc_radial[t] << ' ' << rc_angular[t];
  }
  output << ' ' << max_radial << ' ' << max_angular << '\n';
  output << "n_max " << n_max_radial << ' ' << n_max_angular << '\n';
  output << "basis_size " << basis_size_radial << ' ' << basis_size_angular << '\n';
  output << "l_max " << L_max << ' ' << 2 * has_q_222 << ' ' << has_q_1111;
  const int extra_flags[] = {has_q_112, has_q_123, has_q_233, has_q_134};
  int last_flag = 3;
  while (last_flag >= 0 && extra_flags[last_flag] == 0) {
    --last_flag;
  }
  for (int i = 0; i <= last_flag; ++i) {
    output << ' ' << extra_flags[i];
  }
  output << '\n';
  output << "ANN " << num_neurons1 << " 0\n";
  for (const float value : parameters) {
    output << value << '\n';
  }
  for (const float value : q_scaler) {
    output << value << '\n';
  }
  for (const float value : zbl_parameters) {
    output << value << '\n';
  }
  input = output.str();
  return true;
}
