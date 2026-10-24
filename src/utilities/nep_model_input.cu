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

#include "error.cuh"
#include "nep_model_file.cuh"
#include "nep_model_input.cuh"

bool NEP_Model_Input::is_new_format(const std::string& token)
{
  return token.size() == 3 && (token[0] == 'n' || token[0] == 'N') &&
         (token[1] == 'e' || token[1] == 'E') && (token[2] == 'p' || token[2] == 'P');
}

NEP_Model_Input::NEP_Model_Input(const std::string& filename) : file(filename)
{
  if (!file.is_open()) {
    const std::string error = "Cannot open potential file " + filename + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  const std::vector<std::string> header = ::get_tokens(file);
  file.clear();
  file.seekg(0);
  new_format = !header.empty() && is_new_format(header[0]);
  if (!new_format) {
    return;
  }

  NEP_Model_File model;
  std::string error;
  if (!model.read(file, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  std::string input;
  if (!model.get_md_input(input, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  converted.str(input);
  reference.swap(model.energy_reference);
}

std::vector<std::string> NEP_Model_Input::get_tokens()
{
  std::string line;
  std::getline(new_format ? static_cast<std::istream&>(converted) : file, line);
  return ::get_tokens(line);
}

const std::vector<double>& NEP_Model_Input::energy_reference() const { return reference; }

std::vector<float> NEP_Model_Input::get_descriptor_parameters_type_pair(
  const std::vector<float>& parameters,
  const int num_para_ann,
  const int num_types,
  const int n_max_radial,
  const int n_max_angular,
  const int basis_size_radial,
  const int basis_size_angular) const
{
  const int num_types_sq = num_types * num_types;
  const int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  const int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  const int num_c_radial = num_types_sq * num_radial_basis;

  // New files already use the type-pair-major layout required by MD kernels.
  if (new_format) {
    return std::vector<float>(
      parameters.begin() + num_para_ann,
      parameters.begin() + num_para_ann + num_c_radial + num_types_sq * num_angular_basis);
  }

  // Legacy files use basis-major order, separately for radial and angular parameters.
  std::vector<float> descriptor_parameters(
    num_c_radial + num_types_sq * num_angular_basis);

  for (int type_pair = 0; type_pair < num_types_sq; ++type_pair) {
    for (int basis = 0; basis < num_radial_basis; ++basis) {
      descriptor_parameters[type_pair * num_radial_basis + basis] =
        parameters[num_para_ann + basis * num_types_sq + type_pair];
    }
    for (int basis = 0; basis < num_angular_basis; ++basis) {
      descriptor_parameters[num_c_radial + type_pair * num_angular_basis + basis] =
        parameters[num_para_ann + num_c_radial + basis * num_types_sq + type_pair];
    }
  }

  return descriptor_parameters;
}
