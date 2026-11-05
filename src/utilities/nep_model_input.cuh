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

#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class NEP_Model_Input
{
public:
  explicit NEP_Model_Input(const std::string& filename);
  std::vector<std::string> get_tokens();
  const std::vector<double>& energy_reference() const;
  std::vector<float> get_descriptor_parameters_type_pair(
    const std::vector<float>& parameters,
    const int num_para_ann,
    const int num_types,
    const int n_max_radial,
    const int n_max_angular,
    const int basis_size_radial,
    const int basis_size_angular) const;

  static bool is_new_format(const std::string& token);

private:
  std::ifstream file;
  std::istringstream converted;
  bool new_format = false;
  std::vector<double> reference;
};
