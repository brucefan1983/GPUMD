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
#include <iosfwd>
#include <string>
#include <vector>

class NEP_Model_File
{
public:
  struct ModelLayout {
    int dim_radial = 0;
    int dim_angular = 0;
    int dim = 0;
    int ann_count = 0;
    int global_parameters = 0;
    int number_of_variables_ann_1 = 0;
    int number_of_variables_ann = 0;
    int number_of_variables_descriptor = 0;
    int number_of_variables = 0;
  };

  std::string name;
  std::vector<std::string> elements;
  int num_types = 0;
  int num_neurons1 = 0;
  int num_neurons2 = 0;
  int n_max_radial = 0;
  int n_max_angular = 0;
  int basis_size_radial = 0;
  int basis_size_angular = 0;
  int L_max = 0;
  int has_q_222 = 0;
  int has_q_1111 = 0;
  int has_q_112 = 0;
  int has_q_123 = 0;
  int has_q_233 = 0;
  int has_q_134 = 0;
  bool has_multiple_cutoffs = false;
  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
  int max_radial = 0;
  int max_angular = 0;
  bool enable_zbl = false;
  bool flexible_zbl = false;
  bool use_typewise_cutoff_zbl = false;
  float zbl_rc_inner = 0.0f;
  float zbl_rc_outer = 0.0f;
  float typewise_cutoff_zbl_factor = 0.0f;
  ModelLayout layout;
  std::vector<double> energy_reference;
  std::vector<float> parameters;
  std::vector<float> q_scaler;
  std::vector<float> zbl_parameters;

  bool read(std::istream& input, std::string& error);
};
