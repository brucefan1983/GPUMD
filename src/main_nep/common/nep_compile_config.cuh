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

#include <string>
#include <vector>

class NEP_Parameters;

const int NEP_SPECIALIZED_INTERFACE_VERSION = 4;

struct NEP_Compile_Config {
  std::string model_name;
  std::string source_file;
  std::string extra_config;

  int num_types = 0;
  int ann_dim = 0;
  int num_neurons1 = 0;
  int num_neurons2 = 0;
  int num_hidden_layers = 0;
  int one_ann_no_bias = 0;
  int ann_set_size = 0;

  // Offset from the start of the training parameter vector to descriptor cij.
  int c_offset = 0;

  int n_max_radial = 0;
  int n_max_angular = 0;
  int basis_size_radial = 0;
  int basis_size_angular = 0;
  int L_max = 0;
  int dim_angular = 0;
  int has_q_222 = 0;
  int has_q_1111 = 0;
  int has_q_112 = 0;
  int has_q_123 = 0;
  int has_q_233 = 0;
  int has_q_134 = 0;
  int num_L = 0;
  int num_c_radial = 0;

  bool descriptor_use_cj = false;

  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
};

NEP_Compile_Config make_nep_compile_config(const NEP_Parameters& para);
std::string make_nep_compile_array(const std::vector<float>& values);

#if !defined(USE_HIP) && !defined(_WIN32)

bool validate_nep_compile_config(const NEP_Compile_Config& config);
std::string make_nep_compile_config_text(const NEP_Compile_Config& config);

#endif
