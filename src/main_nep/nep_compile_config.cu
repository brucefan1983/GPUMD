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

#include "nep_compile_config.cuh"
#include "nep_parameters.cuh"
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
void warning_compile(const std::string& message)
{
  std::cerr << "Warning: NEP training specialization disabled: "
            << message << std::endl;
}

#if !defined(USE_HIP) && !defined(_WIN32)

std::string float_literal(const float value)
{
  std::ostringstream output;
  output << std::scientific << std::setprecision(9) << value << "f";
  return output.str();
}

std::string make_float_array(const std::vector<float>& values)
{
  std::ostringstream output;
  output << "{";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << float_literal(values[i]);
  }
  output << "}";
  return output.str();
}

bool all_equal(const std::vector<float>& values)
{
  if (values.empty()) return true;
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] != values[0]) return false;
  }
  return true;
}
#endif
}

NEP_Compile_Config make_nep_compile_config(
  const NEP_Parameters& para,
  const NEP_Compile_Mode mode,
  const float* c6_ref_sqrt,
  int model_type)
{
#ifdef USE_CJ
  if (
    mode == NEP_Compile_Mode::CHARGE ||
    mode == NEP_Compile_Mode::CHARGE_VDW ||
    mode == NEP_Compile_Mode::TNEP) {
    warning_compile(
      "qNEP, charge-vdW, and TNEP specialization is not available in "
      "this USE_CJ configuration.");
    return NEP_Compile_Config();
  }
#endif

  NEP_Compile_Config c;
  c.mode = mode;
  c.model_type = model_type;
  c.num_types = para.num_types;
  c.ann_dim = para.dim;
  c.num_neurons1 = para.num_neurons1;
  c.num_neurons2 = para.num_neurons2;
  c.num_hidden_layers = para.num_hidden_layers;
  c.one_ann_no_bias = para.number_of_variables_ann_1;
  c.ann_set_size = para.number_of_variables_ann;

  // All NEP-family parameter vectors place descriptor cij immediately after
  // the ANN parameter block. TNEP polarizability has two ANN parameter sets.
  c.c_offset = para.number_of_variables_ann * para.ann_count;

  c.n_max_radial = para.n_max_radial;
  c.n_max_angular = para.n_max_angular;
  c.basis_size_radial = para.basis_size_radial;
  c.basis_size_angular = para.basis_size_angular;
  c.L_max = para.L_max;
  c.dim_angular = para.dim_angular;
  c.has_q_222 = para.has_q_222;
  c.has_q_1111 = para.has_q_1111;
  c.has_q_112 = para.has_q_112;
  c.has_q_123 = para.has_q_123;
  c.has_q_233 = para.has_q_233;
  c.has_q_134 = para.has_q_134;
  c.num_L = para.dim_angular / (para.n_max_angular + 1);

#ifdef USE_CJ
  c.descriptor_use_cj =
    (mode == NEP_Compile_Mode::NEP || mode == NEP_Compile_Mode::VDW);
#else
  c.descriptor_use_cj = false;
#endif

  const int num_descriptor_types =
    c.descriptor_use_cj ? para.num_types : para.num_types * para.num_types;
  c.num_c_radial =
    num_descriptor_types *
    (para.n_max_radial + 1) *
    (para.basis_size_radial + 1);

  c.rc_radial.resize(para.num_types);
  c.rc_angular.resize(para.num_types);
  if (mode == NEP_Compile_Mode::CHARGE) {
    // qNEP's generic ParaMB uses the first, uniform cutoff value.
    for (int t = 0; t < para.num_types; ++t) {
      c.rc_radial[t] = para.rc_radial[0];
      c.rc_angular[t] = para.rc_angular[0];
    }
  } else {
    for (int t = 0; t < para.num_types; ++t) {
      c.rc_radial[t] = para.rc_radial[t];
      c.rc_angular[t] = para.rc_angular[t];
    }
  }

  if (
    mode == NEP_Compile_Mode::VDW ||
    mode == NEP_Compile_Mode::CHARGE_VDW) {
    if (c6_ref_sqrt == nullptr) {
      warning_compile("vdW specialization requires C6 reference values.");
      return NEP_Compile_Config();
    }
    c.c6_ref_sqrt.assign(c6_ref_sqrt, c6_ref_sqrt + para.num_types);
  }

  return c;
}

#if !defined(USE_HIP) && !defined(_WIN32)

bool validate_nep_compile_config(const NEP_Compile_Config& c)
{
  if (c.num_types <= 0 || c.ann_dim <= 0 || c.num_neurons1 <= 0) {
    warning_compile("Invalid NEP model dimensions.");
    return false;
  }
  if (
    static_cast<int>(c.rc_radial.size()) != c.num_types ||
    static_cast<int>(c.rc_angular.size()) != c.num_types) {
    warning_compile("Invalid cutoff data.");
    return false;
  }
  if (c.num_hidden_layers != 1 && c.num_hidden_layers != 2) {
    warning_compile("Only one or two hidden ANN layers are supported.");
    return false;
  }
  if (
    (c.mode == NEP_Compile_Mode::CHARGE ||
     c.mode == NEP_Compile_Mode::VDW ||
     c.mode == NEP_Compile_Mode::CHARGE_VDW) &&
    c.num_hidden_layers != 1) {
    warning_compile(
      "qNEP/vdW/charge-vdW specialization requires one hidden ANN layer.");
    return false;
  }
  if (
    (c.mode == NEP_Compile_Mode::VDW ||
     c.mode == NEP_Compile_Mode::CHARGE_VDW) &&
    static_cast<int>(c.c6_ref_sqrt.size()) != c.num_types) {
    warning_compile("vdW specialization requires C6 reference values.");
    return false;
  }
  if (c.mode == NEP_Compile_Mode::TNEP && c.model_type != 1 && c.model_type != 2) {
    warning_compile("TNEP specialization requires model_type 1 or 2.");
    return false;
  }
  return true;
}

std::string make_nep_compile_config_text(const NEP_Compile_Config& c)
{
  const int num_abc = (c.L_max + 1) * (c.L_max + 1) - 1;
  const int dim_radial = c.n_max_radial + 1;
  const bool common_radial = all_equal(c.rc_radial);
  const bool common_angular = all_equal(c.rc_angular);

  std::vector<float> c6 = c.c6_ref_sqrt;
  if (c6.empty()) c6.assign(c.num_types, 0.0f);

  std::ostringstream output;
  output << "#pragma once\n";
  output << "#define NEP_SPECIAL_CONFIG_INTERFACE_VERSION "
         << NEP_SPECIALIZED_INTERFACE_VERSION << "\n";
  output << "#define NEP_MODEL_NEP 0\n";
  output << "#define NEP_MODEL_CHARGE 1\n";
  output << "#define NEP_MODEL_VDW 2\n";
  output << "#define NEP_MODEL_CHARGE_VDW 3\n";
  output << "#define NEP_MODEL_TNEP 4\n";
  output << "#define NEP_MODEL_MODE_JIT " << static_cast<int>(c.mode) << "\n";
  output << "#define MODEL_TYPE_JIT " << c.model_type << "\n";
  output << "#define NUM_TYPES_JIT " << c.num_types << "\n";
  output << "#define ANN_DIM_JIT " << c.ann_dim << "\n";
  output << "#define NUM_NEURONS1_JIT " << c.num_neurons1 << "\n";
  output << "#define NUM_NEURONS2_JIT " << c.num_neurons2 << "\n";
  output << "#define NUM_HIDDEN_LAYERS_JIT " << c.num_hidden_layers << "\n";
  output << "#define ONE_ANN_NO_BIAS_JIT " << c.one_ann_no_bias << "\n";
  output << "#define ANN_SET_SIZE_JIT " << c.ann_set_size << "\n";
  output << "#define C_OFFSET_JIT " << c.c_offset << "\n";
  output << "#define N_MAX_RADIAL_JIT " << c.n_max_radial << "\n";
  output << "#define N_MAX_ANGULAR_JIT " << c.n_max_angular << "\n";
  output << "#define BASIS_SIZE_RADIAL_JIT " << c.basis_size_radial << "\n";
  output << "#define BASIS_SIZE_ANGULAR_JIT " << c.basis_size_angular << "\n";
  output << "#define L_MAX_JIT " << c.L_max << "\n";
  output << "#define DIM_RADIAL_JIT " << dim_radial << "\n";
  output << "#define DIM_ANGULAR_JIT " << c.dim_angular << "\n";
  output << "#define NUM_ABC_JIT " << num_abc << "\n";
  output << "#define HAS_Q_222_JIT " << c.has_q_222 << "\n";
  output << "#define HAS_Q_1111_JIT " << c.has_q_1111 << "\n";
  output << "#define HAS_Q_112_JIT " << c.has_q_112 << "\n";
  output << "#define HAS_Q_123_JIT " << c.has_q_123 << "\n";
  output << "#define HAS_Q_233_JIT " << c.has_q_233 << "\n";
  output << "#define HAS_Q_134_JIT " << c.has_q_134 << "\n";
  output << "#define NUM_L_JIT " << c.num_L << "\n";
  output << "#define NUM_C_RADIAL_JIT " << c.num_c_radial << "\n";
  output << "#define DESCRIPTOR_USE_CJ_JIT "
         << (c.descriptor_use_cj ? 1 : 0) << "\n";
  output << "#define RC_RADIAL_COMMON_JIT " << (common_radial ? 1 : 0) << "\n";
  output << "#define RC_ANGULAR_COMMON_JIT " << (common_angular ? 1 : 0) << "\n";
  output << "#define RC_RADIAL_COMMON_VALUE_JIT "
         << float_literal(c.rc_radial[0]) << "\n";
  output << "#define RC_ANGULAR_COMMON_VALUE_JIT "
         << float_literal(c.rc_angular[0]) << "\n";
  output << "#define C6_SCALING_FACTOR_JIT 1.000000000e-01f\n";
  output << "static __device__ __constant__ float RC_RADIAL_JIT[NUM_TYPES_JIT] = "
         << make_float_array(c.rc_radial) << ";\n";
  output << "static __device__ __constant__ float RC_ANGULAR_JIT[NUM_TYPES_JIT] = "
         << make_float_array(c.rc_angular) << ";\n";
  output << "static __device__ __constant__ float C6_REF_SQRT_JIT[NUM_TYPES_JIT] = "
         << make_float_array(c6) << ";\n";
  return output.str();
}

#endif
