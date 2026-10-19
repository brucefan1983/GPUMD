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
#include "parameters.cuh"
#include <iostream>

namespace
{
void warning_compile(const std::string& message)
{
  std::cerr << "Warning: NEP training specialization disabled: "
            << message << std::endl;
}
}

NEP_Compile_Config make_nep_compile_config(
  const Parameters& para,
  const NEP_Compile_Mode mode,
  const float* c6_ref_sqrt)
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
  c.model_type = para.model_type;
  c.num_types = para.num_types;
  c.ann_dim = para.dim;
  c.num_neurons1 = para.num_neurons1;
  c.num_neurons2 = para.num_neurons2;
  c.num_hidden_layers = para.num_hidden_layers;
  c.one_ann_no_bias = para.number_of_variables_ann_1;
  c.ann_set_size = para.number_of_variables_ann;

  // All NEP-family parameter vectors place descriptor cij immediately after
  // the ANN parameter block. TNEP polarizability has two ANN parameter sets.
  c.c_offset = para.number_of_variables_ann;
  if (mode == NEP_Compile_Mode::TNEP && para.model_type == 2) {
    c.c_offset *= 2;
  }

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
