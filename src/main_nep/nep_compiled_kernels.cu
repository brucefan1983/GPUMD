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

#include "nep_compiled_kernels.cuh"

bool NEP_Compiled_Kernels::load(NEP_Compile& library)
{
  bool valid = true;
  valid &= library.load_symbol("nep_train_launch_descriptor_radial", descriptor_radial_);
  valid &= library.load_symbol("nep_train_launch_descriptor_angular", descriptor_angular_);
  valid &= library.load_symbol("nep_train_launch_force_radial", force_radial_);
  valid &= library.load_symbol("nep_train_launch_force_angular", force_angular_);
  return valid;
}

void NEP_Compiled_Kernels::launch_descriptor_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors)
{
  check_nep_launch(
    descriptor_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors),
    "descriptor_radial_jit");
}

void NEP_Compiled_Kernels::launch_descriptor_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  check_nep_launch(
    descriptor_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, descriptors, sum_fxyz),
    "descriptor_angular_jit");
}

bool NEP_Compiled_BEC::load(NEP_Compile& library)
{
  bool valid = true;
  valid &= library.load_symbol("nep_train_launch_bec_radial", bec_radial_);
  valid &= library.load_symbol("nep_train_launch_bec_angular", bec_angular_);
  return valid;
}

void NEP_Compiled_BEC::launch_bec_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* charge_derivative, float* bec)
{
  check_nep_launch(
    bec_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, charge_derivative, bec),
    "bec_radial_jit");
}

void NEP_Compiled_BEC::launch_bec_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* charge_derivative,
  const float* sum_fxyz, float* bec)
{
  check_nep_launch(
    bec_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, charge_derivative, sum_fxyz, bec),
    "bec_angular_jit");
}
