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
#include "nep_compile.cuh"

struct NEP_Compiled_Kernels
{
  using DescriptorRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*, const float*, float*);
  using DescriptorAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*, const float*, float*, float*);
  using ForceRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*, const float*, const float*,
    int, float*, float*, float*, float*);
  using ForceAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*, const float*, const float*, const float*,
    const float*, int, float*, float*, float*, float*);
  DescriptorRadialFunction descriptor_radial_ = nullptr;
  DescriptorAngularFunction descriptor_angular_ = nullptr;
  ForceRadialFunction force_radial_ = nullptr;
  ForceAngularFunction force_angular_ = nullptr;
  bool load(NEP_Compile& library);
  void launch_descriptor_radial(
    int N,
    const int* NN_sum,
    const int* NN,
    const int* NL,
    const int* type,
    const float* x12,
    const float* y12,
    const float* z12,
    const float* parameters,
    float* descriptors);
  void launch_descriptor_angular(
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
    float* sum_fxyz);
};

struct NEP_Compiled_BEC
{
  using BecRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*, float*);
  using BecAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*, const float*, float*);
  BecRadialFunction bec_radial_ = nullptr;
  BecAngularFunction bec_angular_ = nullptr;
  bool load(NEP_Compile& library);
  void launch_bec_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* charge_derivative, float* bec);
  void launch_bec_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* charge_derivative,
    const float* sum_fxyz, float* bec);
};
