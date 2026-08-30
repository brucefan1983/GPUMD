/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include <string>
#include <vector>

struct NEP_Compile_Config
{
  int num_types = 0;
  int ann_dim = 0;
  int num_neurons1 = 0;
  int num_neurons2 = 0;
  int num_hidden_layers = 0;
  int one_ann_no_bias = 0;
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
  int number_of_variables_descriptor = 0;
  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
};

class NEP_Compile
{
public:
  explicit NEP_Compile(const NEP_Compile_Config& config);
  ~NEP_Compile();

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

  void launch_ann(
    int N,
    const int* type,
    const float* descriptors,
    const float* q_scaler,
    const float* parameters,
    float* pe,
    float* Fp);

  void launch_force_radial(
    int N,
    const int* NN_sum,
    const int* NN,
    const int* NL,
    const int* type,
    const float* x12,
    const float* y12,
    const float* z12,
    const float* parameters,
    const float* Fp,
    float* fx,
    float* fy,
    float* fz,
    float* virial);

  void launch_force_angular(
    int N,
    const int* NN_sum,
    const int* NN,
    const int* NL,
    const int* type,
    const float* x12,
    const float* y12,
    const float* z12,
    const float* parameters,
    const float* Fp,
    const float* sum_fxyz,
    float* fx,
    float* fy,
    float* fz,
    float* virial);

private:
  using DescriptorRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*, const float*, const float*, const float*,
    const float*, float*);
  using DescriptorAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*, const float*, const float*, const float*,
    const float*, float*, float*);
  using AnnFunction = int (*)(
    int, const int*, const float*, const float*, const float*, float*, float*);
  using ForceRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*, const float*, const float*, const float*,
    const float*, const float*, float*, float*, float*, float*);
  using ForceAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*, const float*, const float*, const float*,
    const float*, const float*, const float*, float*, float*, float*, float*);

  void* library_ = nullptr;
  DescriptorRadialFunction descriptor_radial_ = nullptr;
  DescriptorAngularFunction descriptor_angular_ = nullptr;
  AnnFunction ann_ = nullptr;
  ForceRadialFunction force_radial_ = nullptr;
  ForceAngularFunction force_angular_ = nullptr;
  std::string temp_dir_;
  std::string source_file_;
  std::string library_file_;
  std::string log_file_;

  void compile(const NEP_Compile_Config& config);
  void cleanup_files();
};
