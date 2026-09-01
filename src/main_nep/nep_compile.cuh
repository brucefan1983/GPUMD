/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>
#include <vector>

class Parameters;

enum class NEP_Compile_Mode {
  NEP = 0,
  CHARGE = 1,
  VDW = 2,
  CHARGE_VDW = 3,
  TNEP = 4
};

struct NEP_Compile_Config {
  NEP_Compile_Mode mode = NEP_Compile_Mode::NEP;
  int train_mode = 0;

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

  // Ordinary NEP/NEP-vdW follow USE_CJ when that build option is enabled.
  // qNEP/charge-vdW/TNEP remain pair-dependent, matching their generic paths.
  bool descriptor_use_cj = false;

  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
  std::vector<float> c6_ref_sqrt;
};

NEP_Compile_Config make_nep_compile_config(
  const Parameters& para,
  NEP_Compile_Mode mode,
  const float* c6_ref_sqrt = nullptr);

class NEP_Compile
{
public:
  explicit NEP_Compile(const NEP_Compile_Config& config);
  ~NEP_Compile();

  NEP_Compile(const NEP_Compile&) = delete;
  NEP_Compile& operator=(const NEP_Compile&) = delete;

  bool is_valid() const { return valid_; }

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

  void launch_ann_temperature(
    int N,
    const int* type,
    const float* descriptors,
    float* q_scaler,
    const float* temperature,
    const float* parameters,
    float* pe,
    float* Fp);

  void launch_ann_charge(
    int N,
    const int* type,
    const float* descriptors,
    const float* q_scaler,
    const float* parameters,
    float* pe,
    float* Fp,
    float* charge,
    float* charge_derivative);

  void launch_ann_vdw(
    int N,
    const int* type,
    const float* descriptors,
    const float* q_scaler,
    const float* parameters,
    float* pe,
    float* Fp,
    float* C6,
    float* C6_derivative);

  void launch_ann_charge_vdw(
    int N,
    const int* type,
    const float* descriptors,
    const float* q_scaler,
    const float* parameters,
    float* pe,
    float* Fp,
    float* charge,
    float* charge_derivative,
    float* C6,
    float* C6_derivative);

  void launch_ann_tnep_pol(
    int N,
    const int* type,
    const float* descriptors,
    const float* q_scaler,
    const float* parameters,
    float* virial,
    float* Fp);

  void launch_bec_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* charge_derivative, float* bec);
  void launch_bec_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* charge_derivative,
    const float* sum_fxyz, float* bec);

  // Model-specific public interfaces keep the original training code readable.
  void launch_force_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp, const float* sum_fxyz,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_charge_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    const float* charge_derivative, const float* D_real,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_charge_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    const float* charge_derivative, const float* D_real,
    const float* sum_fxyz,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_vdw_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    const float* C6_derivative, const float* D_C6,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_vdw_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    const float* C6_derivative, const float* D_C6,
    const float* sum_fxyz,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_charge_vdw_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    const float* charge_derivative, const float* D_real,
    const float* C6_derivative, const float* D_C6,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_charge_vdw_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    const float* charge_derivative, const float* D_real,
    const float* C6_derivative, const float* D_C6,
    const float* sum_fxyz,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_tnep_radial(
    bool is_dipole,
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    float* fx, float* fy, float* fz, float* virial);

  void launch_force_tnep_angular(
    bool is_dipole,
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp, const float* sum_fxyz,
    float* fx, float* fy, float* fz, float* virial);

private:
  bool valid_ = false;
  using DescriptorRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*, const float*, float*);

  using DescriptorAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*, const float*, float*, float*);

  using AnnNepFunction =
    int (*)(int, const int*, const float*, const float*, const float*, float*, float*);

  using AnnTemperatureFunction =
    int (*)(int, const int*, const float*, float*, const float*, const float*, float*, float*);

  using AnnChargeFunction =
    int (*)(int, const int*, const float*, const float*, const float*,
            float*, float*, float*, float*);

  using AnnChargeVdwFunction =
    int (*)(int, const int*, const float*, const float*, const float*,
            float*, float*, float*, float*, float*, float*);

  using AnnTnepPolFunction =
    int (*)(int, const int*, const float*, const float*, const float*, float*, float*);

  using BecRadialFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*, float*);
  using BecAngularFunction = int (*)(
    int, const int*, const int*, const int*, const int*,
    const float*, const float*, const float*,
    const float*, const float*, const float*, float*);

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

  void* library_ = nullptr;
  DescriptorRadialFunction descriptor_radial_ = nullptr;
  DescriptorAngularFunction descriptor_angular_ = nullptr;
  AnnNepFunction ann_nep_ = nullptr;
  AnnTemperatureFunction ann_temperature_ = nullptr;
  AnnChargeFunction ann_charge_ = nullptr;
  AnnChargeFunction ann_vdw_ = nullptr;
  AnnChargeVdwFunction ann_charge_vdw_ = nullptr;
  AnnTnepPolFunction ann_tnep_pol_ = nullptr;
  BecRadialFunction bec_radial_ = nullptr;
  BecAngularFunction bec_angular_ = nullptr;
  ForceRadialFunction force_radial_ = nullptr;
  ForceAngularFunction force_angular_ = nullptr;

  std::string temp_dir_;
  std::string config_file_;
  std::string library_file_;
  std::string log_file_;

  bool compile(const NEP_Compile_Config& config);
  void cleanup_files();
  void check_launch(int error_code, const char* kernel_name);
};
