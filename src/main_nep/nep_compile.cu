/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "nep_compile.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#if !defined(USE_HIP) && !defined(_WIN32)
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#endif

namespace
{
void fail_compile(const std::string& message)
{
  std::cerr << "NEP training runtime compilation error:\n    " << message << std::endl;
  std::exit(1);
}

#if !defined(USE_HIP) && !defined(_WIN32)
bool file_exists(const std::string& filename)
{
  std::ifstream input(filename.c_str());
  return input.good();
}

std::string shell_quote(const std::string& text)
{
  std::string result = "'";
  for (char c : text) {
    if (c == '\'') {
      result += "'\\''";
    } else {
      result += c;
    }
  }
  result += "'";
  return result;
}

std::string read_text_file(const std::string& filename)
{
  std::ifstream input(filename.c_str());
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

std::string get_gpumd_source_dir()
{
  const char* source_env = std::getenv("GPUMD_SRC");
  if (source_env != nullptr && source_env[0] != '\0') {
    std::string source_dir(source_env);
    if (file_exists(source_dir + "/utilities/nep_utilities.cuh")) {
      return source_dir;
    }
    fail_compile("GPUMD_SRC does not point to the GPUMD src directory.");
  }

  char path[PATH_MAX];
  const ssize_t size = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (size <= 0) {
    fail_compile("Cannot determine the location of the nep executable.");
  }
  path[size] = '\0';
  std::string executable(path);
  const size_t pos = executable.find_last_of('/');
  if (pos == std::string::npos) {
    fail_compile("Cannot determine the GPUMD source directory.");
  }
  const std::string source_dir = executable.substr(0, pos);
  if (!file_exists(source_dir + "/utilities/nep_utilities.cuh")) {
    fail_compile(
      "Cannot find GPUMD headers next to the nep executable. "
      "Run nep from the compiled src tree or set GPUMD_SRC.");
  }
  return source_dir;
}

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

void write_specialized_source(const std::string& filename, const NEP_Compile_Config& c)
{
  if (c.num_types <= 0 ||
      static_cast<int>(c.rc_radial.size()) != c.num_types ||
      static_cast<int>(c.rc_angular.size()) != c.num_types) {
    fail_compile("Invalid cutoff data for runtime specialization.");
  }
  std::ofstream output(filename.c_str());
  if (!output.is_open()) {
    fail_compile("Cannot create the specialized CUDA source file.");
  }

  const int num_abc = (c.L_max + 1) * (c.L_max + 1) - 1;
  const int dim_radial = c.n_max_radial + 1;
  bool common_radial_cutoff = true;
  bool common_angular_cutoff = true;
  for (int t = 1; t < c.num_types; ++t) {
    if (c.rc_radial[t] != c.rc_radial[0]) {
      common_radial_cutoff = false;
    }
    if (c.rc_angular[t] != c.rc_angular[0]) {
      common_angular_cutoff = false;
    }
  }

  output << R"JIT(
#include <cuda_runtime.h>
#include <cmath>
#include "utilities/nep_utilities.cuh"

)JIT";

  output << "constexpr int NUM_TYPES_JIT = " << c.num_types << ";\n";
  output << "constexpr int NUM_TYPES_SQ_JIT = " << c.num_types * c.num_types << ";\n";
  output << "constexpr int ANN_DIM_JIT = " << c.ann_dim << ";\n";
  output << "constexpr int NUM_NEURONS1_JIT = " << c.num_neurons1 << ";\n";
  output << "constexpr int NUM_NEURONS2_JIT = " << c.num_neurons2 << ";\n";
  output << "constexpr int NUM_HIDDEN_LAYERS_JIT = " << c.num_hidden_layers << ";\n";
  output << "constexpr int ONE_ANN_NO_BIAS_JIT = " << c.one_ann_no_bias << ";\n";
  output << "constexpr int N_MAX_RADIAL_JIT = " << c.n_max_radial << ";\n";
  output << "constexpr int N_MAX_ANGULAR_JIT = " << c.n_max_angular << ";\n";
  output << "constexpr int BASIS_SIZE_RADIAL_JIT = " << c.basis_size_radial << ";\n";
  output << "constexpr int BASIS_SIZE_ANGULAR_JIT = " << c.basis_size_angular << ";\n";
  output << "constexpr int L_MAX_JIT = " << c.L_max << ";\n";
  output << "constexpr int DIM_RADIAL_JIT = " << dim_radial << ";\n";
  output << "constexpr int DIM_ANGULAR_JIT = " << c.dim_angular << ";\n";
  output << "constexpr int NUM_ABC_JIT = " << num_abc << ";\n";
  output << "constexpr int HAS_Q_222_JIT = " << c.has_q_222 << ";\n";
  output << "constexpr int HAS_Q_1111_JIT = " << c.has_q_1111 << ";\n";
  output << "constexpr int HAS_Q_112_JIT = " << c.has_q_112 << ";\n";
  output << "constexpr int HAS_Q_123_JIT = " << c.has_q_123 << ";\n";
  output << "constexpr int HAS_Q_233_JIT = " << c.has_q_233 << ";\n";
  output << "constexpr int HAS_Q_134_JIT = " << c.has_q_134 << ";\n";
  output << "constexpr int NUM_L_JIT = " << c.num_L << ";\n";
  output << "constexpr int NUM_C_RADIAL_JIT = " << c.num_c_radial << ";\n";
  output << "constexpr int NUM_VARIABLES_DESCRIPTOR_JIT = " << c.number_of_variables_descriptor << ";\n";
  if (common_radial_cutoff) {
    output << "constexpr float RC_RADIAL_COMMON_JIT = " << float_literal(c.rc_radial[0]) << ";\n";
  } else {
    output << "__device__ __constant__ float RC_RADIAL_JIT[NUM_TYPES_JIT] = "
           << make_float_array(c.rc_radial) << ";\n";
  }
  if (common_angular_cutoff) {
    output << "constexpr float RC_ANGULAR_COMMON_JIT = " << float_literal(c.rc_angular[0]) << ";\n";
  } else {
    output << "__device__ __constant__ float RC_ANGULAR_JIT[NUM_TYPES_JIT] = "
           << make_float_array(c.rc_angular) << ";\n";
  }
  output << "\n";

  if (common_radial_cutoff) {
    output << "__device__ __forceinline__ float get_rc_radial_jit(const int, const int)\n";
    output << "{ return RC_RADIAL_COMMON_JIT; }\n\n";
  } else {
    output << "__device__ __forceinline__ float get_rc_radial_jit(const int t1, const int t2)\n";
    output << "{ return 0.5f * (RC_RADIAL_JIT[t1] + RC_RADIAL_JIT[t2]); }\n\n";
  }
  if (common_angular_cutoff) {
    output << "__device__ __forceinline__ float get_rc_angular_jit(const int, const int)\n";
    output << "{ return RC_ANGULAR_COMMON_JIT; }\n\n";
  } else {
    output << "__device__ __forceinline__ float get_rc_angular_jit(const int t1, const int t2)\n";
    output << "{ return 0.5f * (RC_ANGULAR_JIT[t1] + RC_ANGULAR_JIT[t2]); }\n\n";
  }

  output << R"JIT(
__device__ __forceinline__ const float* get_c(const float* parameters)
{
  return parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT + 1;
}

#ifdef TRAIN_CUTOFF
__device__ __forceinline__ const float* get_train_cutoff(const float* parameters)
{
  return get_c(parameters) + NUM_VARIABLES_DESCRIPTOR_JIT;
}
#endif

__global__ void descriptor_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c(parameters);
#ifdef TRAIN_CUTOFF
  const float* rc_train = get_train_cutoff(parameters);
#endif
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_RADIAL_JIT] = {0.0f};

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float x12 = x12_all[index];
    const float y12 = y12_all[index];
    const float z12 = z12_all[index];
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    float rc = get_rc_radial_jit(t1, t2);
#ifdef TRAIN_CUTOFF
    rc -= ((1.0f + tanh(rc_train[t1])) + (1.0f + tanh(rc_train[t2]))) * 0.25f;
#endif
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn(BASIS_SIZE_RADIAL_JIT, rcinv, d12, fc12, fn12);
#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gn12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
#ifdef USE_CJ
        const int c_index = (n * (BASIS_SIZE_RADIAL_JIT + 1) + k) * NUM_TYPES_JIT + t2;
#else
        int c_index = (n * (BASIS_SIZE_RADIAL_JIT + 1) + k) * NUM_TYPES_SQ_JIT;
        c_index += t1 * NUM_TYPES_JIT + t2;
#endif
        gn12 += fn12[k] * c[c_index];
      }
      q[n] += gn12;
    }
  }
#pragma unroll
  for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
    descriptors[n1 + n * N] = q[n];
  }
}

__global__ void descriptor_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c(parameters);
#ifdef TRAIN_CUTOFF
  const float* rc_train = get_train_cutoff(parameters);
#endif
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_ANGULAR_JIT] = {0.0f};

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
    float s[NUM_ABC_JIT] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = NN_sum[n1] + i1;
      const int n2 = NL[index];
      const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
      const float x12 = x12_all[index];
      const float y12 = y12_all[index];
      const float z12 = z12_all[index];
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float rc = get_rc_angular_jit(t1, t2);
#ifdef TRAIN_CUTOFF
      rc -= ((1.0f + tanh(rc_train[NUM_TYPES_JIT + t1])) +
             (1.0f + tanh(rc_train[NUM_TYPES_JIT + t2]))) * 0.25f;
#endif
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
      find_fn(BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
#ifdef USE_CJ
        const int c_index = NUM_C_RADIAL_JIT +
          (n * (BASIS_SIZE_ANGULAR_JIT + 1) + k) * NUM_TYPES_JIT + t2;
#else
        int c_index = NUM_C_RADIAL_JIT +
          (n * (BASIS_SIZE_ANGULAR_JIT + 1) + k) * NUM_TYPES_SQ_JIT;
        c_index += t1 * NUM_TYPES_JIT + t2;
#endif
        gn12 += fn12[k] * c[c_index];
      }
      accumulate_s(L_MAX_JIT, d12, x12, y12, z12, gn12, s);
    }
    find_q(
      L_MAX_JIT,
      HAS_Q_222_JIT,
      HAS_Q_1111_JIT,
      HAS_Q_112_JIT,
      HAS_Q_123_JIT,
      HAS_Q_233_JIT,
      HAS_Q_134_JIT,
      N_MAX_ANGULAR_JIT + 1,
      n,
      s,
      q);
#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[(n * NUM_ABC_JIT + abc) * N + n1] = s[abc];
    }
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int l = 0; l < NUM_L_JIT; ++l) {
      const int ln = l * (N_MAX_ANGULAR_JIT + 1) + n;
      descriptors[n1 + (DIM_RADIAL_JIT + ln) * N] = q[ln];
    }
  }
}

__global__ void ann_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  float F = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  const float* wb = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* b = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;

  if (NUM_HIDDEN_LAYERS_JIT == 2) {
    apply_ann_two_layers(
      ANN_DIM_JIT,
      NUM_NEURONS1_JIT,
      NUM_NEURONS2_JIT,
      wb,
      wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1 + NUM_NEURONS2_JIT),
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1 + NUM_NEURONS2_JIT) + NUM_NEURONS2_JIT,
      b,
      q,
      F,
      Fp);
  } else {
    apply_ann_one_layer(
      ANN_DIM_JIT,
      NUM_NEURONS1_JIT,
      wb,
      wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
      b,
      q,
      F,
      Fp);
  }

  pe[n1] = F;
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__global__ void force_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c(parameters);
#ifdef TRAIN_CUTOFF
  const float* rc_train = get_train_cutoff(parameters);
#endif
  float sxx = 0.0f, syy = 0.0f, szz = 0.0f;
  float sxy = 0.0f, syz = 0.0f, szx = 0.0f;
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {x12_all[index], y12_all[index], z12_all[index]};
    const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    float rc = get_rc_radial_jit(t1, t2);
#ifdef TRAIN_CUTOFF
    rc -= ((1.0f + tanh(rc_train[t1])) + (1.0f + tanh(rc_train[t2]))) * 0.25f;
#endif
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    float fnp12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn_and_fnp(BASIS_SIZE_RADIAL_JIT, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gnp12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
#ifdef USE_CJ
        const int c_index = (n * (BASIS_SIZE_RADIAL_JIT + 1) + k) * NUM_TYPES_JIT + t2;
#else
        int c_index = (n * (BASIS_SIZE_RADIAL_JIT + 1) + k) * NUM_TYPES_SQ_JIT;
        c_index += t1 * NUM_TYPES_JIT + t2;
#endif
        gnp12 += fnp12[k] * c[c_index];
      }
      const float tmp12 = Fp[n1 + n * N] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);
    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + N * 2] += szz;
  virial[n1 + N * 3] = sxy;
  virial[n1 + N * 4] = syz;
  virial[n1 + N * 5] = szx;
}

__global__ void force_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp_in,
  const float* sum_fxyz_in,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c(parameters);
#ifdef TRAIN_CUTOFF
  const float* rc_train = get_train_cutoff(parameters);
#endif
  float sxx = 0.0f, syy = 0.0f, szz = 0.0f;
  float sxy = 0.0f, syz = 0.0f, szx = 0.0f;

  float Fp[DIM_ANGULAR_JIT] = {0.0f};
  float sum_fxyz[(N_MAX_ANGULAR_JIT + 1) * NUM_OF_ABC];
#pragma unroll
  for (int d = 0; d < DIM_ANGULAR_JIT; ++d) {
    Fp[d] = Fp_in[(DIM_RADIAL_JIT + d) * N + n1];
  }
#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[n * NUM_OF_ABC + abc] = sum_fxyz_in[(n * NUM_ABC_JIT + abc) * N + n1];
    }
  }

  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {x12_all[index], y12_all[index], z12_all[index]};
    const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float rc = get_rc_angular_jit(t1, t2);
#ifdef TRAIN_CUTOFF
    rc -= ((1.0f + tanh(rc_train[NUM_TYPES_JIT + t1])) +
           (1.0f + tanh(rc_train[NUM_TYPES_JIT + t2]))) * 0.25f;
#endif
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
    float fnp12[BASIS_SIZE_ANGULAR_JIT + 1];
    find_fn_and_fnp(BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
#ifdef USE_CJ
        const int c_index = NUM_C_RADIAL_JIT +
          (n * (BASIS_SIZE_ANGULAR_JIT + 1) + k) * NUM_TYPES_JIT + t2;
#else
        int c_index = NUM_C_RADIAL_JIT +
          (n * (BASIS_SIZE_ANGULAR_JIT + 1) + k) * NUM_TYPES_SQ_JIT;
        c_index += t1 * NUM_TYPES_JIT + t2;
#endif
        const float c_value = c[c_index];
        gn12 += fn12[k] * c_value;
        gnp12 += fnp12[k] * c_value;
      }
      accumulate_f12(
        L_MAX_JIT,
        HAS_Q_222_JIT,
        HAS_Q_1111_JIT,
        HAS_Q_112_JIT,
        HAS_Q_123_JIT,
        HAS_Q_233_JIT,
        HAS_Q_134_JIT,
        NUM_L_JIT,
        n,
        N_MAX_ANGULAR_JIT + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp,
        sum_fxyz,
        f12);
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);
    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + N * 2] += szz;
  virial[n1 + N * 3] += sxy;
  virial[n1 + N * 4] += syz;
  virial[n1 + N * 5] += szx;
}

extern "C" int nep_train_launch_descriptor_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters, float* descriptors)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_radial_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_descriptor_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters,
  float* descriptors, float* sum_fxyz)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_angular_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors, sum_fxyz);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann(
  int N, const int* type, const float* descriptors, const float* q_scaler,
  const float* parameters, float* pe, float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_jit<<<grid_size, block_size>>>(N, type, descriptors, q_scaler, parameters, pe, Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters,
  const float* Fp, float* fx, float* fy, float* fz, float* virial)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_radial_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, Fp, fx, fy, fz, virial);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters,
  const float* Fp, const float* sum_fxyz, float* fx, float* fy, float* fz, float* virial)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_angular_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, Fp, sum_fxyz, fx, fy, fz, virial);
  return static_cast<int>(cudaGetLastError());
}
)JIT";

  output.close();
}

template <typename T>
T load_symbol(void* library, const char* name)
{
  dlerror();
  T function = reinterpret_cast<T>(dlsym(library, name));
  const char* error = dlerror();
  if (error != nullptr || function == nullptr) {
    fail_compile(std::string("Cannot find symbol ") + name + " in the specialized NEP library.");
  }
  return function;
}
#endif
} // namespace

NEP_Compile::NEP_Compile(const NEP_Compile_Config& config)
{
#if defined(USE_HIP)
  (void)config;
  fail_compile("nep_compile on is not supported by the HIP build.");
#elif defined(_WIN32)
  (void)config;
  fail_compile("nep_compile on is supported on Linux only in this first training version.");
#else
  compile(config);
#endif
}

NEP_Compile::~NEP_Compile()
{
#if !defined(USE_HIP) && !defined(_WIN32)
  if (library_ != nullptr) {
    dlclose(library_);
    library_ = nullptr;
  }
#endif
  cleanup_files();
}

void NEP_Compile::compile(const NEP_Compile_Config& config)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  if (config.num_types <= 0 || config.ann_dim <= 0 || config.num_neurons1 <= 0) {
    fail_compile("Invalid NEP model dimensions for runtime specialization.");
  }
  if (config.num_hidden_layers != 1 && config.num_hidden_layers != 2) {
    fail_compile("Runtime specialization supports one or two hidden ANN layers.");
  }

  const std::string source_dir = get_gpumd_source_dir();
  char temp_template[] = "/tmp/gpumd-nep-train-compile-XXXXXX";
  char* temp_dir = mkdtemp(temp_template);
  if (temp_dir == nullptr) {
    fail_compile("Cannot create a temporary directory in /tmp.");
  }
  temp_dir_ = temp_dir;
  source_file_ = temp_dir_ + "/nep_train_specialized.cu";
  library_file_ = temp_dir_ + "/nep_train_specialized.so";
  log_file_ = temp_dir_ + "/build.log";
  write_specialized_source(source_file_, config);

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    cleanup_files();
    fail_compile(cudaGetErrorString(error));
  }
  cudaDeviceProp properties;
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    cleanup_files();
    fail_compile(cudaGetErrorString(error));
  }

  const char* cuda_compiler_env = std::getenv("CUDACXX");
  const std::string cuda_compiler =
    (cuda_compiler_env != nullptr && cuda_compiler_env[0] != '\0') ? cuda_compiler_env : "nvcc";

  std::ostringstream command;
  command << shell_quote(cuda_compiler)
          << " -std=c++14 -O3"
          << " -arch=sm_" << properties.major << properties.minor
          << " -shared -Xcompiler=-fPIC"
          << " -I" << shell_quote(source_dir);
#ifdef USE_CJ
  command << " -DUSE_CJ";
#endif
#ifdef TRAIN_CUTOFF
  command << " -DTRAIN_CUTOFF";
#endif
  command << " " << shell_quote(source_file_)
          << " -o " << shell_quote(library_file_)
          << " > " << shell_quote(log_file_) << " 2>&1";

  printf("Compile specialized NEP training kernels (sm_%d%d).\n", properties.major, properties.minor);
  printf(
    "    types=%d, dim=%d, neurons=%d, n_max=(%d,%d), basis=(%d,%d), l_max=%d.\n",
    config.num_types,
    config.ann_dim,
    config.num_neurons1,
    config.n_max_radial,
    config.n_max_angular,
    config.basis_size_radial,
    config.basis_size_angular,
    config.L_max);
  fflush(stdout);

  const int status = std::system(command.str().c_str());
  if (status != 0) {
    const std::string log = read_text_file(log_file_);
    std::cerr << log;
    cleanup_files();
    fail_compile("nvcc failed while compiling specialized NEP training kernels.");
  }

  library_ = dlopen(library_file_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr) {
    const std::string message = dlerror();
    cleanup_files();
    fail_compile("Cannot load the specialized NEP training library: " + message);
  }

  descriptor_radial_ = load_symbol<DescriptorRadialFunction>(library_, "nep_train_launch_descriptor_radial");
  descriptor_angular_ = load_symbol<DescriptorAngularFunction>(library_, "nep_train_launch_descriptor_angular");
  ann_ = load_symbol<AnnFunction>(library_, "nep_train_launch_ann");
  force_radial_ = load_symbol<ForceRadialFunction>(library_, "nep_train_launch_force_radial");
  force_angular_ = load_symbol<ForceAngularFunction>(library_, "nep_train_launch_force_angular");

  printf("Specialized NEP training kernels are ready.\n");
  fflush(stdout);
#else
  (void)config;
#endif
}


void NEP_Compile::launch_descriptor_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters, float* descriptors)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  const int error = descriptor_radial_(N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors);
  if (error != static_cast<int>(cudaSuccess)) fail_compile(cudaGetErrorString(static_cast<cudaError_t>(error)));
#else
  (void)N; (void)NN_sum; (void)NN; (void)NL; (void)type; (void)x12; (void)y12; (void)z12; (void)parameters; (void)descriptors;
#endif
}

void NEP_Compile::launch_descriptor_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters,
  float* descriptors, float* sum_fxyz)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  const int error = descriptor_angular_(N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors, sum_fxyz);
  if (error != static_cast<int>(cudaSuccess)) fail_compile(cudaGetErrorString(static_cast<cudaError_t>(error)));
#else
  (void)N; (void)NN_sum; (void)NN; (void)NL; (void)type; (void)x12; (void)y12; (void)z12; (void)parameters; (void)descriptors; (void)sum_fxyz;
#endif
}

void NEP_Compile::launch_ann(
  int N, const int* type, const float* descriptors, const float* q_scaler,
  const float* parameters, float* pe, float* Fp)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  const int error = ann_(N, type, descriptors, q_scaler, parameters, pe, Fp);
  if (error != static_cast<int>(cudaSuccess)) fail_compile(cudaGetErrorString(static_cast<cudaError_t>(error)));
#else
  (void)N; (void)type; (void)descriptors; (void)q_scaler; (void)parameters; (void)pe; (void)Fp;
#endif
}

void NEP_Compile::launch_force_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters,
  const float* Fp, float* fx, float* fy, float* fz, float* virial)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  const int error = force_radial_(N, NN_sum, NN, NL, type, x12, y12, z12, parameters, Fp, fx, fy, fz, virial);
  if (error != static_cast<int>(cudaSuccess)) fail_compile(cudaGetErrorString(static_cast<cudaError_t>(error)));
#else
  (void)N; (void)NN_sum; (void)NN; (void)NL; (void)type; (void)x12; (void)y12; (void)z12; (void)parameters; (void)Fp; (void)fx; (void)fy; (void)fz; (void)virial;
#endif
}

void NEP_Compile::launch_force_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12, const float* parameters,
  const float* Fp, const float* sum_fxyz, float* fx, float* fy, float* fz, float* virial)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  const int error = force_angular_(N, NN_sum, NN, NL, type, x12, y12, z12, parameters, Fp, sum_fxyz, fx, fy, fz, virial);
  if (error != static_cast<int>(cudaSuccess)) fail_compile(cudaGetErrorString(static_cast<cudaError_t>(error)));
#else
  (void)N; (void)NN_sum; (void)NN; (void)NL; (void)type; (void)x12; (void)y12; (void)z12; (void)parameters; (void)Fp; (void)sum_fxyz; (void)fx; (void)fy; (void)fz; (void)virial;
#endif
}

void NEP_Compile::cleanup_files()
{
  if (!source_file_.empty()) std::remove(source_file_.c_str());
  if (!library_file_.empty()) std::remove(library_file_.c_str());
  if (!log_file_.empty()) std::remove(log_file_.c_str());
#if !defined(USE_HIP) && !defined(_WIN32)
  if (!temp_dir_.empty()) rmdir(temp_dir_.c_str());
#endif
}
