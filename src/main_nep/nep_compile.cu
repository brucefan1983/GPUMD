/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "nep_compile.cuh"
#include "parameters.cuh"

#include <cerrno>
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
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace
{
void warning_compile(const std::string& message)
{
  std::cerr << "Warning: NEP training specialization disabled: "
            << message << std::endl;
}

const char* mode_name(const NEP_Compile_Mode mode)
{
  switch (mode) {
    case NEP_Compile_Mode::NEP:
      return "NEP";
    case NEP_Compile_Mode::CHARGE:
      return "qNEP";
    case NEP_Compile_Mode::VDW:
      return "NEP-vdW";
    case NEP_Compile_Mode::CHARGE_VDW:
      return "NEP-charge-vdW";
    case NEP_Compile_Mode::TNEP:
      return "TNEP";
    default:
      return "unknown";
  }
}

#if !defined(USE_HIP) && !defined(_WIN32)

bool file_exists(const std::string& filename)
{
  std::ifstream input(filename.c_str(), std::ios::binary);
  return input.good();
}

std::string shell_quote(const std::string& text)
{
  std::string result = "'";
  for (const char c : text) {
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
  std::ifstream input(filename.c_str(), std::ios::binary);
  if (!input.is_open()) {
    return std::string();
  }
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

std::string get_gpumd_source_dir()
{
  const char* source_env = std::getenv("GPUMD_SRC");
  if (source_env != nullptr && source_env[0] != '\0') {
    const std::string source_dir(source_env);
    if (
      file_exists(source_dir + "/utilities/nep_utilities.cuh") &&
      file_exists(source_dir + "/main_nep/nep_specialized.cu")) {
      return source_dir;
    }
    warning_compile(
      "GPUMD_SRC does not point to a valid GPUMD src directory.");
    return std::string();
  }

  char path[PATH_MAX];
  const ssize_t size = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (size <= 0) {
    warning_compile("Cannot determine the location of the nep executable.");
    return std::string();
  }
  path[size] = '\0';

  const std::string executable(path);
  const size_t pos = executable.find_last_of('/');
  if (pos == std::string::npos) {
    warning_compile("Cannot determine the GPUMD source directory.");
    return std::string();
  }

  const std::string source_dir = executable.substr(0, pos);
  if (
    !file_exists(source_dir + "/utilities/nep_utilities.cuh") ||
    !file_exists(source_dir + "/main_nep/nep_specialized.cu")) {
    warning_compile(
      "Cannot find runtime-specialization sources next to the nep executable. "
      "Run nep from the compiled src tree or set GPUMD_SRC.");
    return std::string();
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

bool all_equal(const std::vector<float>& values)
{
  if (values.empty()) return true;
  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] != values[0]) return false;
  }
  return true;
}

bool validate_config(const NEP_Compile_Config& c)
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
  if (c.mode == NEP_Compile_Mode::TNEP && c.train_mode != 1 && c.train_mode != 2) {
    warning_compile("TNEP specialization requires train_mode 1 or 2.");
    return false;
  }
  return true;
}

std::string make_config_text(const NEP_Compile_Config& c)
{
  const int num_abc = (c.L_max + 1) * (c.L_max + 1) - 1;
  const int dim_radial = c.n_max_radial + 1;
  const bool common_radial = all_equal(c.rc_radial);
  const bool common_angular = all_equal(c.rc_angular);

  std::vector<float> c6 = c.c6_ref_sqrt;
  if (c6.empty()) c6.assign(c.num_types, 0.0f);

  std::ostringstream output;
  output << "#pragma once\n";
  output << "#define NEP_MODEL_NEP 0\n";
  output << "#define NEP_MODEL_CHARGE 1\n";
  output << "#define NEP_MODEL_VDW 2\n";
  output << "#define NEP_MODEL_CHARGE_VDW 3\n";
  output << "#define NEP_MODEL_TNEP 4\n";
  output << "#define NEP_MODEL_MODE_JIT " << static_cast<int>(c.mode) << "\n";
  output << "#define TRAIN_MODE_JIT " << c.train_mode << "\n";
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

template <typename T>
bool load_symbol(void* library, const char* name, T& function)
{
  dlerror();
  function = reinterpret_cast<T>(dlsym(library, name));
  const char* error = dlerror();
  if (error != nullptr || function == nullptr) {
    warning_compile(
      std::string("Cannot find symbol ") + name +
      " in the specialized NEP library.");
    function = nullptr;
    return false;
  }
  return true;
}
#endif
} // namespace



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
  c.train_mode = para.train_mode;
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
  if (mode == NEP_Compile_Mode::TNEP && para.train_mode == 2) {
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

NEP_Compile::NEP_Compile(const NEP_Compile_Config& config)
{
#if defined(USE_HIP)
  (void)config;
  warning_compile("nep_compile on is not supported by the HIP build.");
#elif defined(_WIN32)
  (void)config;
  warning_compile("nep_compile on is supported on Linux only.");
#else
  valid_ = compile(config);
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

bool NEP_Compile::compile(const NEP_Compile_Config& config)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  if (!validate_config(config)) {
    return false;
  }

  const std::string source_dir = get_gpumd_source_dir();
  if (source_dir.empty()) {
    return false;
  }
  const std::string specialized_file =
    source_dir + "/main_nep/nep_specialized.cu";
  const std::string config_text = make_config_text(config);

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    warning_compile(cudaGetErrorString(error));
    return false;
  }

  cudaDeviceProp properties;
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    warning_compile(cudaGetErrorString(error));
    return false;
  }

  // Runtime specialization is intentionally non-persistent.
  // Prefer TMPDIR supplied by the user/scheduler; fall back to /tmp.
  const char* tmpdir_env = std::getenv("TMPDIR");
  std::string tmp_base =
    (tmpdir_env != nullptr && tmpdir_env[0] != '\0') ? tmpdir_env : "/tmp";
  while (tmp_base.size() > 1 && tmp_base.back() == '/') {
    tmp_base.pop_back();
  }

  std::string temp_template =
    tmp_base + "/gpumd-nep-train-compile-XXXXXX";
  std::vector<char> temp_buffer(
    temp_template.begin(), temp_template.end());
  temp_buffer.push_back('\0');

  char* temp_dir = mkdtemp(temp_buffer.data());
  if (temp_dir == nullptr) {
    warning_compile(
      "Cannot create the runtime-specialization temporary directory under " +
      tmp_base + ".");
    return false;
  }

  temp_dir_ = temp_dir;
  config_file_ = temp_dir_ + "/nep_special_config.cuh";
  log_file_ = temp_dir_ + "/build.log";
  library_file_ = temp_dir_ + "/nep_specialized.so";

  {
    std::ofstream output(config_file_.c_str());
    if (!output.is_open()) {
      cleanup_files();
      warning_compile("Cannot create nep_special_config.cuh.");
      return false;
    }
    output << config_text;
  }

  const char* compiler_env = std::getenv("CUDACXX");
  const std::string compiler =
    (compiler_env != nullptr && compiler_env[0] != '\0')
      ? compiler_env
      : "nvcc";

  std::ostringstream command;
  command
    << shell_quote(compiler)
    << " -std=c++14 -O3"
    << " -arch=sm_" << properties.major << properties.minor
    << " -shared -Xcompiler=-fPIC"
    << " -DNEP_SPECIALIZED_RUNTIME_BUILD"
    << " -I" << shell_quote(temp_dir_)
    << " -I" << shell_quote(source_dir)
    << " " << shell_quote(specialized_file)
    << " -o " << shell_quote(library_file_)
    << " > " << shell_quote(log_file_) << " 2>&1";

  printf(
    "Compile specialized %s training kernels (sm_%d%d).\n",
    mode_name(config.mode),
    properties.major,
    properties.minor);
  printf(
    "    types=%d, dim=%d, neurons=(%d,%d), n_max=(%d,%d), "
    "basis=(%d,%d), l_max=%d.\n",
    config.num_types,
    config.ann_dim,
    config.num_neurons1,
    config.num_neurons2,
    config.n_max_radial,
    config.n_max_angular,
    config.basis_size_radial,
    config.basis_size_angular,
    config.L_max);
  fflush(stdout);

  const int status = std::system(command.str().c_str());
  if (status != 0) {
    const std::string log = read_text_file(log_file_);
    if (!log.empty()) {
      std::cerr << log;
    }
    cleanup_files();
    warning_compile(
      "nvcc failed while compiling specialized NEP training kernels.");
    return false;
  }

  library_ = dlopen(library_file_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr) {
    const char* error_message = dlerror();
    const std::string message =
      (error_message == nullptr) ? "unknown dlopen error" : error_message;
    cleanup_files();
    warning_compile(
      "Cannot load the specialized NEP training library: " + message);
    return false;
  }

  bool symbols_ok = true;
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_descriptor_radial", descriptor_radial_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_descriptor_angular", descriptor_angular_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_ann_nep", ann_nep_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_ann_temperature", ann_temperature_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_ann_charge", ann_charge_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_ann_vdw", ann_vdw_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_ann_charge_vdw", ann_charge_vdw_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_ann_tnep_pol", ann_tnep_pol_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_force_radial", force_radial_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_force_angular", force_angular_);

  if (!symbols_ok) {
    dlclose(library_);
    library_ = nullptr;
    cleanup_files();
    return false;
  }

  return true;
#else
  (void)config;
  return false;
#endif
}

void NEP_Compile::cleanup_files()
{
#if !defined(USE_HIP) && !defined(_WIN32)
  if (!config_file_.empty()) {
    std::remove(config_file_.c_str());
    config_file_.clear();
  }
  if (!log_file_.empty()) {
    std::remove(log_file_.c_str());
    log_file_.clear();
  }
  if (!library_file_.empty()) {
    std::remove(library_file_.c_str());
    library_file_.clear();
  }
  if (!temp_dir_.empty()) {
    rmdir(temp_dir_.c_str());
    temp_dir_.clear();
  }
#endif
}

void NEP_Compile::check_launch(
  const int error_code,
  const char* kernel_name)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  if (error_code != static_cast<int>(cudaSuccess)) {
    std::cerr << "Specialized kernel launch failed in "
              << kernel_name << ": "
              << cudaGetErrorString(static_cast<cudaError_t>(error_code))
              << std::endl;
    std::exit(1);
  }
#else
  (void)error_code;
  (void)kernel_name;
#endif
}



void NEP_Compile::launch_descriptor_radial(
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
  check_launch(
    descriptor_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors),
    "descriptor_radial_jit");
}

void NEP_Compile::launch_descriptor_angular(
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
  check_launch(
    descriptor_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, descriptors, sum_fxyz),
    "descriptor_angular_jit");
}

void NEP_Compile::launch_ann(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp)
{
  check_launch(
    ann_nep_(
      N, type, descriptors, q_scaler, parameters, pe, Fp),
    "ann_nep_jit");
}

void NEP_Compile::launch_ann_temperature(
  int N,
  const int* type,
  const float* descriptors,
  float* q_scaler,
  const float* temperature,
  const float* parameters,
  float* pe,
  float* Fp)
{
  check_launch(
    ann_temperature_(
      N, type, descriptors, q_scaler, temperature,
      parameters, pe, Fp),
    "ann_temperature_jit");
}

void NEP_Compile::launch_ann_charge(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* charge,
  float* charge_derivative)
{
  check_launch(
    ann_charge_(
      N, type, descriptors, q_scaler, parameters,
      pe, Fp, charge, charge_derivative),
    "ann_charge_jit");
}

void NEP_Compile::launch_ann_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* C6,
  float* C6_derivative)
{
  check_launch(
    ann_vdw_(
      N, type, descriptors, q_scaler, parameters,
      pe, Fp, C6, C6_derivative),
    "ann_vdw_jit");
}

void NEP_Compile::launch_ann_charge_vdw(
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
  float* C6_derivative)
{
  check_launch(
    ann_charge_vdw_(
      N, type, descriptors, q_scaler, parameters,
      pe, Fp, charge, charge_derivative,
      C6, C6_derivative),
    "ann_charge_vdw_jit");
}

void NEP_Compile::launch_ann_tnep_pol(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* virial,
  float* Fp)
{
  check_launch(
    ann_tnep_pol_(
      N, type, descriptors, q_scaler, parameters, virial, Fp),
    "ann_tnep_pol_jit");
}

namespace
{
const float* null_float()
{
  return nullptr;
}
}

void NEP_Compile::launch_force_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      null_float(), null_float(), null_float(), null_float(),
      0, fx, fy, fz, virial),
    "force_radial_jit");
}

void NEP_Compile::launch_force_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp, const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      null_float(), null_float(), null_float(), null_float(),
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_angular_jit");
}

void NEP_Compile::launch_force_charge_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* charge_derivative, const float* D_real,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      charge_derivative, D_real, null_float(), null_float(),
      0, fx, fy, fz, virial),
    "force_charge_radial_jit");
}

void NEP_Compile::launch_force_charge_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* charge_derivative, const float* D_real,
  const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      charge_derivative, D_real, null_float(), null_float(),
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_charge_angular_jit");
}

void NEP_Compile::launch_force_vdw_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* C6_derivative, const float* D_C6,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      null_float(), null_float(), C6_derivative, D_C6,
      0, fx, fy, fz, virial),
    "force_vdw_radial_jit");
}

void NEP_Compile::launch_force_vdw_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* C6_derivative, const float* D_C6,
  const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      null_float(), null_float(), C6_derivative, D_C6,
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_vdw_angular_jit");
}

void NEP_Compile::launch_force_charge_vdw_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* charge_derivative, const float* D_real,
  const float* C6_derivative, const float* D_C6,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      charge_derivative, D_real, C6_derivative, D_C6,
      0, fx, fy, fz, virial),
    "force_charge_vdw_radial_jit");
}

void NEP_Compile::launch_force_charge_vdw_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* charge_derivative, const float* D_real,
  const float* C6_derivative, const float* D_C6,
  const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      charge_derivative, D_real, C6_derivative, D_C6,
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_charge_vdw_angular_jit");
}

void NEP_Compile::launch_force_tnep_radial(
  bool is_dipole,
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      null_float(), null_float(), null_float(), null_float(),
      is_dipole ? 1 : 0, fx, fy, fz, virial),
    "force_tnep_radial_jit");
}

void NEP_Compile::launch_force_tnep_angular(
  bool is_dipole,
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp, const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_launch(
    force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      null_float(), null_float(), null_float(), null_float(),
      sum_fxyz, is_dipole ? 1 : 0, fx, fy, fz, virial),
    "force_tnep_angular_jit");
}
