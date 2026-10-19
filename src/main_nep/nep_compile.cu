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

#include "nep_compile.cuh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
  if (!validate_nep_compile_config(config)) {
    return false;
  }

  const std::string source_dir = get_gpumd_source_dir();
  if (source_dir.empty()) {
    return false;
  }
  const std::string specialized_file =
    source_dir + "/main_nep/nep_specialized.cu";
  const std::string config_text = make_nep_compile_config_text(config);

  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess || device_count < 1) {
    warning_compile(
      error == cudaSuccess ? "No CUDA device is visible." : cudaGetErrorString(error));
    return false;
  }

  std::vector<int> architectures;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) {
      warning_compile(cudaGetErrorString(error));
      return false;
    }

    const int architecture = properties.major * 10 + properties.minor;
    bool found = false;
    for (size_t i = 0; i < architectures.size(); ++i) {
      if (architectures[i] == architecture) {
        found = true;
        break;
      }
    }
    if (!found) {
      architectures.push_back(architecture);
    }
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
    << " -std=c++14 -O3";
  for (size_t i = 0; i < architectures.size(); ++i) {
    command
      << " -gencode=arch=compute_" << architectures[i]
      << ",code=sm_" << architectures[i];
  }
  command
    << " -shared -Xcompiler=-fPIC"
    << " -DNEP_SPECIALIZED_RUNTIME_BUILD"
    << " -I" << shell_quote(temp_dir_)
    << " -I" << shell_quote(source_dir)
    << " " << shell_quote(specialized_file)
    << " -o " << shell_quote(library_file_)
    << " > " << shell_quote(log_file_) << " 2>&1";

  printf("Compile specialized %s training kernels (", mode_name(config.mode));
  for (size_t i = 0; i < architectures.size(); ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("sm_%d", architectures[i]);
  }
  printf(").\n");
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

  typedef int (*InterfaceVersionFunction)();
  InterfaceVersionFunction interface_version = nullptr;
  if (!load_symbol(
        library_,
        "nep_train_specialized_interface_version",
        interface_version) ||
      interface_version() != NEP_SPECIALIZED_INTERFACE_VERSION) {
    warning_compile("Incompatible NEP specialization interface version.");
    dlclose(library_);
    library_ = nullptr;
    cleanup_files();
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
    library_, "nep_train_launch_bec_radial", bec_radial_);
  symbols_ok &= load_symbol(
    library_, "nep_train_launch_bec_angular", bec_angular_);
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
#ifdef STRONG_DEBUG
  const cudaError_t sync_error = cudaDeviceSynchronize();
  if (sync_error != cudaSuccess) {
    std::cerr << "Specialized kernel execution failed in "
              << kernel_name << ": "
              << cudaGetErrorString(sync_error)
              << std::endl;
    std::exit(1);
  }
#endif
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

void NEP_Compile::launch_bec_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* charge_derivative, float* bec)
{
  check_launch(
    bec_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, charge_derivative, bec),
    "bec_radial_jit");
}
void NEP_Compile::launch_bec_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* charge_derivative,
  const float* sum_fxyz, float* bec)
{
  check_launch(
    bec_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, charge_derivative, sum_fxyz, bec),
    "bec_angular_jit");
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
      nullptr, nullptr, nullptr, nullptr,
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
      nullptr, nullptr, nullptr, nullptr,
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
      charge_derivative, D_real, nullptr, nullptr,
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
      charge_derivative, D_real, nullptr, nullptr,
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
      nullptr, nullptr, C6_derivative, D_C6,
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
      nullptr, nullptr, C6_derivative, D_C6,
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
      nullptr, nullptr, nullptr, nullptr,
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
      nullptr, nullptr, nullptr, nullptr,
      sum_fxyz, is_dipole ? 1 : 0, fx, fy, fz, virial),
    "force_tnep_angular_jit");
}
