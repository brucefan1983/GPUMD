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

void warning_nep_compile(const std::string& message)
{
  std::cerr << "Warning: NEP training specialization disabled: "
            << message << std::endl;
}

namespace
{
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

std::string get_gpumd_source_dir(const std::string& source_file)
{
  const char* source_env = std::getenv("GPUMD_SRC");
  if (source_env != nullptr && source_env[0] != '\0') {
    const std::string source_dir(source_env);
    if (
      file_exists(source_dir + "/utilities/nep_utilities.cuh") &&
      file_exists(source_dir + "/" + source_file)) {
      return source_dir;
    }
    warning_nep_compile(
      "GPUMD_SRC does not point to a valid GPUMD src directory.");
    return std::string();
  }

  char path[PATH_MAX];
  const ssize_t size = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (size <= 0) {
    warning_nep_compile("Cannot determine the location of the nep executable.");
    return std::string();
  }
  path[size] = '\0';

  const std::string executable(path);
  const size_t pos = executable.find_last_of('/');
  if (pos == std::string::npos) {
    warning_nep_compile("Cannot determine the GPUMD source directory.");
    return std::string();
  }

  const std::string source_dir = executable.substr(0, pos);
  if (
    !file_exists(source_dir + "/utilities/nep_utilities.cuh") ||
    !file_exists(source_dir + "/" + source_file)) {
    warning_nep_compile(
      "Cannot find runtime-specialization sources next to the nep executable. "
      "Run nep from the compiled src tree or set GPUMD_SRC.");
    return std::string();
  }
  return source_dir;
}

#endif
} // namespace

NEP_Compile::NEP_Compile(const NEP_Compile_Config& config)
{
#if defined(USE_HIP)
  (void)config;
  warning_nep_compile("nep_compile on is not supported by the HIP build.");
#elif defined(_WIN32)
  (void)config;
  warning_nep_compile("nep_compile on is supported on Linux only.");
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

  const std::string source_dir = get_gpumd_source_dir(config.source_file);
  if (source_dir.empty()) {
    return false;
  }
  const std::string specialized_file =
    source_dir + "/" + config.source_file;
  const std::string config_text = make_nep_compile_config_text(config);

  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess || device_count < 1) {
    warning_nep_compile(
      error == cudaSuccess ? "No CUDA device is visible." : cudaGetErrorString(error));
    return false;
  }

  std::vector<int> architectures;
  for (int device = 0; device < device_count; ++device) {
    cudaDeviceProp properties;
    error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) {
      warning_nep_compile(cudaGetErrorString(error));
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
    warning_nep_compile(
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
      warning_nep_compile("Cannot create nep_special_config.cuh.");
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

  printf("Compile specialized %s training kernels (", config.model_name.c_str());
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
    warning_nep_compile(
      "nvcc failed while compiling specialized NEP training kernels.");
    return false;
  }

  library_ = dlopen(library_file_.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr) {
    const char* error_message = dlerror();
    const std::string message =
      (error_message == nullptr) ? "unknown dlopen error" : error_message;
    cleanup_files();
    warning_nep_compile(
      "Cannot load the specialized NEP training library: " + message);
    return false;
  }

  typedef int (*InterfaceVersionFunction)();
  InterfaceVersionFunction interface_version = nullptr;
  if (!load_symbol(
        "nep_train_specialized_interface_version",
        interface_version) ||
      interface_version() != NEP_SPECIALIZED_INTERFACE_VERSION) {
    warning_nep_compile("Incompatible NEP specialization interface version.");
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

void check_nep_launch(
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

void* NEP_Compile::find_symbol(const char* name)
{
#if !defined(USE_HIP) && !defined(_WIN32)
  if (library_ == nullptr) {
    return nullptr;
  }
  dlerror();
  void* function = dlsym(library_, name);
  const char* error = dlerror();
  if (error != nullptr || function == nullptr) {
    warning_nep_compile(
      std::string("Cannot find symbol ") + name +
      " in the specialized NEP library.");
    return nullptr;
  }
  return function;
#else
  (void)name;
  return nullptr;
#endif
}
