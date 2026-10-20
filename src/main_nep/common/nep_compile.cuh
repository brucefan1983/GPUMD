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
#include "nep_compile_config.cuh"
#include <string>

void warning_nep_compile(const std::string& message);
void check_nep_launch(int error_code, const char* kernel_name);

class NEP_Compile
{
public:
  explicit NEP_Compile(const NEP_Compile_Config& config);
  ~NEP_Compile();
  NEP_Compile(const NEP_Compile&) = delete;
  NEP_Compile& operator=(const NEP_Compile&) = delete;
  bool is_valid() const { return valid_; }

  template <typename T>
  bool load_symbol(const char* name, T& function)
  {
    function = reinterpret_cast<T>(find_symbol(name));
    return function != nullptr;
  }

private:
  bool valid_ = false;
  void* library_ = nullptr;
  std::string temp_dir_;
  std::string config_file_;
  std::string library_file_;
  std::string log_file_;
  bool compile(const NEP_Compile_Config& config);
  void* find_symbol(const char* name);
  void cleanup_files();
};
