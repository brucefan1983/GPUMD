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
#include <string>
#include <vector>

// Model selectors are used only while reading input and constructing the model.
class ParameterInput
{
public:
  ParameterInput();
  std::string model; // non-empty when the model keyword is used
  int model_type = 0;
  int spin_version = 0; // zero means the Spin model was not selected
  int charge_mode = 0;
  int vdw = 0;
  int charge_vdw = 0;
  int flip_charge = 0;
  bool is_model_type_set = false;
  bool is_charge_mode_set = false;
  bool is_vdw_set = false;
  bool is_charge_vdw_set = false;
  std::vector<std::vector<std::string>> keywords;
  void report_disabled_models() const;

private:
  bool is_flip_charge_set = false;
  bool is_legacy_flip_charge_set = false;
  void parse_model(const char** param, int num_param);
  void parse_flip_charge(const char** param, int num_param);
  void report_deprecated_selectors() const;
  void parse_model_type(const char** param, int num_param);
  void parse_charge_mode(const char** param, int num_param);
  void parse_vdw(const char** param, int num_param);
  void parse_charge_vdw(const char** param, int num_param);
};
