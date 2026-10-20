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

#include "parameter_input.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <iostream>

ParameterInput::ParameterInput()
{
  print_line_1();
  printf("Started reading nep.in.\n");
  print_line_2();
  std::ifstream input("nep.in");
  if (!input.is_open()) {
    std::cout << "Failed to open nep.in." << std::endl;
    exit(1);
  }
  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens_without_comments(input);
    if (tokens.empty()) {
      continue;
    }
    std::vector<const char*> param(tokens.size());
    for (int n = 0; n < tokens.size(); ++n) {
      param[n] = tokens[n].c_str();
    }
    if (tokens[0] == "model_type") {
      parse_model_type(param.data(), param.size());
    } else if (tokens[0] == "charge_mode") {
      parse_charge_mode(param.data(), param.size());
    } else if (tokens[0] == "vdw") {
      parse_vdw(param.data(), param.size());
    } else if (tokens[0] == "charge_vdw") {
      parse_charge_vdw(param.data(), param.size());
    } else {
      keywords.push_back(tokens);
    }
  }
  if ((charge_mode > 0) + vdw + charge_vdw > 1) {
    PRINT_INPUT_ERROR("charge_mode, vdw, and charge_vdw cannot be enabled simultaneously.");
  }
  if ((charge_mode || charge_vdw || vdw) && model_type != 0) {
    PRINT_INPUT_ERROR("Charge/vdW is only supported for potential model.");
  }
}

void ParameterInput::report_disabled_models() const
{
  if (is_vdw_set && !vdw) {
    printf("    (input)   do not add environment-dependent vdW.\n");
  }
  if (is_charge_vdw_set && !charge_vdw) {
    printf("    (input)   do not use the combined charge-vdW model.\n");
  }
}

void ParameterInput::parse_model_type(const char** param, int num_param)
{
  is_model_type_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("model_type should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &model_type)) {
    PRINT_INPUT_ERROR("model_type should be an integer.\n");
  }
  if (model_type != 0 && model_type != 1 && model_type != 2 && model_type != 3) {
    PRINT_INPUT_ERROR("model_type should = 0 or 1 or 2 or 3.");
  }
}

void ParameterInput::parse_charge_mode(const char** param, int num_param)
{
  is_charge_mode_set = true;

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("charge_mode should have one or two parameters.\n");
  }
  if (!is_valid_int(param[1], &charge_mode)) {
    PRINT_INPUT_ERROR("charge mode should be an integer.\n");
  }
  if (charge_mode < 0 || charge_mode > 2) {
    PRINT_INPUT_ERROR("charge mode should be 0 or 1 or 2.");
  }
  if (num_param == 3) {
    if (!is_valid_int(param[2], &flip_charge)) {
      PRINT_INPUT_ERROR("flip_charge should be an integer.\n");
    }
    if (flip_charge < 0 || flip_charge > 1) {
      PRINT_INPUT_ERROR("flip_charge should be 0 or 1.");
    }
  }
}

void ParameterInput::parse_vdw(const char** param, int num_param)
{
  is_vdw_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("vdw should have one parameter.\n");
  }
  if (!is_valid_int(param[1], &vdw)) {
    PRINT_INPUT_ERROR("vdw should be an integer.\n");
  }
  if (vdw < 0 || vdw > 1) {
    PRINT_INPUT_ERROR("vdw should be 0 or 1.");
  }
}

void ParameterInput::parse_charge_vdw(const char** param, int num_param)
{
  is_charge_vdw_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("charge_vdw should have one parameter.\n");
  }
  if (!is_valid_int(param[1], &charge_vdw)) {
    PRINT_INPUT_ERROR("charge_vdw should be an integer.\n");
  }
  if (charge_vdw < 0 || charge_vdw > 1) {
    PRINT_INPUT_ERROR("charge_vdw should be 0 or 1.");
  }
}
