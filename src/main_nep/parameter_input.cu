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
  bool has_spin_selector = false;
  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens_without_comments(input);
    if (tokens.empty()) {
      continue;
    }
    std::vector<const char*> param(tokens.size());
    for (int n = 0; n < tokens.size(); ++n) {
      param[n] = tokens[n].c_str();
    }
    if (tokens[0] == "model") {
      parse_model(param.data(), param.size());
    } else if (tokens[0] == "flip_charge") {
      parse_flip_charge(param.data(), param.size());
    } else if (tokens[0] == "spin_version") {
      if (has_spin_selector) {
        PRINT_INPUT_ERROR("Specify spin_version only once.");
      }
      has_spin_selector = true;
      if (tokens.size() != 2 || tokens[1] != "1") {
        PRINT_INPUT_ERROR("Unsupported spin_version (expected 1).");
      }
      spin_version = 1;
    } else if (tokens[0] == "model_type") {
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
  if (!model.empty()) {
    if (is_model_type_set || is_charge_mode_set || is_vdw_set || is_charge_vdw_set) {
      PRINT_INPUT_ERROR("model cannot be combined with model_type, charge_mode, vdw, or charge_vdw.");
    }
    if (has_spin_selector && model != "nep_spin") {
      PRINT_INPUT_ERROR("spin_version requires model nep_spin.");
    }
  }
  if ((charge_mode > 0) + vdw + charge_vdw > 1) {
    PRINT_INPUT_ERROR("charge_mode, vdw, and charge_vdw cannot be enabled simultaneously.");
  }
  if ((charge_mode || charge_vdw || vdw) && model_type != 0) {
    PRINT_INPUT_ERROR("Charge/vdW is only supported for potential model.");
  }
  if (is_flip_charge_set && !charge_mode && !charge_vdw) {
    PRINT_INPUT_ERROR("flip_charge is only supported for charge models.");
  }
  report_deprecated_selectors();
}

void ParameterInput::parse_model(const char** param, int num_param)
{
  if (!model.empty()) {
    PRINT_INPUT_ERROR("Specify model only once.");
  }
  if (num_param != 2) {
    PRINT_INPUT_ERROR("model should have one parameter.");
  }
  model = param[1];
  if (model == "nep") {
    model_type = 0;
  } else if (model == "nep_dipole") {
    model_type = 1;
  } else if (model == "nep_polarizability") {
    model_type = 2;
  } else if (model == "nep_temperature") {
    model_type = 3;
  } else if (model == "nep_spin") {
    spin_version = 1;
  } else if (model == "qnep1") {
    charge_mode = 1;
  } else if (model == "qnep2") {
    charge_mode = 2;
  } else if (model == "nep_vdw") {
    vdw = 1;
  } else if (model == "qnep3_vdw") {
    charge_vdw = 1;
  } else {
    PRINT_INPUT_ERROR(
      "model should be nep, nep_dipole, nep_polarizability, nep_temperature, "
      "nep_spin, qnep1, qnep2, nep_vdw, or qnep3_vdw.");
  }
}

void ParameterInput::parse_flip_charge(const char** param, int num_param)
{
  if (is_flip_charge_set || is_legacy_flip_charge_set) {
    PRINT_INPUT_ERROR(
      "Specify flip_charge only once, either as a keyword or as the second charge_mode parameter.");
  }
  is_flip_charge_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("flip_charge should have one parameter.");
  }
  if (!is_valid_int(param[1], &flip_charge)) {
    PRINT_INPUT_ERROR("flip_charge should be an integer.");
  }
  if (flip_charge < 0 || flip_charge > 1) {
    PRINT_INPUT_ERROR("flip_charge should be 0 or 1.");
  }
}

void ParameterInput::report_deprecated_selectors() const
{
  if (!model.empty() ||
      !(is_model_type_set || is_charge_mode_set || is_vdw_set || is_charge_vdw_set || spin_version)) {
    return;
  }
  const char* model_types[] = {"nep", "nep_dipole", "nep_polarizability", "nep_temperature"};
  const char* name = model_types[model_type];
  if (spin_version) {
    name = "nep_spin";
  } else if (charge_vdw) {
    name = "qnep3_vdw";
  } else if (charge_mode) {
    name = charge_mode == 1 ? "qnep1" : "qnep2";
  } else if (vdw) {
    name = "nep_vdw";
  }
  printf("Warning: legacy model selection in nep.in is deprecated. Use 'model %s'", name);
  if (flip_charge && (charge_mode || charge_vdw) && !spin_version) {
    printf(" and 'flip_charge 1'");
  }
  printf(" instead.\n\n");
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
    if (is_flip_charge_set) {
      PRINT_INPUT_ERROR(
        "Specify flip_charge only once, either as a keyword or as the second charge_mode parameter.");
    }
    is_legacy_flip_charge_set = true;
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
