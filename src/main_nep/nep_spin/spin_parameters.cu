/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#include "spin_parameters.cuh"
#include "spin_io.cuh"
#include "main_nep/common/nep_model.cuh"
#include "utilities/common.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
Spin_Parameters::Spin_Parameters(const ParameterInput& input) {
  if (input.model_type != 0 || input.charge_mode || input.vdw || input.charge_vdw) {
    PRINT_INPUT_ERROR("Spin NEP only supports a NEP4 potential model without charge/vdW.");
  }
  read(input);
  if (!is_spin_cutoff_set) {
    spin_cutoff = rc_radial_max;
    spin_cutoff_by_type.assign(num_types, spin_cutoff);
  }
  validate_spin_parameters();
  calculate_nep_model_parameters(*this, 0, 0, 1, 1);
  dim_struct = dim;
  calculate_spin_dimensions();
  calculate_nep_model_parameters(*this, dim_spin, 0, 1, 1);
  number_of_variables_descriptor_spin = num_types * num_types * spin_compress * (spin_basis_size[0] + 1);
  number_of_variables_spin_projection = 4 * spin_compress * spin_compress;
  number_of_variables_descriptor += number_of_variables_descriptor_spin + number_of_variables_spin_projection;
  number_of_variables = number_of_variables_ann + number_of_variables_descriptor;
  model_name = "nep_spin";
  set_regularization_defaults();
  initialize_spin_checkpoint(*this);
  // Check the effective masks after any omitted input fields were restored.
  for (int type = 0; type < num_types; ++type) {
    if (spin_dof_type_active[type] && !spin_env_type_active[type]) {
      PRINT_INPUT_ERROR("spin_dof_type must be a subset of spin_env_type.");
    }
  }
  report_inputs(input, "Spin3 potential", nullptr);
  printf("Spin3 O%d/C%d: %d structural + %d magnetic descriptors, %d parameters.\n",
    spin_order, spin_compress, dim_struct, dim_spin, number_of_variables);
  printf("Finished reading nep.in.\n");
}
void Spin_Parameters::report_model_inputs() const
{
  NEP_Energy_Parameters::report_model_inputs();
  printf("    spin_version = 1.\n");
  printf("    spin_order = %d, spin_compress = %d, spin_soc = %d.\n", spin_order, spin_compress, spin_soc);
  printf("    spin_basis_size = %d, spin_l_max = %d.\n", spin_basis_size[0], spin_l_max[0]);
  if (split_mforce_loss) {
    printf("    lambda_m_parallel = %g, lambda_m_perp = %g (independent RMSE terms).\n",
      lambda_m_parallel, lambda_m_perp);
  } else {
    printf("    lambda_m = %g (Cartesian RMSE).\n", lambda_m);
  }
  printf("    lambda_tau = %g, lambda_spin_response = %g.\n", lambda_tau, lambda_spin_response);
  printf("    sigma0 = %g.\n", sigma0);
  printf("    spin_cutoff:");
  for (int t = 0; t < num_types; ++t) printf(" %s=%g", elements[t].c_str(), spin_cutoff_by_type[t]);
  printf("\n    spin_dof_type:");
  for (int t = 0; t < num_types; ++t) if (spin_dof_type_active[t]) printf(" %s", elements[t].c_str());
  printf("\n    spin_env_type:");
  for (int t = 0; t < num_types; ++t) if (spin_env_type_active[t]) printf(" %s", elements[t].c_str());
  printf("\n    Spin q_scaler = %s; restart = %s.\n",
    (prediction || spin_restart || import_q_scaler) ? "loaded from nep.txt" : "fixed reference initialization",
    spin_restart ? "yes" : "no");
}

bool Spin_Parameters::parse_keyword(const char** param, int num_param)
{
  if (strcmp(param[0], "lambda_m") == 0 ||
      strcmp(param[0], "lambda_mforce") == 0) {
    parse_lambda_m(param, num_param);
  } else if (strcmp(param[0], "lambda_m_parallel") == 0) {
    parse_lambda_m_component(param, num_param, lambda_m_parallel, is_lambda_m_parallel_set);
  } else if (strcmp(param[0], "lambda_m_perp") == 0) {
    parse_lambda_m_component(param, num_param, lambda_m_perp, is_lambda_m_perp_set);
  } else if (strcmp(param[0], "lambda_tau") == 0) {
    parse_lambda_tau(param, num_param);
  } else if (strcmp(param[0], "lambda_spin_response") == 0) {
    parse_lambda_spin_response(param, num_param);
  } else if (strcmp(param[0], "spin_compress") == 0) {
    parse_spin_compress(param, num_param);
  } else if (strcmp(param[0], "spin_order") == 0) {
    parse_spin_order(param, num_param);
  } else if (strcmp(param[0], "spin_soc") == 0) {
    parse_spin_soc(param, num_param);

  } else if (strcmp(param[0], "spin_basis_size") == 0) {
    parse_spin_basis_size(param, num_param);
  } else if (strcmp(param[0], "spin_l_max") == 0) {
    parse_spin_l_max(param, num_param);
  } else if (strcmp(param[0], "spin_cutoff") == 0) {
    parse_spin_cutoff(param, num_param);
  } else if (strcmp(param[0], "spin_dof_type") == 0) {
    parse_spin_active_types(
      param,
      num_param,
      "spin_dof_type",
      spin_dof_type_names,
      is_spin_dof_type_set);
  } else if (strcmp(param[0], "spin_env_type") == 0) {
    parse_spin_active_types(
      param,
      num_param,
      "spin_env_type",
      spin_env_type_names,
      is_spin_env_type_set);
  } else {
    return NEP_Energy_Parameters::parse_keyword(param, num_param);
  }
  return true;
}

void Spin_Parameters::validate_spin_parameters()
{
  split_mforce_loss = is_lambda_m_parallel_set || is_lambda_m_perp_set;
  if (split_mforce_loss && is_lambda_m_set) {
    PRINT_INPUT_ERROR("Do not combine lambda_m/lambda_mforce with lambda_m_parallel or lambda_m_perp.\n");
  }
  if (split_mforce_loss && !(is_lambda_m_parallel_set && is_lambda_m_perp_set)) {
    PRINT_INPUT_ERROR("Specify both lambda_m_parallel and lambda_m_perp; use 0 to disable a component.\n");
  }
  if (version != 4) {
    PRINT_INPUT_ERROR("Spin NEP only supports a NEP4 potential model.\n");
  }
  if (has_multiple_cutoffs) {
    PRINT_INPUT_ERROR(
      "Spin3 does not support charge/vdW or type-dependent structural cutoffs.\n");
  }
  if (spin_cutoff > rc_radial_max) {
    PRINT_INPUT_ERROR("spin_cutoff must not exceed the NEP radial cutoff.\n");
  }
  if (num_hidden_layers == 2) {
    PRINT_INPUT_ERROR("Spin NEP only supports one hidden layer.\n");
  }
  if (fine_tune) {
    PRINT_INPUT_ERROR(
      "Spin NEP fine_tune is not supported until the "
      "species-transfer checkpoint reader is implemented.\n");
  }
  if (
    spin_compress < 1 || spin_compress > 9 ||
    spin_basis_size[0] != 8 || spin_basis_size[1] != 0 ||
    spin_l_max[0] < 0 || spin_l_max[0] > 2 ||
    spin_l_max[1] != 0 || spin_l_max[2] != 0 ||
    spin_order < 1 || spin_order > 3 ||
    (spin_soc != 0 && spin_soc != 1) ||
    spin_cutoff <= 0.0f) {
    PRINT_INPUT_ERROR("Unsupported unified Spin NEP O/C shape in nep.in.\n");
  }
#ifdef USE_CJ
  PRINT_INPUT_ERROR("Spin3 requires pair-dependent descriptor coefficients (no USE_CJ).\n");
#endif

  auto resolve_active_types = [&](const std::vector<std::string>& names,
                                  bool is_set,
                                  std::vector<int>& active) {
    active.assign(num_types, is_set ? 0 : 1);
    if (!is_set) {
      return;
    }
    for (const auto& name : names) {
      auto found = std::find(elements.begin(), elements.end(), name);
      if (found == elements.end()) {
        PRINT_INPUT_ERROR("spin_dof_type or spin_env_type contains an unknown atom type.\n");
      }
      active[found - elements.begin()] = 1;
    }
  };
  resolve_active_types(
    spin_dof_type_names,
    is_spin_dof_type_set,
    spin_dof_type_active);
  if (is_spin_env_type_set) {
    resolve_active_types(
      spin_env_type_names,
      true,
      spin_env_type_active);
  } else {
    spin_env_type_active = spin_dof_type_active;
  }
}

void Spin_Parameters::calculate_spin_dimensions()
{
  dim_spin = NEP_Spin_Model_File::spin_dimension(
    spin_compress, spin_l_max[0], spin_order, spin_soc);
  if (dim_spin > 96) {
    PRINT_INPUT_ERROR("Number of spin descriptors should not exceed 96.\n");
  }
  dim = dim_struct + dim_spin;
  if (dim > MAX_DIM_SPIN) {
    PRINT_INPUT_ERROR(
      "Combined structural and spin descriptor dimension exceeds GPUMD MAX_DIM_SPIN.\n");
  }
}

void Spin_Parameters::parse_lambda_m(const char** param, int num_param)
{
  if (is_lambda_m_set) {
    PRINT_INPUT_ERROR("Duplicate lambda_m/lambda_mforce keyword.\n");
  }
  is_lambda_m_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_m should have 1 parameter.\n");
  }
  double value = 0.0;
  if (!is_valid_real(param[1], &value) || !std::isfinite(value) || value < 0.0 ||
      value > std::numeric_limits<float>::max()) {
    PRINT_INPUT_ERROR("Magnetic-force loss weight should be a non-negative number.\n");
  }
  lambda_m = value;
}

void Spin_Parameters::parse_lambda_m_component(
  const char** param, int num_param, float& weight, bool& is_set)
{
  if (is_set) {
    PRINT_INPUT_ERROR("Duplicate magnetic-force component lambda keyword.\n");
  }
  is_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("Each magnetic-force component lambda should have 1 parameter.\n");
  }
  double value = 0.0;
  if (!is_valid_real(param[1], &value) || !std::isfinite(value) || value < 0.0 ||
      value > std::numeric_limits<float>::max()) {
    PRINT_INPUT_ERROR("Magnetic-force component lambda should be finite and non-negative.\n");
  }
  weight = value;
}

void Spin_Parameters::parse_lambda_tau(const char** param, int num_param)
{
  if (is_lambda_tau_set) {
    PRINT_INPUT_ERROR("Duplicate lambda_tau keyword.\n");
  }
  is_lambda_tau_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_tau should have 1 parameter.\n");
  }
  double value = 0.0;
  if (!is_valid_real(param[1], &value) || !std::isfinite(value) || value < 0.0 ||
      value > std::numeric_limits<float>::max()) {
    PRINT_INPUT_ERROR("Spin-torque loss weight should be finite and non-negative.\n");
  }
  lambda_tau = value;
}

void Spin_Parameters::parse_lambda_spin_response(const char** param, int num_param)
{
  if (is_lambda_spin_response_set) {
    PRINT_INPUT_ERROR("Duplicate lambda_spin_response keyword.\n");
  }
  is_lambda_spin_response_set = true;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_spin_response should have 1 parameter.\n");
  }
  double value = 0.0;
  if (!is_valid_real(param[1], &value) || !std::isfinite(value) || value < 0.0 ||
      value > std::numeric_limits<float>::max()) {
    PRINT_INPUT_ERROR("lambda_spin_response should be finite and non-negative.\n");
  }
  lambda_spin_response = value;
}


void Spin_Parameters::parse_spin_compress(const char** param, int num_param)
{
  if (is_spin_compress_set) {
    PRINT_INPUT_ERROR("Duplicate spin_compress keyword.\n");
  }
  is_spin_compress_set = true;
  if (num_param != 2 || !is_valid_int(param[1], &spin_compress) ||
      spin_compress < 1 || spin_compress > 9) {
    PRINT_INPUT_ERROR("spin_compress should be an integer from 1 to 9.\n");
  }
}

void Spin_Parameters::parse_spin_order(const char** param, int num_param)
{
  if (is_spin_order_set) {
    PRINT_INPUT_ERROR("Duplicate spin_order keyword.\n");
  }
  is_spin_order_set = true;
  if (num_param != 2 || !is_valid_int(param[1], &spin_order) ||
      spin_order < 1 || spin_order > 3) {
    PRINT_INPUT_ERROR("spin_order should be an integer from 1 to 3.\n");
  }
}

void Spin_Parameters::parse_spin_soc(const char** param, int num_param)
{
  if (is_spin_soc_set) {
    PRINT_INPUT_ERROR("Duplicate spin_soc keyword.\n");
  }
  is_spin_soc_set = true;
  if (num_param != 2 || !is_valid_int(param[1], &spin_soc) ||
      (spin_soc != 0 && spin_soc != 1)) {
    PRINT_INPUT_ERROR("spin_soc should be 0 or 1.\n");
  }
}


void Spin_Parameters::parse_spin_basis_size(const char** param, int num_param)
{
  if (is_spin_basis_size_set) {
    PRINT_INPUT_ERROR("Duplicate spin_basis_size keyword.\n");
  }
  is_spin_basis_size_set = true;
  if (num_param != 3 ||
      !is_valid_int(param[1], &spin_basis_size[0]) ||
      !is_valid_int(param[2], &spin_basis_size[1]) ||
      spin_basis_size[0] < 0 || spin_basis_size[1] < 0) {
    PRINT_INPUT_ERROR("spin_basis_size should have two non-negative integers.\n");
  }
}

void Spin_Parameters::parse_spin_l_max(const char** param, int num_param)
{
  if (is_spin_l_max_set) {
    PRINT_INPUT_ERROR("Duplicate spin_l_max keyword.\n");
  }
  is_spin_l_max_set = true;
  if (num_param != 4 ||
      !is_valid_int(param[1], &spin_l_max[0]) ||
      !is_valid_int(param[2], &spin_l_max[1]) ||
      !is_valid_int(param[3], &spin_l_max[2])) {
    PRINT_INPUT_ERROR("spin_l_max should have three integers.\n");
  }
}

void Spin_Parameters::parse_spin_cutoff(const char** param, int num_param)
{
  if (is_spin_cutoff_set) {
    PRINT_INPUT_ERROR("Duplicate spin_cutoff keyword.\n");
  }
  is_spin_cutoff_set = true;
  if (num_types <= 0 || (num_param != 2 && num_param != num_types + 1)) {
    PRINT_INPUT_ERROR(
      "spin_cutoff should have one positive number or one per atom type.\n");
  }
  spin_cutoff_by_type.resize(num_types);
  for (int type = 0; type < num_types; ++type) {
    double cutoff = 0.0;
    const int value_index = num_param == 2 ? 1 : type + 1;
    if (!is_valid_real(param[value_index], &cutoff) || !std::isfinite(cutoff) || cutoff <= 0.0 ||
        cutoff > std::numeric_limits<float>::max()) {
      PRINT_INPUT_ERROR("spin_cutoff values must be finite positive numbers within float range.\n");
    }
    spin_cutoff_by_type[type] = static_cast<float>(cutoff);
  }
  spin_cutoff = *std::max_element(
    spin_cutoff_by_type.begin(), spin_cutoff_by_type.end());
}

void Spin_Parameters::parse_spin_active_types(
  const char** param,
  int num_param,
  const char* keyword,
  std::vector<std::string>& names,
  bool& is_set)
{
  if (is_set) {
    PRINT_INPUT_ERROR("Duplicate spin_dof_type or spin_env_type keyword.\n");
  }
  is_set = true;
  if (num_param < 2) {
    PRINT_INPUT_ERROR("spin_dof_type and spin_env_type must enable at least one type.\n");
  }
  names.clear();
  for (int n = 1; n < num_param; ++n) {
    if (std::find(names.begin(), names.end(), param[n]) != names.end()) {
      PRINT_INPUT_ERROR("Duplicate atom type in spin_dof_type or spin_env_type.\n");
    }
    names.emplace_back(param[n]);
  }
  (void)keyword;
}

void Spin_Parameters::read_zbl_in()
{
  // Prediction ZBL metadata is read from the model, not from zbl.in.
  if (!prediction) {
    FILE* input = fopen("zbl.in", "r");
    if (input != nullptr) {
      fclose(input);
      PRINT_INPUT_ERROR("NEP-spin does not support flexible ZBL: remove zbl.in to use universal ZBL.");
    }
  }
}
