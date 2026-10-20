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

#include "nep_parameters.cuh"
#include "nep_elements.cuh"
#include "nep_io.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>

NEP_Parameters::NEP_Parameters()
{
  is_version_set = false;
  is_type_set = false;
  is_cutoff_set = false;
  is_n_max_set = false;
  is_basis_size_set = false;
  is_l_max_set = false;
  is_neuron_set = false;
  is_lambda_e_set = false;
  is_lambda_f_set = false;
  is_lambda_v_set = false;
  is_lambda_shear_set = false;
  is_type_weight_set = false;
  is_zbl_set = false;
  is_force_delta_set = false;
  is_use_typewise_cutoff_zbl_set = false;

  version = 4;                 // NEP4 is the best
  basis_size_radial = 6;       // large enough in most cases
  basis_size_angular = 6;      // large enough in most cases
  n_max_radial = 6;            // large enough in most cases
  n_max_angular = 6;           // large enough in most cases
  L_max = 4;                   // default value
  has_q_222 = 1;               // default is to include q_222
  has_q_1111 = 0;              // default is not to include q_1111
  has_q_112 = 0;               // default is not to include q_112
  has_q_123 = 0;               // default is not to include q_123
  has_q_233 = 0;               // default is not to include q_233
  has_q_134 = 0;               // default is not to include q_134
  num_neurons1 = 30;           // a relatively small value to achieve high speed
  num_neurons2 = 0;            // no second hidden layer by default
  num_hidden_layers = 1;       // one hidden layer by default
  lambda_e = lambda_f = 1.0f;  // energy and force are more important
  lambda_v = 0.1f;             // virial is less important
  lambda_shear = 1.0f;         // do not weight shear virial more by default
  force_delta = 0.0f;          // no modification of force loss
  use_typewise_cutoff_zbl = false;
  typewise_cutoff_zbl_factor = -1.0f;
  output_descriptor = false;
  nep_compile = true;

  type_weight_cpu.resize(NUM_ELEMENTS);
  rc_radial.resize(NUM_ELEMENTS);
  rc_angular.resize(NUM_ELEMENTS);
  zbl_para.resize(550); // Maximum number of zbl parameters
  for (int n = 0; n < NUM_ELEMENTS; ++n) {
    type_weight_cpu[n] = 1.0f; // uniform weight by default
    rc_radial[n] = 8.0f;
    rc_angular[n] = 4.0f;
  }
  rc_radial_max = 8.0f;
  rc_angular_max = 4.0f;
  enable_zbl = false;   // default is not to include ZBL
  flexible_zbl = false; // default Universal ZBL
}

void NEP_Parameters::read(const ParameterInput& input)
{
  for (const auto& tokens : input.keywords) {
    std::vector<const char*> param(tokens.size());
    for (int n = 0; n < tokens.size(); ++n) {
      param[n] = tokens[n].c_str();
    }
    if (!parse_keyword(param.data(), param.size())) {
      PRINT_INPUT_ERROR((tokens[0] + " is not supported by this model.").c_str());
    }
  }
  if (!is_type_set) {
    PRINT_INPUT_ERROR("type in nep.in has not been set.");
  }
  if (is_zbl_set) {
    read_zbl_in();
  }
}

void NEP_Parameters::finish_reading(
  const ParameterInput& input, const char* model_name, const char* model_description)
{
  set_regularization_defaults();
  check_existing_nep_model(*this);
  initialize_q_scaler();
  report_inputs(input, model_name, model_description);
  print_line_1();
  printf("Finished reading nep.in.\n");
  print_line_2();
}

bool NEP_Parameters::parse_keyword(const char** param, int num_param)
{
  if (strcmp(param[0], "nep_compile") == 0) {
    parse_nep_compile(param, num_param);
  } else if (strcmp(param[0], "version") == 0) {
    parse_version(param, num_param);
  } else if (strcmp(param[0], "type") == 0) {
    parse_type(param, num_param);
  } else if (strcmp(param[0], "type_weight") == 0) {
    parse_type_weight(param, num_param);
  } else if (strcmp(param[0], "zbl") == 0) {
    parse_zbl(param, num_param);
  } else if (strcmp(param[0], "force_delta") == 0) {
    parse_force_delta(param, num_param);
  } else if (strcmp(param[0], "cutoff") == 0) {
    parse_cutoff(param, num_param);
  } else if (strcmp(param[0], "n_max") == 0) {
    parse_n_max(param, num_param);
  } else if (strcmp(param[0], "basis_size") == 0) {
    parse_basis_size(param, num_param);
  } else if (strcmp(param[0], "l_max") == 0) {
    parse_l_max(param, num_param);
  } else if (strcmp(param[0], "neuron") == 0) {
    parse_neuron(param, num_param);
  } else if (strcmp(param[0], "lambda_e") == 0) {
    parse_lambda_e(param, num_param);
  } else if (strcmp(param[0], "lambda_f") == 0) {
    parse_lambda_f(param, num_param);
  } else if (strcmp(param[0], "lambda_v") == 0) {
    parse_lambda_v(param, num_param);
  } else if (strcmp(param[0], "lambda_shear") == 0) {
    parse_lambda_shear(param, num_param);
  } else if (strcmp(param[0], "use_typewise_cutoff_zbl") == 0) {
    parse_use_typewise_cutoff_zbl(param, num_param);
  } else if (strcmp(param[0], "output_descriptor") == 0) {
    parse_output_descriptor(param, num_param);
  } else if (strcmp(param[0], "fine_tune") == 0) {
    parse_fine_tune(param, num_param);
  } else if (strcmp(param[0], "import_q_scaler") == 0) {
    parse_import_q_scaler(param, num_param);
  } else {
    return Parameters::parse_keyword(param, num_param);
  }
  return true;
}

void NEP_Parameters::read_zbl_in()
{
  FILE* fid_zbl = fopen("zbl.in", "r");
  if (fid_zbl == NULL) {
    flexible_zbl = false;
  } else {
    flexible_zbl = true;
    for (int n = 0; n < (num_types * (num_types + 1) / 2) * 10; ++n) {
      int count = fscanf(fid_zbl, "%f", &zbl_para[n]);
      PRINT_SCANF_ERROR(count, 1, "Reading error for zbl.in.");
    }
    fclose(fid_zbl);
  }
}

void NEP_Parameters::initialize_q_scaler()
{
  q_scaler_cpu.resize(dim,  1.0e10f);
  read_nep_q_scaler(*this);

  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  for (int device_id = 0; device_id < deviceCount; device_id++) {
    CHECK(gpuSetDevice(device_id));
    q_scaler_gpu[device_id].resize(dim);
    q_scaler_gpu[device_id].copy_from_host(q_scaler_cpu.data());

    q_scaler_max[device_id].resize(dim);
    q_scaler_min[device_id].resize(dim);
    std::vector<float> q_scaler_max_cpu(dim, -1.0e10f);
    std::vector<float> q_scaler_min_cpu(dim, 1.0e10f);
    q_scaler_max[device_id].copy_from_host(q_scaler_max_cpu.data());
    q_scaler_min[device_id].copy_from_host(q_scaler_min_cpu.data());
  }
}

void NEP_Parameters::parse_nep_compile(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("nep_compile should have 1 parameter: on or off.\n");
  }
  if (strcmp(param[1], "on") == 0 || strcmp(param[1], "1") == 0) {
    nep_compile = true;
  } else if (strcmp(param[1], "off") == 0 || strcmp(param[1], "0") == 0) {
    nep_compile = false;
  } else {
    PRINT_INPUT_ERROR("nep_compile should be on/off (or 1/0).\n");
  }
}

void NEP_Parameters::parse_version(const char** param, int num_param)
{
  is_version_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("version should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &version)) {
    PRINT_INPUT_ERROR("version should be an integer.\n");
  }
  if (version != 4) {
    PRINT_INPUT_ERROR("version should = 4.");
  }
}

void NEP_Parameters::parse_type(const char** param, int num_param)
{
  is_type_set = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("type should have at least 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &num_types)) {
    PRINT_INPUT_ERROR("number of types should be integer.\n");
  }

  if (num_types < 1 || num_types > NUM_ELEMENTS) {
    PRINT_INPUT_ERROR("number of types should >=1 and <= NUM_ELEMENTS.");
  }
  if (num_param != 2 + num_types) {
    PRINT_INPUT_ERROR("number of types and the number of listed elements do not match.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    elements.emplace_back(param[2 + n]);
    bool is_valid_element = false;
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (elements.back() == NEP_ELEMENTS[m]) {
        is_valid_element = true;
        atomic_number = m + 1;
        break;
      }
    }
    atomic_numbers.emplace_back(atomic_number);
    if (!is_valid_element) {
      PRINT_INPUT_ERROR("Some element in nep.in is not in the periodic table.");
    }
  }
}

void NEP_Parameters::parse_type_weight(const char** param, int num_param)
{
  is_type_weight_set = true;

  if (!is_type_set) {
    PRINT_INPUT_ERROR("Please set type before setting type weight.\n");
  }

  if (num_param != 1 + num_types) {
    PRINT_INPUT_ERROR("type_weight should have num_types parameters.\n");
  }

  for (int n = 0; n < num_types; ++n) {
    double weight_tmp = 0.0;
    if (!is_valid_real(param[1 + n], &weight_tmp)) {
      PRINT_INPUT_ERROR("type weight should be a number.\n");
    }
    type_weight_cpu[n] = weight_tmp;
  }
}

void NEP_Parameters::parse_zbl(const char** param, int num_param)
{
  is_zbl_set = true;
  enable_zbl = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("zbl should have 1 parameter.\n");
  }

  double zbl_rc_outer_tmp = 0.0;
  if (!is_valid_real(param[1], &zbl_rc_outer_tmp)) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should be a number.\n");
  }
  zbl_rc_outer = zbl_rc_outer_tmp;
  zbl_rc_inner = zbl_rc_outer * 0.5f;

  if (zbl_rc_outer < 1.0f) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should >= 1.0 A.");
  } else if (zbl_rc_outer > 4.0f) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should <= 4.0 A.");
  }
}

void NEP_Parameters::parse_force_delta(const char** param, int num_param)
{
  is_force_delta_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("force_delta should have 1 parameter.\n");
  }

  double force_delta_tmp = 0.0;
  if (!is_valid_real(param[1], &force_delta_tmp)) {
    PRINT_INPUT_ERROR("force_delta should be a number.\n");
  }
  force_delta = force_delta_tmp;
}

void NEP_Parameters::parse_cutoff(const char** param, int num_param)
{
  is_cutoff_set = true;

  if (!is_type_set) {
    PRINT_INPUT_ERROR("Please set type before setting cutoff.\n");
  }

  if (num_param != 3 && num_param != num_types * 2 + 1) {
    PRINT_INPUT_ERROR("cutoff should have 2 or num_types * 2 parameters.\n");
  }

  if (num_param == 3) {
    double rc_radial_tmp = 0.0;
    if (!is_valid_real(param[1], &rc_radial_tmp)) {
      PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
    }
    for (int n = 0; n < num_types; ++ n) {
      rc_radial[n] = rc_radial_tmp;
    }

    double rc_angular_tmp = 0.0;
    if (!is_valid_real(param[2], &rc_angular_tmp)) {
      PRINT_INPUT_ERROR("angular cutoff should be a number.\n");
    }
    for (int n = 0; n < num_types; ++ n) {
      rc_angular[n] = rc_angular_tmp;
    }

    if (rc_angular_tmp > rc_radial_tmp) {
      PRINT_INPUT_ERROR("angular cutoff should <= radial cutoff.");
    }
    if (rc_angular_tmp < 3.0f) {
      PRINT_INPUT_ERROR("angular cutoff should >= 3.0 A.");
    }
    if (rc_radial_tmp > 100.0f) {
      PRINT_INPUT_ERROR("radial cutoff should <= 100 A.");
    }
  } else {
    has_multiple_cutoffs = true;

    for (int n = 0; n < num_types; ++n) {
      double rc_radial_tmp = 0.0;
      if (!is_valid_real(param[1 + n * 2], &rc_radial_tmp)) {
        PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
      }
      rc_radial[n] = rc_radial_tmp;

      double rc_angular_tmp = 0.0;
      if (!is_valid_real(param[2 + n * 2], &rc_angular_tmp)) {
        PRINT_INPUT_ERROR("angular cutoff should be a number.\n");
      }
      rc_angular[n] = rc_angular_tmp;

      if (rc_angular_tmp > rc_radial_tmp) {
        PRINT_INPUT_ERROR("angular cutoff should <= radial cutoff.");
      }
      if (rc_angular_tmp < 3.0f) {
        PRINT_INPUT_ERROR("angular cutoff should >= 3.0 A.");
      }
      if (rc_radial_tmp > 100.0f) {
        PRINT_INPUT_ERROR("radial cutoff should <= 100 A.");
      }
    }
  }

  rc_radial_max = 0.0f;
  rc_angular_max = 0.0f;
  for (int n = 0; n < num_types; ++ n) {
    if (rc_radial[n] > rc_radial_max) {
      rc_radial_max = rc_radial[n];
    }
    if (rc_angular[n] > rc_angular_max) {
      rc_angular_max = rc_angular[n];
    }
  }
}

void NEP_Parameters::parse_n_max(const char** param, int num_param)
{
  is_n_max_set = true;

  if (num_param != 3) {
    PRINT_INPUT_ERROR("n_max should have 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &n_max_radial)) {
    PRINT_INPUT_ERROR("n_max_radial should be an integer.\n");
  }
  if (!is_valid_int(param[2], &n_max_angular)) {
    PRINT_INPUT_ERROR("n_max_angular should be an integer.\n");
  }
  if (n_max_radial < 0) {
    PRINT_INPUT_ERROR("n_max_radial should >= 0.");
  } else if (n_max_radial > 12) {
    PRINT_INPUT_ERROR("n_max_radial should <= 12.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 8) {
    PRINT_INPUT_ERROR("n_max_angular should <= 8.");
  }
}

void NEP_Parameters::parse_basis_size(const char** param, int num_param)
{
  is_basis_size_set = true;

  if (num_param != 3) {
    PRINT_INPUT_ERROR("basis_size should have 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &basis_size_radial)) {
    PRINT_INPUT_ERROR("basis_size_radial should be an integer.\n");
  }
  if (!is_valid_int(param[2], &basis_size_angular)) {
    PRINT_INPUT_ERROR("basis_size_angular should be an integer.\n");
  }
  if (basis_size_radial < 0) {
    PRINT_INPUT_ERROR("basis_size_radial should >= 0.");
  } else if (basis_size_radial > 16) {
    PRINT_INPUT_ERROR("basis_size_radial should <= 16.");
  }
  if (basis_size_angular < 0) {
    PRINT_INPUT_ERROR("basis_size_angular should >= 0.");
  } else if (basis_size_angular > 12) {
    PRINT_INPUT_ERROR("basis_size_angular should <= 12.");
  }
}

void NEP_Parameters::parse_l_max(const char** param, int num_param)
{
  is_l_max_set = true;

  if (num_param < 2 || num_param > 8) {
    PRINT_INPUT_ERROR("l_max should have 1 to 7 parameters.\n");
  }
  if (!is_valid_int(param[1], &L_max)) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should be an integer.\n");
  }
  if (L_max < 2) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should >= 2.");
  }
  if (L_max > 8) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should <= 8.");
  }

  if (num_param >= 3) {
    if (!is_valid_int(param[2], &has_q_222)) {
      PRINT_INPUT_ERROR("has_q_222 should be an integer.\n");
    }
  }

  if (num_param >= 4) {
    if (!is_valid_int(param[3], &has_q_1111)) {
      PRINT_INPUT_ERROR("has_q_1111 should be an integer.\n");
    }
  }

  if (num_param >= 5) {
    if (!is_valid_int(param[4], &has_q_112)) {
      PRINT_INPUT_ERROR("has_q_112 should be an integer.\n");
    }
  }

  if (num_param >= 6) {
    if (!is_valid_int(param[5], &has_q_123)) {
      PRINT_INPUT_ERROR("has_q_123 should be an integer.\n");
    }
  }

  if (num_param >= 7) {
    if (!is_valid_int(param[6], &has_q_233)) {
      PRINT_INPUT_ERROR("has_q_233 should be an integer.\n");
    }
  }

  if (num_param >= 8) {
    if (!is_valid_int(param[7], &has_q_134)) {
      PRINT_INPUT_ERROR("has_q_134 should be an integer.\n");
    }
  }

}

void NEP_Parameters::parse_neuron(const char** param, int num_param)
{
  is_neuron_set = true;

  if (num_param < 2 || num_param > 3) {
    PRINT_INPUT_ERROR("neuron should have 1 or 2 parameters.\n");
  }

  if (!is_valid_int(param[1], &num_neurons1)) {
    PRINT_INPUT_ERROR("number of neurons1 should be an integer.\n");
  }

  if (num_neurons1 < 1) {
    PRINT_INPUT_ERROR("number of neurons1 should >= 1.");
  } else if (num_neurons1 > 120) {
    PRINT_INPUT_ERROR("number of neurons1 should <= 120.");
  }
  num_hidden_layers = 1;

  if (num_param == 3) {
    if (!is_valid_int(param[2], &num_neurons2)) {
      PRINT_INPUT_ERROR("number of neurons in the second hidden layer should be an integer.\n");
    }
    if (num_neurons2 < 0) {
      PRINT_INPUT_ERROR("number of neurons in the second hidden layer should >= 0.");
    } else if (num_neurons2 > 120) {
      PRINT_INPUT_ERROR("number of neurons in the second hidden layer should <= 120.");
    }
    num_hidden_layers = 2;

    if (num_neurons2 == 0) {
      num_hidden_layers = 1;
    }
  }
}

void NEP_Parameters::parse_lambda_e(const char** param, int num_param)
{
  is_lambda_e_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_e should have 1 parameter.\n");
  }

  double lambda_e_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_e_tmp)) {
    PRINT_INPUT_ERROR("Energy loss weight should be a number.\n");
  }
  lambda_e = lambda_e_tmp;

  if (lambda_e < 0.0f) {
    PRINT_INPUT_ERROR("Energy loss weight should >= 0.");
  }
}

void NEP_Parameters::parse_lambda_f(const char** param, int num_param)
{
  is_lambda_f_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_f should have 1 parameter.\n");
  }

  double lambda_f_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_f_tmp)) {
    PRINT_INPUT_ERROR("Force loss weight should be a number.\n");
  }
  lambda_f = lambda_f_tmp;

  if (lambda_f < 0.0f) {
    PRINT_INPUT_ERROR("Force loss weight should >= 0.");
  }
}

void NEP_Parameters::parse_lambda_v(const char** param, int num_param)
{
  is_lambda_v_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_v should have 1 parameter.\n");
  }

  double lambda_v_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_v_tmp)) {
    PRINT_INPUT_ERROR("Virial loss weight should be a number.\n");
  }
  lambda_v = lambda_v_tmp;

  if (lambda_v < 0.0f) {
    PRINT_INPUT_ERROR("Virial loss weight should >= 0.");
  }
}

void NEP_Parameters::parse_lambda_shear(const char** param, int num_param)
{
  is_lambda_shear_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_shear should have 1 parameter.\n");
  }

  double lambda_shear_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_shear_tmp)) {
    PRINT_INPUT_ERROR("Shear virial weight should be a number.\n");
  }
  lambda_shear = lambda_shear_tmp;

  if (lambda_shear < 0.0f) {
    PRINT_INPUT_ERROR("Shear virial weight should >= 0.");
  }
}

void NEP_Parameters::parse_use_typewise_cutoff_zbl(const char** param, int num_param)
{
  if (num_param != 1 && num_param != 2) {
    PRINT_INPUT_ERROR("use_typewise_cutoff_zbl should have 0 or 1 parameter.\n");
  }
  use_typewise_cutoff_zbl = true;
  is_use_typewise_cutoff_zbl_set = true;
  typewise_cutoff_zbl_factor = 0.7f;

  if (num_param == 2) {
    double typewise_cutoff_zbl_factor_temp = 0.0;
    if (!is_valid_real(param[1], &typewise_cutoff_zbl_factor_temp)) {
      PRINT_INPUT_ERROR("typewise_cutoff_zbl_factor should be a number.\n");
    }
    typewise_cutoff_zbl_factor = typewise_cutoff_zbl_factor_temp;
  }

  if (typewise_cutoff_zbl_factor < 0.5f) {
    PRINT_INPUT_ERROR("typewise_cutoff_zbl_factor must >= 0.5.\n");
  }
}

void NEP_Parameters::parse_output_descriptor(const char** param, int num_param)
{
  output_descriptor = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("output_descriptor should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &output_descriptor)) {
    PRINT_INPUT_ERROR("output_descriptor should be an integer.\n");
  }
  if (output_descriptor < 0 || output_descriptor > 2) {
    PRINT_INPUT_ERROR("output_descriptor should >= 0 and <= 2.");
  }
}

void NEP_Parameters::parse_fine_tune(const char** param, int num_param)
{
  fine_tune = 1;

  if (num_param != 3 && num_param != 4) {
    PRINT_INPUT_ERROR("fine_tune should have 2 or 3 parameters.\n");
  }

  fine_tune_nep_txt = param[1];
  fine_tune_nep_restart = param[2];

  if (num_param == 4) {
    if (!is_valid_int(param[3], &fine_tune_descriptor)) {
      PRINT_INPUT_ERROR("fine_tune_descriptor should be an integer.\n");
    }
    if (fine_tune_descriptor < 0 || fine_tune_descriptor > 1) {
      PRINT_INPUT_ERROR("fine_tune_descriptor should be 0 or 1.");
    }
  }
}

void NEP_Parameters::parse_import_q_scaler(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("import_q_scaler should have 1 parameter.\n");
  }

  int flag = 0;
  if (!is_valid_int(param[1], &flag)) {
    PRINT_INPUT_ERROR("import_q_scaler should be an integer.\n");
  }
  if (flag != 0 && flag != 1) {
    PRINT_INPUT_ERROR("import_q_scaler should be 0 or 1.");
  }
  import_q_scaler = (flag == 1);
}

void NEP_Parameters::report_inputs(
  const ParameterInput& input, const char* model_name, const char* model_description)
{
  printf("Input or default parameters:\n");

  printf("    (%s) %smodel_type = %s.\n",
    input.is_model_type_set ? "input" : "default", input.is_model_type_set ? "  " : "",
    model_name);

  std::string calculation_mode_name = "train";
  if (prediction == 1) {
    calculation_mode_name = "predict";
  }
  if (is_prediction_set) {
    printf("    (input)   calculation mode = %s.\n", calculation_mode_name.c_str());
  } else {
    printf("    (default) calculation mode = %s.\n", calculation_mode_name.c_str());
  }

  if (is_version_set) {
    printf("    (input)   use NEP version %d.\n", version);
  } else {
    printf("    (default) use NEP version %d.\n", version);
  }
  printf("    (input)   number of atom types = %d.\n", num_types);
  for (int n = 0; n < num_types; ++n) {
    printf(
      "        type %d (%s with Z = %d) has cutoffs (%g A, %g A).\n",
      n,
      elements[n].c_str(),
      atomic_numbers[n],
      rc_radial[n],
      rc_angular[n]);
  }

  if (is_zbl_set) {
    if (flexible_zbl) {
      printf("    (input)   will add the flexible ZBL potential\n");
    } else {
      if (use_typewise_cutoff_zbl) {
        printf(
          "    (input)   will add the universal ZBL potential with typewise cutoff factor %g\n",
          typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    (input)   will add the universal ZBL potential with outer cutoff %g A and inner "
          "cutoff %g A.\n",
          zbl_rc_outer,
          zbl_rc_inner);
      }
    }
  } else {
    printf("    (default) will not add the ZBL potential.\n");
  }

  if (model_description) {
    printf("    (input)   %s\n", model_description);
  }
  input.report_disabled_models();

  if (is_n_max_set) {
    printf("    (input)   n_max_radial = %d.\n", n_max_radial);
    printf("    (input)   n_max_angular = %d.\n", n_max_angular);
  } else {
    printf("    (default) n_max_radial = %d.\n", n_max_radial);
    printf("    (default) n_max_angular = %d.\n", n_max_angular);
  }

  if (is_basis_size_set) {
    printf("    (input)   basis_size_radial = %d.\n", basis_size_radial);
    printf("    (input)   basis_size_angular = %d.\n", basis_size_angular);
  } else {
    printf("    (default) basis_size_radial = %d.\n", basis_size_radial);
    printf("    (default) basis_size_angular = %d.\n", basis_size_angular);
  }

  if (is_l_max_set) {
    printf("    (input)   l_max_3body = %d.\n", L_max);
    printf("    (input)   has_q_222 = %d.\n", has_q_222);
    printf("    (input)   has_q_1111 = %d.\n", has_q_1111);
    printf("    (input)   has_q_112 = %d.\n", has_q_112);
    printf("    (input)   has_q_123 = %d.\n", has_q_123);
    printf("    (input)   has_q_233 = %d.\n", has_q_233);
    printf("    (input)   has_q_134 = %d.\n", has_q_134);
  } else {
    printf("    (default) l_max_3body = %d.\n", L_max);
    printf("    (default) has_q_222 = %d.\n", has_q_222);
    printf("    (default) has_q_1111 = %d.\n", has_q_1111);
    printf("    (default) has_q_112 = %d.\n", has_q_112);
    printf("    (default) has_q_123 = %d.\n", has_q_123);
    printf("    (default) has_q_233 = %d.\n", has_q_233);
    printf("    (default) has_q_134 = %d.\n", has_q_134);
  }

  if (is_neuron_set) {
    if (num_hidden_layers == 2) {
      printf("    (input)   number of neurons = %d-%d.\n", num_neurons1, num_neurons2);
    } else {
      printf("    (input)   number of neurons = %d.\n", num_neurons1);
    }
  } else {
    printf("    (default) number of neurons = %d.\n", num_neurons1);
  }

  if (is_lambda_1_set) {
    printf("    (input)   lambda_1 = %g.\n", lambda_1);
  } else {
    printf("    (default) lambda_1 = %g.\n", lambda_1);
  }

  if (is_lambda_2_set) {
    printf("    (input)   lambda_2 = %g.\n", lambda_2);
  } else {
    printf("    (default) lambda_2 = %g.\n", lambda_2);
  }

  if (is_lambda_e_set) {
    printf("    (input)   lambda_e = %g.\n", lambda_e);
  } else {
    printf("    (default) lambda_e = %g.\n", lambda_e);
  }

  if (is_lambda_f_set) {
    printf("    (input)   lambda_f = %g.\n", lambda_f);
  } else {
    printf("    (default) lambda_f = %g.\n", lambda_f);
  }

  if (is_lambda_v_set) {
    printf("    (input)   lambda_v = %g.\n", lambda_v);
  } else {
    printf("    (default) lambda_v = %g.\n", lambda_v);
  }

  if (is_lambda_shear_set) {
    printf("    (input)   lambda_shear = %g.\n", lambda_shear);
  } else {
    printf("    (default) lambda_shear = %g.\n", lambda_shear);
  }

  if (is_force_delta_set) {
    printf("    (input)   force_delta = %g.\n", force_delta);
  } else {
    printf("    (default) force_delta = %g.\n", force_delta);
  }

  if (is_batch_set) {
    printf("    (input)   batch size = %d.\n", batch_size);
  } else {
    printf("    (default) batch size = %d.\n", batch_size);
  }

  if (is_population_set) {
    printf("    (input)   population size = %d.\n", population_size);
  } else {
    printf("    (default) population size = %d.\n", population_size);
  }

  if (is_seed_set) {
    printf("    (input)   random seed = %d.\n", seed);
  } else {
    printf("    (default) random seed not set.\n");
  }

  if (is_generation_set) {
    printf("    (input)   maximum number of generations = %d.\n", maximum_generation);
  } else {
    printf("    (default) maximum number of generations = %d.\n", maximum_generation);
  }

  if (is_save_potential_set) {
    printf("    (input)   save potential every N = %d generations.\n", save_potential);
  } else {
    printf("    (default) save potential every N = %d generations.\n", save_potential);
  }

  if (is_output_interval_set) {
    printf("    (input)   output_interval = %d generations.\n", output_interval);
  } else {
    printf("    (default) output_interval = %d generations.\n", output_interval);
  }

  if (fine_tune) {
    printf("    (input)   will fine-tune based on %s and %s.\n",
      fine_tune_nep_txt.c_str(), fine_tune_nep_restart.c_str());
  }

  report_model_inputs();

  // some calculated parameters:
  printf("Some calculated parameters:\n");
  printf("    number of radial descriptor components = %d.\n", dim_radial);
  printf("    number of angular descriptor components = %d.\n", dim_angular);
  printf("    total number of descriptor components = %d.\n", dim);
  if (num_hidden_layers == 2) {
    printf("    NN architecture = %d-%d-%d-1.\n", dim, num_neurons1, num_neurons2);
  } else {
    printf("    NN architecture = %d-%d-1.\n", dim, num_neurons1);
  }
  printf(
    "    number of NN parameters to be optimized = %d.\n",
    number_of_variables_ann * ann_count);
  printf(
    "    number of descriptor parameters to be optimized = %d.\n", number_of_variables_descriptor);
  printf("    total number of parameters to be optimized = %d.\n", number_of_variables);
}

bool NEP_Energy_Parameters::parse_keyword(const char** param, int num_param)
{
  if (strcmp(param[0], "lambda_d") == 0) {
    parse_lambda_d(param, num_param);
  } else {
    return NEP_Parameters::parse_keyword(param, num_param);
  }
  return true;
}

void NEP_Energy_Parameters::parse_lambda_d(const char** param, int num_param)
{
  is_lambda_d_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_d should have 1 parameter.\n");
  }

  double lambda_d_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_d_tmp)) {
    PRINT_INPUT_ERROR("Energy difference loss weight should be a number.\n");
  }
  // the weight has to be positive and survive the conversion to float
  if (!std::isfinite(lambda_d_tmp) || lambda_d_tmp < FLT_MIN || lambda_d_tmp > FLT_MAX) {
    PRINT_INPUT_ERROR(
      "Energy difference loss weight should be a finite positive number within the range of "
      "float.");
  }
  lambda_d = lambda_d_tmp;
}

void NEP_Energy_Parameters::report_model_inputs() const
{
  if (is_lambda_d_set) {
    printf("    (input)   lambda_d = %g.\n", lambda_d);
  }
}
