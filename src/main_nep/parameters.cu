/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#include <iostream>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

Parameters::Parameters()
{
  print_line_1();
  printf("Started reading nep.in.\n");
  print_line_2();

  set_default_parameters();
  read_nep_in();
  if (is_zbl_set) {
    read_zbl_in();
  }
  calculate_parameters();
  report_inputs();

  print_line_1();
  printf("Finished reading nep.in.\n");
  print_line_2();
}

void Parameters::set_default_parameters()
{
  is_train_mode_set = false;
  is_prediction_set = false;
  is_version_set = false;
  is_type_set = false;
  is_cutoff_set = false;
  is_n_max_set = false;
  is_basis_size_set = false;
  is_l_max_set = false;
  is_neuron_set = false;
  is_lambda_1_set = false;
  is_lambda_2_set = false;
  is_lambda_e_set = false;
  is_lambda_f_set = false;
  is_lambda_v_set = false;
  is_lambda_shear_set = false;
  is_batch_set = false;
  is_population_set = false;
  is_generation_set = false;
  is_type_weight_set = false;
  is_zbl_set = false;
  is_force_delta_set = false;
  is_use_typewise_cutoff_set = false;
  is_use_typewise_cutoff_zbl_set = false;

  train_mode = 0;              // potential
  prediction = 0;              // not prediction mode
  version = 4;                 // NEP4 is the best
  rc_radial = 8.0f;            // large enough for vdw/coulomb
  rc_angular = 4.0f;           // large enough in most cases
  basis_size_radial = 8;       // large enough in most cases
  basis_size_angular = 8;      // large enough in most cases
  n_max_radial = 4;            // a relatively small value to achieve high speed
  n_max_angular = 4;           // a relatively small value to achieve high speed
  L_max = 4;                   // the only supported value
  L_max_4body = 2;             // default is to include 4body
  L_max_5body = 0;             // default is not to include 5body
  num_neurons1 = 30;           // a relatively small value to achieve high speed
  lambda_1 = lambda_2 = -1.0f; // automatic regularization
  lambda_e = lambda_f = 1.0f;  // energy and force are more important
  lambda_v = 0.1f;             // virial is less important
  lambda_shear = 1.0f;         // do not weight shear virial more by default
  force_delta = 0.0f;          // no modification of force loss
  batch_size = 1000;           // large enough in most cases
  use_full_batch = 0;          // default is not to enable effective full-batch
  population_size = 50;        // almost optimal
  maximum_generation = 100000; // a good starting point
  initial_para = 1.0f;
  sigma0 = 0.1f;
  use_typewise_cutoff = false;
  use_typewise_cutoff_zbl = false;
  typewise_cutoff_radial_factor = -1.0f;
  typewise_cutoff_angular_factor = -1.0f;
  typewise_cutoff_zbl_factor = -1.0f;

  type_weight_cpu.resize(NUM_ELEMENTS);
  zbl_para.resize(550); // Maximum number of zbl parameters
  for (int n = 0; n < NUM_ELEMENTS; ++n) {
    type_weight_cpu[n] = 1.0f; // uniform weight by default
  }
  enable_zbl = false;   // default is not to include ZBL
  flexible_zbl = false; // default Universal ZBL
}

void Parameters::read_nep_in()
{
  std::ifstream input("nep.in");
  if (!input.is_open()) {
    std::cout << "Failed to open nep.in." << std::endl;
    exit(1);
  }

  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    std::vector<std::string> tokens_without_comments;
    for (const auto& t : tokens) {
      if (t[0] != '#') {
        tokens_without_comments.emplace_back(t);
      } else {
        break;
      }
    }
    if (tokens_without_comments.size() > 0) {
      parse_one_keyword(tokens_without_comments);
    }
  }

  input.close();
}

void Parameters::read_zbl_in()
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

void Parameters::calculate_parameters()
{
  if (version == 5 && train_mode != 0) {
    PRINT_INPUT_ERROR("Can only use NEP5 for potential model.");
  }

  if (train_mode != 0 && train_mode != 3) {
    // take virial as dipole or polarizability
    lambda_e = lambda_f = 0.0f;
    enable_zbl = false;
    if (!is_lambda_v_set) {
      lambda_v = 1.0f;
    }
  }
  dim_radial = n_max_radial + 1;             // 2-body descriptors q^i_n
  dim_angular = (n_max_angular + 1) * L_max; // 3-body descriptors q^i_nl
  if (L_max_4body == 2) {                    // 4-body descriptors q^i_n222
    dim_angular += n_max_angular + 1;
  }
  if (L_max_5body == 1) { // 5-body descriptors q^i_n1111
    dim_angular += n_max_angular + 1;
  }
  dim = dim_radial + dim_angular;
  if (train_mode == 3) {
    dim += 1; // concatenate temeprature with descriptors
  }
  q_scaler_cpu.resize(dim, 1.0e10f);
#ifdef USE_FIXED_SCALER
  for (int n = 0; n < q_scaler_cpu.size(); ++n) {
    q_scaler_cpu[n] = 0.01f;
  }
#endif

  if (version == 3) {
    number_of_variables_ann = (dim + 2) * num_neurons1 + 1;
  } else if (version == 4) {
    number_of_variables_ann = (dim + 2) * num_neurons1 * num_types + 1;
  } else if (version == 5) {
    number_of_variables_ann = ((dim + 2) * num_neurons1 + 1) * num_types + 1;
  }

  number_of_variables_descriptor =
    num_types * num_types *
    (dim_radial * (basis_size_radial + 1) + (n_max_angular + 1) * (basis_size_angular + 1));

  number_of_variables = number_of_variables_ann + number_of_variables_descriptor;
  if (train_mode == 2) {
    number_of_variables += number_of_variables_ann;
  }

  if (version != 3) {
    if (!is_lambda_1_set) {
      lambda_1 = sqrt(number_of_variables * 1.0e-6f / num_types);
    }
    if (!is_lambda_2_set) {
      lambda_2 = sqrt(number_of_variables * 1.0e-6f / num_types);
    }
  } else {
    if (!is_lambda_1_set) {
      lambda_1 = sqrt(number_of_variables * 1.0e-6f);
    }
    if (!is_lambda_2_set) {
      lambda_2 = sqrt(number_of_variables * 1.0e-6f);
    }
  }

  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));
  for (int device_id = 0; device_id < deviceCount; device_id++) {
    CHECK(cudaSetDevice(device_id));
    q_scaler_gpu[device_id].resize(dim);
    q_scaler_gpu[device_id].copy_from_host(q_scaler_cpu.data());
  }
}

void Parameters::report_inputs()
{
  if (!is_type_set) {
    PRINT_INPUT_ERROR("type in nep.in has not been set.");
  }

  printf("Input or default parameters:\n");

  std::string train_mode_name = "potential";
  if (train_mode == 1) {
    train_mode_name = "dipole";
  } else if (train_mode == 2) {
    train_mode_name = "polarizability";
  } else if (train_mode == 3) {
    train_mode_name = "temperature-dependent free energy";
  }
  if (is_train_mode_set) {
    printf("    (input)   model_type = %s.\n", train_mode_name.c_str());
  } else {
    printf("    (default) model_type = %s.\n", train_mode_name.c_str());
  }

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
  if (is_type_weight_set) {
    for (int n = 0; n < num_types; ++n) {
      printf(
        "        (input)   type %d (%s with Z = %d) has force weight of %g.\n",
        n,
        elements[n].c_str(),
        atomic_numbers[n],
        type_weight_cpu[n]);
    }
  } else {
    for (int n = 0; n < num_types; ++n) {
      printf(
        "        (default) type %d (%s with Z = %d) has force weight of %g.\n",
        n,
        elements[n].c_str(),
        atomic_numbers[n],
        type_weight_cpu[n]);
    }
  }

  if (is_zbl_set) {
    if (flexible_zbl) {
      printf("    (input)   will add the flexible ZBL potential\n");
    } else {
      printf(
        "    (input)   will add the universal ZBL potential with outer cutoff %g A and inner "
        "cutoff %g A.\n",
        zbl_rc_outer,
        zbl_rc_inner);
    }
  } else {
    printf("    (default) will not add the ZBL potential.\n");
  }

  if (is_cutoff_set) {
    printf("    (input)   radial cutoff = %g A.\n", rc_radial);
    printf("    (input)   angular cutoff = %g A.\n", rc_angular);
  } else {
    printf("    (default) radial cutoff = %g A.\n", rc_radial);
    printf("    (default) angular cutoff = %g A.\n", rc_angular);
  }

  if (is_use_typewise_cutoff_set) {
    printf("    (input)   use %s cutoff for NEP.\n", use_typewise_cutoff ? "typewise" : "global");
    printf("              radial factor = %g.\n", typewise_cutoff_radial_factor);
    printf("              angular factor = %g.\n", typewise_cutoff_angular_factor);
  } else {
    printf("    (default) use %s cutoff for NEP.\n", use_typewise_cutoff ? "typewise" : "global");
  }

  if (is_use_typewise_cutoff_zbl_set) {
    printf(
      "    (input)   use %s cutoff for ZBL.\n", use_typewise_cutoff_zbl ? "typewise" : "global");
    printf("              factor = %g.\n", typewise_cutoff_zbl_factor);
  } else {
    printf(
      "    (default) use %s cutoff for ZBL.\n", use_typewise_cutoff_zbl ? "typewise" : "global");
  }

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
    printf("    (input)   l_max_4body = %d.\n", L_max_4body);
    printf("    (input)   l_max_5body = %d.\n", L_max_5body);
  } else {
    printf("    (default) l_max_3body = %d.\n", L_max);
    printf("    (default) l_max_4body = %d.\n", L_max_4body);
    printf("    (default) l_max_5body = %d.\n", L_max_5body);
  }

  if (is_neuron_set) {
    printf("    (input)   number of neurons = %d.\n", num_neurons1);
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
    if (use_full_batch) {
      printf("        enable effective full-batch.\n");
    }
  } else {
    printf("    (default) batch size = %d.\n", batch_size);
  }

  if (is_population_set) {
    printf("    (input)   population size = %d.\n", population_size);
  } else {
    printf("    (default) population size = %d.\n", population_size);
  }

  if (is_generation_set) {
    printf("    (input)   maximum number of generations = %d.\n", maximum_generation);
  } else {
    printf("    (default) maximum number of generations = %d.\n", maximum_generation);
  }

  // some calcuated parameters:
  printf("Some calculated parameters:\n");
  printf("    number of radial descriptor components = %d.\n", dim_radial);
  printf("    number of angular descriptor components = %d.\n", dim_angular);
  printf("    total number of descriptor components = %d.\n", dim);
  printf("    NN architecture = %d-%d-1.\n", dim, num_neurons1);
  printf(
    "    number of NN parameters to be optimized = %d.\n",
    number_of_variables_ann * (train_mode == 2 ? 2 : 1));
  printf(
    "    number of descriptor parameters to be optimized = %d.\n", number_of_variables_descriptor);
  printf("    total number of parameters to be optimized = %d.\n", number_of_variables);
}

void Parameters::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const char* param[105]; // never use more than 104 parameters
  for (int n = 0; n < num_param; ++n) {
    param[n] = tokens[n].c_str();
  }
  if (strcmp(param[0], "model_type") == 0 || strcmp(param[0], "mode") == 0) {
    parse_mode(param, num_param);
  } else if (strcmp(param[0], "prediction") == 0) {
    parse_prediction(param, num_param);
  } else if (strcmp(param[0], "version") == 0) {
    parse_version(param, num_param);
  } else if (strcmp(param[0], "type") == 0) {
    parse_type(param, num_param);
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
  } else if (strcmp(param[0], "batch") == 0) {
    parse_batch(param, num_param);
  } else if (strcmp(param[0], "population") == 0) {
    parse_population(param, num_param);
  } else if (strcmp(param[0], "generation") == 0) {
    parse_generation(param, num_param);
  } else if (strcmp(param[0], "lambda_1") == 0) {
    parse_lambda_1(param, num_param);
  } else if (strcmp(param[0], "lambda_2") == 0) {
    parse_lambda_2(param, num_param);
  } else if (strcmp(param[0], "lambda_e") == 0) {
    parse_lambda_e(param, num_param);
  } else if (strcmp(param[0], "lambda_f") == 0) {
    parse_lambda_f(param, num_param);
  } else if (strcmp(param[0], "lambda_v") == 0) {
    parse_lambda_v(param, num_param);
  } else if (strcmp(param[0], "lambda_shear") == 0) {
    parse_lambda_shear(param, num_param);
  } else if (strcmp(param[0], "type_weight") == 0) {
    parse_type_weight(param, num_param);
  } else if (strcmp(param[0], "force_delta") == 0) {
    parse_force_delta(param, num_param);
  } else if (strcmp(param[0], "zbl") == 0) {
    parse_zbl(param, num_param);
  } else if (strcmp(param[0], "initial_para") == 0) {
    parse_initial_para(param, num_param);
  } else if (strcmp(param[0], "sigma0") == 0) {
    parse_sigma0(param, num_param);
  } else if (strcmp(param[0], "use_typewise_cutoff") == 0) {
    parse_use_typewise_cutoff(param, num_param);
  } else if (strcmp(param[0], "use_typewise_cutoff_zbl") == 0) {
    parse_use_typewise_cutoff_zbl(param, num_param);
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}

void Parameters::parse_mode(const char** param, int num_param)
{
  is_train_mode_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("model_type should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &train_mode)) {
    PRINT_INPUT_ERROR("mode should be an integer.\n");
  }
  if (train_mode != 0 && train_mode != 1 && train_mode != 2 && train_mode != 3) {
    PRINT_INPUT_ERROR("model_type should = 0 or 1 or 2 or 3.");
  }
}

void Parameters::parse_prediction(const char** param, int num_param)
{
  is_prediction_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("prediction should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &prediction)) {
    PRINT_INPUT_ERROR("prediction should be an integer.\n");
  }
  if (prediction != 0 && prediction != 1) {
    PRINT_INPUT_ERROR("prediction should = 0 or 1.");
  }
}

void Parameters::parse_version(const char** param, int num_param)
{
  is_version_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("version should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &version)) {
    PRINT_INPUT_ERROR("version should be an integer.\n");
  }
  if (version < 3 || version > 5) {
    PRINT_INPUT_ERROR("version should = 3 or 4 or 5.");
  }
}

void Parameters::parse_type(const char** param, int num_param)
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
      if (elements.back() == ELEMENTS[m]) {
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

void Parameters::parse_type_weight(const char** param, int num_param)
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

void Parameters::parse_zbl(const char** param, int num_param)
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
  } else if (zbl_rc_outer > 2.5f) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should <= 2.5 A.");
  }
}

void Parameters::parse_force_delta(const char** param, int num_param)
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

void Parameters::parse_cutoff(const char** param, int num_param)
{
  is_cutoff_set = true;

  if (num_param != 3) {
    PRINT_INPUT_ERROR("cutoff should have 2 parameters.\n");
  }

  double rc_radial_tmp = 0.0;
  if (!is_valid_real(param[1], &rc_radial_tmp)) {
    PRINT_INPUT_ERROR("radial cutoff should be a number.\n");
  }
  rc_radial = rc_radial_tmp;

  double rc_angular_tmp = 0.0;
  if (!is_valid_real(param[2], &rc_angular_tmp)) {
    PRINT_INPUT_ERROR("angular cutoff should be a number.\n");
  }
  rc_angular = rc_angular_tmp;

  if (rc_angular > rc_radial) {
    PRINT_INPUT_ERROR("angular cutoff should <= radial cutoff.");
  }
  if (rc_angular < 2.5f) {
    PRINT_INPUT_ERROR("angular cutoff should >= 2.5 A.");
  }
  if (rc_radial > 10.0f) {
    PRINT_INPUT_ERROR("radial cutoff should <= 10 A.");
  }
}

void Parameters::parse_n_max(const char** param, int num_param)
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
  } else if (n_max_radial > 19) {
    PRINT_INPUT_ERROR("n_max_radial should <= 19.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 19) {
    PRINT_INPUT_ERROR("n_max_angular should <= 19.");
  }
}

void Parameters::parse_basis_size(const char** param, int num_param)
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
  } else if (basis_size_radial > 19) {
    PRINT_INPUT_ERROR("basis_size_radial should <= 19.");
  }
  if (basis_size_angular < 0) {
    PRINT_INPUT_ERROR("basis_size_angular should >= 0.");
  } else if (basis_size_angular > 19) {
    PRINT_INPUT_ERROR("basis_size_angular should <= 19.");
  }
}

void Parameters::parse_l_max(const char** param, int num_param)
{
  is_l_max_set = true;

  if (num_param != 2 && num_param != 3 && num_param != 4) {
    PRINT_INPUT_ERROR("l_max should have 1 or 2 or 3 parameters.\n");
  }
  if (!is_valid_int(param[1], &L_max)) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should be an integer.\n");
  }
  if (L_max < 0) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should >= 0.");
  }
  if (L_max > 8) {
    PRINT_INPUT_ERROR("l_max for 3-body descriptors should <= 8.");
  }

  if (num_param >= 3) {
    if (!is_valid_int(param[2], &L_max_4body)) {
      PRINT_INPUT_ERROR("l_max for 4-body descriptors should be an integer.\n");
    }
    if (L_max_4body != 0 && L_max_4body != 2) {
      PRINT_INPUT_ERROR("l_max for 4-body descriptors should = 0 or 2.");
    }
    if (L_max < L_max_4body) {
      PRINT_INPUT_ERROR("l_max_4body should <= l_max_3body.");
    }
  }

  if (num_param == 4) {
    if (!is_valid_int(param[3], &L_max_5body)) {
      PRINT_INPUT_ERROR("l_max for 5-body descriptors should be an integer.\n");
    }
    if (L_max_5body != 0 && L_max_5body != 1) {
      PRINT_INPUT_ERROR("l_max for 5-body descriptors should = 0 or 1.");
    }
    if (L_max_4body == 0 && L_max_5body == 1) {
      PRINT_INPUT_ERROR("cannot have l_max_4body = 0 with l_max_5body = 1.");
    }
  }
}

void Parameters::parse_neuron(const char** param, int num_param)
{
  is_neuron_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("neuron should have 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &num_neurons1)) {
    PRINT_INPUT_ERROR("number of neurons should be an integer.\n");
  }
  if (num_neurons1 < 1) {
    PRINT_INPUT_ERROR("number of neurons should >= 1.");
  } else if (num_neurons1 > 200) {
    PRINT_INPUT_ERROR("number of neurons should <= 200.");
  }
}

void Parameters::parse_lambda_1(const char** param, int num_param)
{
  is_lambda_1_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_1 should have 1 parameter.\n");
  }

  double lambda_1_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_1_tmp)) {
    PRINT_INPUT_ERROR("L1 regularization loss weight should be a number.\n");
  }
  lambda_1 = lambda_1_tmp;

  if (lambda_1 < 0.0f) {
    PRINT_INPUT_ERROR("L1 regularization loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_2(const char** param, int num_param)
{
  is_lambda_2_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_2 should have 1 parameter.\n");
  }

  double lambda_2_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_2_tmp)) {
    PRINT_INPUT_ERROR("L2 regularization loss weight should be a number.\n");
  }
  lambda_2 = lambda_2_tmp;

  if (lambda_2 < 0.0f) {
    PRINT_INPUT_ERROR("L2 regularization loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_e(const char** param, int num_param)
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

void Parameters::parse_lambda_f(const char** param, int num_param)
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

void Parameters::parse_lambda_v(const char** param, int num_param)
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

void Parameters::parse_lambda_shear(const char** param, int num_param)
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

void Parameters::parse_batch(const char** param, int num_param)
{
  is_batch_set = true;

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("batch should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &batch_size)) {
    PRINT_INPUT_ERROR("batch size should be an integer.\n");
  }
  if (batch_size < 1) {
    PRINT_INPUT_ERROR("batch size should >= 1.");
  }

  if (num_param == 3) {
    if (!is_valid_int(param[2], &use_full_batch)) {
      PRINT_INPUT_ERROR("use_full_batch should be an integer.\n");
    }
    if (use_full_batch != 0 && use_full_batch != 1) {
      PRINT_INPUT_ERROR("use_full_batch should = 0 or 1.");
    }
  }
}

void Parameters::parse_population(const char** param, int num_param)
{
  is_population_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("The population keyword must be followed by a parameter.\n");
  }
  if (!is_valid_int(param[1], &population_size)) {
    PRINT_INPUT_ERROR("population size should be an integer.\n");
  }
  if (population_size < 10) {
    PRINT_INPUT_ERROR("population size should >= 10.");
  } else if (population_size > 200) {
    PRINT_INPUT_ERROR("population size should <= 200.");
  }

  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));
  int fully_used_device = population_size % deviceCount;
  int population_should_increase;
  if (fully_used_device != 0) {
    population_should_increase = deviceCount - fully_used_device;
    population_size += population_should_increase;
  } else {
    population_should_increase = 0;
  }
  if (population_should_increase != 0) {
    printf("The input population size is not divisible by the number of GPUs.\n");
    printf("This causes an inefficient use of resources.\n");
    printf("The population size has therefore been increased to %d.\n", population_size);
  }
}

void Parameters::parse_generation(const char** param, int num_param)
{
  is_generation_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("generation should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &maximum_generation)) {
    PRINT_INPUT_ERROR("maximum number of generations should be an integer.\n");
  }
  if (maximum_generation < 0) {
    PRINT_INPUT_ERROR("maximum number of generations should >= 0.");
  } else if (maximum_generation > 10000000) {
    PRINT_INPUT_ERROR("maximum number of generations should <= 10000000.");
  }
}

void Parameters::parse_initial_para(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("initial_para should have 1 parameter.\n");
  }

  double initial_para_tmp = 0.0;
  if (!is_valid_real(param[1], &initial_para_tmp)) {
    PRINT_INPUT_ERROR("initial_para should be a number.\n");
  }
  initial_para = initial_para_tmp;

  if (initial_para < 0.1f || initial_para > 1.0f) {
    PRINT_INPUT_ERROR("initial_para should be within [0.1, 1].");
  }
}

void Parameters::parse_sigma0(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("sigma0 should have 1 parameter.\n");
  }

  double sigma0_tmp = 0.0;
  if (!is_valid_real(param[1], &sigma0_tmp)) {
    PRINT_INPUT_ERROR("sigma0 should be a number.\n");
  }
  sigma0 = sigma0_tmp;

  if (sigma0 < 0.01f || sigma0 > 0.1f) {
    PRINT_INPUT_ERROR("sigma0 should be within [0.01, 0.1].");
  }
}

void Parameters::parse_use_typewise_cutoff(const char** param, int num_param)
{
  if (num_param != 1 && num_param != 3) {
    PRINT_INPUT_ERROR("use_typewise_cutoff should have 0 or 2 parameters.\n");
  }
  use_typewise_cutoff = true;
  is_use_typewise_cutoff_set = true;
  typewise_cutoff_radial_factor = 2.5f;
  typewise_cutoff_angular_factor = 2.0f;

  if (num_param == 3) {
    double typewise_cutoff_radial_factor_temp = 0.0;
    if (!is_valid_real(param[1], &typewise_cutoff_radial_factor_temp)) {
      PRINT_INPUT_ERROR("typewise_cutoff_radial_factor should be a number.\n");
    }
    typewise_cutoff_radial_factor = typewise_cutoff_radial_factor_temp;

    double typewise_cutoff_angular_factor_temp = 0.0;
    if (!is_valid_real(param[2], &typewise_cutoff_angular_factor_temp)) {
      PRINT_INPUT_ERROR("typewise_cutoff_angular_factor should be a number.\n");
    }
    typewise_cutoff_angular_factor = typewise_cutoff_angular_factor_temp;
  }

  if (typewise_cutoff_angular_factor < 1.5f) {
    PRINT_INPUT_ERROR("typewise_cutoff_angular_factor must >= 1.5.\n");
  }

  if (typewise_cutoff_radial_factor < typewise_cutoff_angular_factor) {
    PRINT_INPUT_ERROR("typewise_cutoff_radial_factor must >= typewise_cutoff_angular_factor.\n");
  }
}

void Parameters::parse_use_typewise_cutoff_zbl(const char** param, int num_param)
{
  if (num_param != 1 && num_param != 2) {
    PRINT_INPUT_ERROR("use_typewise_cutoff_zbl should have 0 or 1 parameter.\n");
  }
  use_typewise_cutoff_zbl = true;
  is_use_typewise_cutoff_zbl_set = true;
  typewise_cutoff_zbl_factor = 0.65f;

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
