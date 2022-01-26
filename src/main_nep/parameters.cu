/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cmath>

const int NUM_ELEMENTS = 103;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

Parameters::Parameters(char* input_dir)
{
  print_line_1();
  printf("Started reading nep.in.\n");
  print_line_2();

  set_default_parameters();
  read_nep_in(input_dir);
  calculate_parameters();
  report_inputs();

  print_line_1();
  printf("Finished reading nep.in.\n");
  print_line_2();
}

void Parameters::set_default_parameters()
{
  is_type_set = false;
  is_cutoff_set = false;
  is_n_max_set = false;
  is_l_max_set = false;
  is_neuron_set = false;
  is_lambda_1_set = false;
  is_lambda_2_set = false;
  is_lambda_e_set = false;
  is_lambda_f_set = false;
  is_lambda_v_set = false;
  is_batch_set = false;
  is_population_set = false;
  is_generation_set = false;
  is_type_weight_set = false;
  is_zbl_set = false;

  rc_radial = 8.0f;              // large enough for vdw/coulomb
  rc_angular = 5.0f;             // large enough in most cases
  n_max_radial = 15;             // large enough in most cases
  n_max_angular = 10;            // large enough in most cases
  L_max = 4;                     // the only supported value
  num_neurons1 = 50;             // large enough in most cases
  lambda_1 = lambda_2 = 5.0e-2f; // good default based on our tests
  lambda_e = lambda_f = 1.0f;    // energy and force are more important
  lambda_v = 0.1f;               // virial is less important
  force_delta = 0.0f;            // no modification of force loss
  batch_size = 1000000;          // a very large number means full-batch
  population_size = 50;          // almost optimal
  maximum_generation = 100000;   // a good starting point
  type_weight_cpu.resize(MAX_NUM_TYPES);
  for (int n = 0; n < MAX_NUM_TYPES; ++n) {
    type_weight_cpu[n] = {1.0f}; // uniform weight by default
  }
  enable_zbl = false; // default is not to include ZBL
}

void Parameters::read_nep_in(char* input_dir)
{
  char file_para[200];
  strcpy(file_para, input_dir);
  strcat(file_para, "/nep.in");
  char* input = get_file_contents(file_para);
  char* input_ptr = input;      // Keep the pointer in order to free later
  const int max_num_param = 20; // never use more than 19 parameters
  int num_param;
  char* param[max_num_param];

  while (input_ptr) {
    input_ptr = row_find_param(input_ptr, param, &num_param);
    if (num_param == 0) {
      continue;
    }
    parse_one_keyword(param, num_param);
  }
  free(input); // Free the input file contents
}

void Parameters::calculate_parameters()
{
  dim_radial = (n_max_radial + 1);
  dim_angular = (n_max_angular + 1) * L_max;
  dim = dim_radial + dim_angular;
  q_scaler_cpu.resize(dim, 1.0e10f);
  q_scaler_gpu.resize(dim);
  q_scaler_gpu.copy_from_host(q_scaler_cpu.data());
  number_of_variables_ann = (dim + 2) * num_neurons1 + 1;
  number_of_variables_descriptor =
    (num_types == 1) ? 0 : num_types * num_types * (n_max_radial + n_max_angular + 2);
  number_of_variables = number_of_variables_ann + number_of_variables_descriptor;
  type_weight_gpu.resize(MAX_NUM_TYPES);
  type_weight_gpu.copy_from_host(type_weight_cpu.data());
}

void Parameters::report_inputs()
{
  if (!is_type_set) {
    PRINT_INPUT_ERROR("type in nep.in has not been set.");
  }

  printf("Input or default parameters:\n");
  printf("    (input)   number of atom types = %d.\n", num_types);
  if (is_type_weight_set) {
    for (int n = 0; n < num_types; ++n) {
      printf(
        "        (input)   type %d (%s with Z = %d) has force weight of %g.\n", n,
        elements[n].c_str(), atomic_numbers[n], type_weight_cpu[n]);
    }
  } else {
    for (int n = 0; n < num_types; ++n) {
      printf(
        "        (default) type %d (%s with Z = %d) has force weight of %g.\n", n,
        elements[n].c_str(), atomic_numbers[n], type_weight_cpu[n]);
    }
  }

  if (is_zbl_set) {
    printf(
      "    (input)   will add the ZBL potential with outer cutoff %g A and inner cutoff %g A.\n",
      zbl_rc_outer, zbl_rc_inner);
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

  if (is_n_max_set) {
    printf("    (input)   n_max_radial = %d.\n", n_max_radial);
    printf("    (input)   n_max_angular = %d.\n", n_max_angular);
  } else {
    printf("    (default) n_max_radial = %d.\n", n_max_radial);
    printf("    (default) n_max_angular = %d.\n", n_max_angular);
  }

  if (is_l_max_set) {
    printf("    (input)   l_max = %d.\n", L_max);
  } else {
    printf("    (default) l_max = %d.\n", L_max);
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

  if (is_generation_set) {
    printf("    (input)   maximum number of generations = %d.\n", maximum_generation);
  } else {
    printf("    (default) maximum number of generations = %d.\n", maximum_generation);
  }

  // some calcuated parameters:
  printf("Some calculated parameters:\n");
  printf("    number of radial descriptor components = %d.\n", dim_radial);
  printf("    number of angualr descriptor components = %d.\n", dim_angular);
  printf("    total number of  descriptor components = %d.\n", dim);
  printf("    NN architecture = %d-%d-1.\n", dim, num_neurons1);
  printf("    number of NN parameters to be optimized = %d.\n", number_of_variables_ann);
  printf(
    "    number of descriptor parameters to be optimized = %d.\n", number_of_variables_descriptor);
  printf("    total number of parameters to be optimized = %d.\n", number_of_variables);
}

void Parameters::parse_one_keyword(char** param, int num_param)
{
  if (strcmp(param[0], "type") == 0) {
    parse_type(param, num_param);
  } else if (strcmp(param[0], "cutoff") == 0) {
    parse_cutoff(param, num_param);
  } else if (strcmp(param[0], "n_max") == 0) {
    parse_n_max(param, num_param);
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
  } else if (strcmp(param[0], "type_weight") == 0) {
    parse_type_weight(param, num_param);
  } else if (strcmp(param[0], "force_delta") == 0) {
    parse_force_delta(param, num_param);
  } else if (strcmp(param[0], "zbl") == 0) {
    parse_zbl(param, num_param);
  } else {
    PRINT_KEYWORD_ERROR(param[0]);
  }
}

void Parameters::parse_type(char** param, int num_param)
{
  is_type_set = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("type should have at least 2 parameters.\n");
  }
  if (!is_valid_int(param[1], &num_types)) {
    PRINT_INPUT_ERROR("number of types should be integer.\n");
  }

  if (num_types < 1 || num_types > MAX_NUM_TYPES) {
    PRINT_INPUT_ERROR("number of types should >=1 and <= MAX_NUM_TYPES.");
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

void Parameters::parse_type_weight(char** param, int num_param)
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

void Parameters::parse_zbl(char** param, int num_param)
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

void Parameters::parse_force_delta(char** param, int num_param)
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

void Parameters::parse_cutoff(char** param, int num_param)
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
  if (rc_angular < 2.0f) {
    PRINT_INPUT_ERROR("angular cutoff should >= 2.0 A.");
  }
  if (rc_radial > 10.0f) {
    PRINT_INPUT_ERROR("radial cutoff should <= 10 A.");
  }
}

void Parameters::parse_n_max(char** param, int num_param)
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

void Parameters::parse_l_max(char** param, int num_param)
{
  is_l_max_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("l_max should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &L_max)) {
    PRINT_INPUT_ERROR("l_max should be an integer.\n");
  }
  if (L_max != 4) {
    PRINT_INPUT_ERROR("l_max should = 4.");
  }
}

void Parameters::parse_neuron(char** param, int num_param)
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
  } else if (num_neurons1 > 100) {
    PRINT_INPUT_ERROR("number of neurons should <= 100.");
  }
}

void Parameters::parse_lambda_1(char** param, int num_param)
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

void Parameters::parse_lambda_2(char** param, int num_param)
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

void Parameters::parse_lambda_e(char** param, int num_param)
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

void Parameters::parse_lambda_f(char** param, int num_param)
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

void Parameters::parse_lambda_v(char** param, int num_param)
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

void Parameters::parse_batch(char** param, int num_param)
{
  is_batch_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("batch should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &batch_size)) {
    PRINT_INPUT_ERROR("batch size should be an integer.\n");
  }
  if (batch_size < 1) {
    PRINT_INPUT_ERROR("batch size should >= 1.");
  }
}

void Parameters::parse_population(char** param, int num_param)
{
  is_population_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("population should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &population_size)) {
    PRINT_INPUT_ERROR("population size should be an integer.\n");
  }
  if (population_size < 10) {
    PRINT_INPUT_ERROR("population size should >= 10.");
  } else if (population_size > 100) {
    PRINT_INPUT_ERROR("population size should <= 100.");
  }
}

void Parameters::parse_generation(char** param, int num_param)
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
