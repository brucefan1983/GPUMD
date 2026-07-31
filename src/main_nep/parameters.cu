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
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cctype>
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
  is_atomic_v_set = false;
  is_lambda_shear_set = false;
  is_batch_set = false;
  is_population_set = false;
  is_generation_set = false;
  is_type_weight_set = false;
  is_zbl_set = false;
  is_force_delta_set = false;
  is_use_typewise_cutoff_zbl_set = false;
  is_charge_mode_set = false;
  is_save_potential_set = false;
  is_output_interval_set = false;

  train_mode = 0;              // potential
  prediction = 0;              // not prediction mode
  version = 4;                 // NEP4 is the best
  basis_size_radial = 6;       // large enough in most cases
  basis_size_angular = 6;      // large enough in most cases
  n_max_radial = 6;            // large enough in most cases
  n_max_angular = 6;           // large enough in most cases
  L_max = 4;                   // the only supported value
  has_q_222 = 1;               // default is to include q_222
  has_q_1111 = 0;              // default is not to include q_1111
  has_q_112 = 0;               // default is not to include q_112
  has_q_123 = 0;               // default is not to include q_123
  has_q_233 = 0;               // default is not to include q_233
  has_q_134 = 0;               // default is not to include q_134
  num_neurons1 = 30;           // a relatively small value to achieve high speed
  lambda_1 = lambda_2 = -1.0f; // automatic regularization
  lambda_e = lambda_f = 1.0f;  // energy and force are more important
  lambda_v = 0.1f;             // virial is less important
  lambda_shear = 1.0f;         // do not weight shear virial more by default
  lambda_q = 0.1f;             // close to optimal
  lambda_z = 0.5f;             // close to optimal
  force_delta = 0.0f;          // no modification of force loss
  batch_size = 1000;           // large enough in most cases
  use_full_batch = 0;          // default is not to enable effective full-batch
  population_size = 50;        // almost optimal
  maximum_generation = 100000; // a good starting point
  save_potential = 100000;     // write checkpoint nep.txt files at these intervals
  save_potential_format = 1;   // 1 = include time stamp when writing checkpoint nep.txt files
  output_interval = 100;       // write loss.out, nep.txt, nep.restart (and related output) every N generations
  initial_para = 1.0f;
  sigma0 = 0.1f;
  atomic_v = 0;
  use_typewise_cutoff_zbl = false;
  typewise_cutoff_zbl_factor = -1.0f;
  output_descriptor = false;
  charge_mode = 0;

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

  // ------------new--------------
  int deviceCount;  
  CHECK(gpuGetDeviceCount(&deviceCount));  
  int fully_used_device = population_size % deviceCount;  
  if (fully_used_device != 0) {  
    int population_should_increase = deviceCount - fully_used_device;  
    population_size += population_should_increase;  
    printf("Default population size adjusted from 50 to %d for GPU compatibility.\n", population_size);  
  }  
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
  if (charge_mode) {
    if (train_mode != 0) {
      PRINT_INPUT_ERROR("Charge is only supported for potential model.");
    }
  }

  if (train_mode == 0) {
    if (atomic_v == 1) {
      PRINT_INPUT_ERROR("Atomic tensor is only supported for dipole or polarizability model.");
    }
  }

  if (train_mode != 0 && train_mode != 3) {
    // take virial as dipole or polarizability
    lambda_e = lambda_f = 0.0f;
    enable_zbl = false;
    if (!is_lambda_v_set) {
      lambda_v = 1.0f; // by default, dipole or polarizability is fitted with global quantities
    }
  }
  dim_radial = n_max_radial + 1;             // 2-body descriptors q^i_n
  dim_angular = (n_max_angular + 1) * L_max; // 3-body descriptors q^i_nl
  if (has_q_222) {
    dim_angular += n_max_angular + 1;
  }
  if (has_q_1111) {
    dim_angular += n_max_angular + 1;
  }
  if (has_q_112) {
    dim_angular += n_max_angular + 1;
  }
  if (has_q_123) {
    dim_angular += n_max_angular + 1;
  }
  if (has_q_233) {
    dim_angular += n_max_angular + 1;
  }
  if (has_q_134) {
    dim_angular += n_max_angular + 1;
  }

  if (dim_angular > 90) {
    PRINT_INPUT_ERROR("Number of angular descriptors should not exceed 90.");
  }

  dim = dim_radial + dim_angular;
  if (train_mode == 3) {
    dim += 1; // concatenate temeprature with descriptors
  }

  if (num_hidden_layers == 2) {
    number_of_variables_ann_1 = (dim + 1) * num_neurons1 + (num_neurons1 + 2) * num_neurons2;
  } else {
    number_of_variables_ann_1 = (dim + 2) * num_neurons1;
  }
  number_of_variables_ann = number_of_variables_ann_1 * num_types + 1;
  if (charge_mode) {
    number_of_variables_ann_1 += num_neurons1;
    number_of_variables_ann += num_neurons1 * num_types + 1;
  }

  number_of_variables_descriptor =
    num_types * num_types *
    (dim_radial * (basis_size_radial + 1) + (n_max_angular + 1) * (basis_size_angular + 1));

  number_of_variables = number_of_variables_ann + number_of_variables_descriptor;
  if (train_mode == 2) {
    number_of_variables += number_of_variables_ann;
  }

  if (!is_lambda_1_set) {
    lambda_1 = sqrt(number_of_variables * 1.0e-6f / num_types);
  }
  if (!is_lambda_2_set) {
    lambda_2 = sqrt(number_of_variables * 1.0e-6f / num_types);
  }

  // check nep.in against any model files that are already present before reading from them
  check_existing_model();

  q_scaler_cpu.resize(dim,  1.0e10f);
  if (fine_tune) {
    std::ifstream input(fine_tune_nep_txt);
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open foundation model file.");
    }
    std::vector<std::string> tokens;
    const int NUM89 = 89;
    const int num_ann = NUM89 * number_of_variables_ann_1 + (charge_mode ? 2 : 1);
    const int num_cnk_radial = NUM89 * NUM89 * (n_max_radial + 1) * (basis_size_radial + 1);
    const int num_cnk_angular = NUM89 * NUM89 * (n_max_angular + 1) * (basis_size_angular + 1);
    const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
    for (int n = 0; n < num_tot + number_of_nep_txt_header_lines; ++n) {
      tokens = get_tokens(input); // not used
    }
    for (int n = 0; n < q_scaler_cpu.size(); ++n) {
      tokens = get_tokens(input);
      q_scaler_cpu[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    input.close();
  } else if (import_q_scaler) {
    std::ifstream input("nep.txt");
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open nep.txt for q_scaler import.");
    }
    std::vector<std::string> tokens;
    // Unlike the fine_tune case above, the imported nep.txt has been validated by
    // check_existing_model to have the exact same architecture and species count as the
    // current run, so no species-count-specific offset arithmetic is needed here: just skip
    // the header lines and this run's own number_of_variables parameter lines, then read the
    // trailing dim-line q_scaler block.
    for (int n = 0; n < number_of_nep_txt_header_lines + number_of_variables; ++n) {
      tokens = get_tokens(input); // not used
    }
    for (int n = 0; n < q_scaler_cpu.size(); ++n) {
      tokens = get_tokens(input);
      q_scaler_cpu[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    input.close();
  }

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

// decompose a model token such as nep4_zbl_charge1 into the properties it encodes
static bool parse_model_token(const std::string& token, NepTxtHeader& header)
{
  if (
    token.size() < 4 || token.compare(0, 3, "nep") != 0 ||
    !isdigit(static_cast<unsigned char>(token[3]))) {
    return false;
  }
  header.version = token[3] - '0';
  header.enable_zbl = false;
  header.train_mode = 0;
  header.charge_mode = 0;
  std::string rest = token.substr(4);
  if (rest.compare(0, 4, "_zbl") == 0) {
    header.enable_zbl = true;
    rest = rest.substr(4);
  }
  if (rest.empty() || rest == "_temperature") {
    header.train_mode = rest.empty() ? 0 : 3;
    return true;
  }
  if (rest == "_dipole") {
    header.train_mode = 1;
    return true;
  }
  if (rest == "_polarizability") {
    header.train_mode = 2;
    return true;
  }
  if (
    rest.size() == 8 && rest.compare(0, 7, "_charge") == 0 &&
    isdigit(static_cast<unsigned char>(rest[7]))) {
    header.charge_mode = rest[7] - '0';
    return true;
  }
  return false;
}

static bool check_header_line(
  const std::vector<std::string>& tokens,
  const char* keyword,
  const size_t min_size,
  const size_t max_size,
  const std::string& filename,
  std::string& error)
{
  if (tokens.size() < min_size || tokens.size() > max_size || tokens[0] != keyword) {
    error = "The " + std::string(keyword) + " line of " + filename + " is malformed.";
    return false;
  }
  return true;
}

bool read_nep_txt_header(const std::string& filename, NepTxtHeader& header, std::string& error)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    error = "Failed to open " + filename + ".";
    return false;
  }

  // first line: the model token, the number of types, and the elements
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    error = "The first line of " + filename + " should have at least 3 items.";
    return false;
  }
  header.model_token = tokens[0];
  if (!parse_model_token(header.model_token, header)) {
    error = "'" + header.model_token + "' in " + filename + " is not a known NEP model type.";
    return false;
  }
  header.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (header.num_types < 1 || tokens.size() != size_t(header.num_types) + 2) {
    error = "The first line of " + filename + " does not list " + std::to_string(header.num_types) +
            " elements.";
    return false;
  }
  header.elements.assign(tokens.begin() + 2, tokens.end());
  header.number_of_header_lines = 1;

  // second line: zbl, written only for a model with ZBL
  header.flexible_zbl = false;
  header.use_typewise_cutoff_zbl = false;
  header.zbl_rc_inner = 0.0f;
  header.zbl_rc_outer = 0.0f;
  header.typewise_cutoff_zbl_factor = -1.0f;
  if (header.enable_zbl) {
    tokens = get_tokens(input);
    ++header.number_of_header_lines;
    if (!check_header_line(tokens, "zbl", 3, 4, filename, error)) {
      return false;
    }
    header.zbl_rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    header.zbl_rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    header.flexible_zbl = (header.zbl_rc_inner == 0.0f && header.zbl_rc_outer == 0.0f);
    if (tokens.size() == 4) {
      header.use_typewise_cutoff_zbl = true;
      header.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
    }
  }

  // cutoff line: the two neighbor counts at the end depend on the training set and are ignored
  tokens = get_tokens(input);
  ++header.number_of_header_lines;
  if (!check_header_line(tokens, "cutoff", 5, size_t(header.num_types) * 2 + 3, filename, error)) {
    return false;
  }
  header.has_multiple_cutoffs =
    (tokens.size() == size_t(header.num_types) * 2 + 3 && tokens.size() > 5);
  const int num_cutoffs = header.has_multiple_cutoffs ? header.num_types : 1;
  if (tokens.size() != size_t(num_cutoffs) * 2 + 3) {
    error = "The cutoff line of " + filename + " is malformed.";
    return false;
  }
  header.rc_radial.resize(num_cutoffs);
  header.rc_angular.resize(num_cutoffs);
  for (int n = 0; n < num_cutoffs; ++n) {
    header.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
    header.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
  }

  tokens = get_tokens(input);
  ++header.number_of_header_lines;
  if (!check_header_line(tokens, "n_max", 3, 3, filename, error)) {
    return false;
  }
  header.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  header.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  tokens = get_tokens(input);
  ++header.number_of_header_lines;
  if (!check_header_line(tokens, "basis_size", 3, 3, filename, error)) {
    return false;
  }
  header.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  header.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max line: between 3 and 7 values, the trailing ones defaulting to zero when absent
  tokens = get_tokens(input);
  ++header.number_of_header_lines;
  if (!check_header_line(tokens, "l_max", 4, 8, filename, error)) {
    return false;
  }
  header.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  header.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  header.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  header.has_q_112 = (tokens.size() > 4) ? get_int_from_token(tokens[4], __FILE__, __LINE__) : 0;
  header.has_q_123 = (tokens.size() > 5) ? get_int_from_token(tokens[5], __FILE__, __LINE__) : 0;
  header.has_q_233 = (tokens.size() > 6) ? get_int_from_token(tokens[6], __FILE__, __LINE__) : 0;
  header.has_q_134 = (tokens.size() > 7) ? get_int_from_token(tokens[7], __FILE__, __LINE__) : 0;

  tokens = get_tokens(input);
  ++header.number_of_header_lines;
  if (!check_header_line(tokens, "ANN", 3, 3, filename, error)) {
    return false;
  }
  header.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  header.num_neurons2 = get_int_from_token(tokens[2], __FILE__, __LINE__);

  input.close();
  return true;
}

static std::string source_of_value(const bool is_set) { return is_set ? " (input)" : " (default)"; }

static std::string float_to_string(const float value)
{
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%g", value);
  return std::string(buffer);
}

static void compare_int(
  const char* name,
  const int from_nep_in,
  const bool is_set,
  const int from_file,
  const std::string& filename,
  std::vector<std::string>& mismatches)
{
  if (from_nep_in != from_file) {
    mismatches.push_back(
      std::string(name) + ": nep.in gives " + std::to_string(from_nep_in) +
      source_of_value(is_set) + ", " + filename + " gives " + std::to_string(from_file) + ".");
  }
}

static void compare_float(
  const std::string& name,
  const float from_nep_in,
  const bool is_set,
  const float from_file,
  const std::string& filename,
  std::vector<std::string>& mismatches)
{
  // the header is written with %g and therefore carries only about six significant digits
  const float scale = fabs(from_nep_in) + fabs(from_file) + 1.0e-10f;
  if (fabs(from_nep_in - from_file) > 1.0e-5f * scale) {
    mismatches.push_back(
      name + ": nep.in gives " + float_to_string(from_nep_in) + source_of_value(is_set) + ", " +
      filename + " gives " + float_to_string(from_file) + ".");
  }
}

void Parameters::compare_with_nep_txt(
  const std::string& filename, std::vector<std::string>& mismatches)
{
  NepTxtHeader header;
  std::string error;
  if (!read_nep_txt_header(filename, header, error)) {
    mismatches.push_back(error);
    return;
  }
  number_of_nep_txt_header_lines = header.number_of_header_lines;

  compare_int("version", version, is_version_set, header.version, filename, mismatches);
  compare_int("model_type", train_mode, is_train_mode_set, header.train_mode, filename, mismatches);
  compare_int(
    "charge_mode", charge_mode, is_charge_mode_set, header.charge_mode, filename, mismatches);
  compare_int(
    "type (number of types)", num_types, is_type_set, header.num_types, filename, mismatches);
  if (num_types == header.num_types && elements != header.elements) {
    std::string from_nep_in, from_file;
    for (int n = 0; n < num_types; ++n) {
      from_nep_in += (n == 0 ? "" : " ") + elements[n];
      from_file += (n == 0 ? "" : " ") + header.elements[n];
    }
    mismatches.push_back(
      "type (elements and their order): nep.in gives " + from_nep_in + ", " + filename + " gives " +
      from_file + ".");
  }

  if (enable_zbl != header.enable_zbl) {
    mismatches.push_back(
      std::string("zbl: nep.in has ZBL ") + (enable_zbl ? "enabled" : "disabled") + ", " +
      filename + " has it " + (header.enable_zbl ? "enabled" : "disabled") + ".");
  } else if (enable_zbl) {
    if (flexible_zbl != header.flexible_zbl) {
      mismatches.push_back(
        std::string("zbl: nep.in requests a ") + (flexible_zbl ? "flexible" : "universal") +
        " ZBL potential, " + filename + " holds a " +
        (header.flexible_zbl ? "flexible" : "universal") + " one.");
    } else if (!flexible_zbl) {
      compare_float(
        "zbl (inner cutoff)", zbl_rc_inner, is_zbl_set, header.zbl_rc_inner, filename, mismatches);
      compare_float(
        "zbl (outer cutoff)", zbl_rc_outer, is_zbl_set, header.zbl_rc_outer, filename, mismatches);
      compare_float(
        "use_typewise_cutoff_zbl",
        typewise_cutoff_zbl_factor,
        is_use_typewise_cutoff_zbl_set,
        header.typewise_cutoff_zbl_factor,
        filename,
        mismatches);
    }
  }

  // a single cutoff in the file applies to every type
  for (int n = 0; n < num_types && n < header.num_types; ++n) {
    const int m = header.has_multiple_cutoffs ? n : 0;
    const std::string suffix = (num_types > 1) ? " for " + elements[n] : "";
    compare_float(
      "cutoff (radial)" + suffix,
      rc_radial[n],
      is_cutoff_set,
      header.rc_radial[m],
      filename,
      mismatches);
    compare_float(
      "cutoff (angular)" + suffix,
      rc_angular[n],
      is_cutoff_set,
      header.rc_angular[m],
      filename,
      mismatches);
  }

  compare_int(
    "n_max_radial", n_max_radial, is_n_max_set, header.n_max_radial, filename, mismatches);
  compare_int(
    "n_max_angular", n_max_angular, is_n_max_set, header.n_max_angular, filename, mismatches);
  compare_int(
    "basis_size_radial",
    basis_size_radial,
    is_basis_size_set,
    header.basis_size_radial,
    filename,
    mismatches);
  compare_int(
    "basis_size_angular",
    basis_size_angular,
    is_basis_size_set,
    header.basis_size_angular,
    filename,
    mismatches);

  compare_int("L_max", L_max, is_l_max_set, header.L_max, filename, mismatches);
  // the 4-body flag is written as 2 or 0 rather than as 1 or 0
  compare_int(
    "L_max_4body", has_q_222 ? 2 : 0, is_l_max_set, header.has_q_222, filename, mismatches);
  compare_int("L_max_5body", has_q_1111, is_l_max_set, header.has_q_1111, filename, mismatches);
  compare_int("has_q_112", has_q_112, is_l_max_set, header.has_q_112, filename, mismatches);
  compare_int("has_q_123", has_q_123, is_l_max_set, header.has_q_123, filename, mismatches);
  compare_int("has_q_233", has_q_233, is_l_max_set, header.has_q_233, filename, mismatches);
  compare_int("has_q_134", has_q_134, is_l_max_set, header.has_q_134, filename, mismatches);

  compare_int("neuron", num_neurons1, is_neuron_set, header.num_neurons1, filename, mismatches);
  compare_int(
    "neuron (second hidden layer)",
    (num_hidden_layers == 2) ? num_neurons2 : 0,
    is_neuron_set,
    header.num_neurons2,
    filename,
    mismatches);
}

void Parameters::check_nep_txt(const std::string& filename, const bool fatal, const char* remedy)
{
  std::vector<std::string> mismatches;
  compare_with_nep_txt(filename, mismatches);
  if (mismatches.empty()) {
    return;
  }
  printf("The model in nep.in is inconsistent with %s:\n", filename.c_str());
  for (const auto& mismatch : mismatches) {
    printf("    %s\n", mismatch.c_str());
  }
  printf("%s\n", remedy);
  if (fatal) {
    PRINT_INPUT_ERROR(("nep.in is inconsistent with " + filename + ".").c_str());
  }
}

void Parameters::check_nep_restart()
{
  std::ifstream input("nep.restart");
  if (!input.is_open()) {
    return;
  }
  // nep.restart has no header, so the number of rows is the only thing that can be checked
  int number_of_rows = 0;
  std::vector<std::string> tokens = get_tokens(input);
  while (tokens.size() >= 2) {
    ++number_of_rows;
    tokens = get_tokens(input);
  }
  input.close();

  if (number_of_rows != number_of_variables) {
    printf(
      "nep.restart holds %d rows but nep.in implies %d parameters.\n",
      number_of_rows,
      number_of_variables);
    printf(
      "The difference of %d rows means that one or more model hyperparameters in nep.in differ "
      "from the model that wrote nep.restart.\n",
      (number_of_rows > number_of_variables) ? number_of_rows - number_of_variables
                                             : number_of_variables - number_of_rows);
    printf("Correct nep.in, or remove nep.restart to start a new training run.\n");
    PRINT_INPUT_ERROR("nep.restart does not match the model implied by nep.in.");
  }
}

static bool does_file_exist(const char* filename)
{
  std::ifstream input(filename);
  return input.is_open();
}

void Parameters::check_existing_model()
{
  if (!is_type_set) {
    return; // report_inputs reports the missing type keyword
  }

  if (fine_tune) {
    check_nep_txt(fine_tune_nep_txt, true, "Correct nep.in to match the foundation model.");
    return; // the restart file to fine-tune from is named by the fine_tune keyword
  }

  if (import_q_scaler) {
    check_nep_txt("nep.txt", true, "Correct nep.in, or switch off import_q_scaler.");
  } else if (train_mode == 0 && does_file_exist("nep.txt")) {
    // nep.txt is an input when predicting or when there is a nep.restart to resume from,
    // and merely a stale output otherwise
    if (prediction == 1) {
      check_nep_txt("nep.txt", true, "Correct nep.in to match the model to predict with.");
    } else if (does_file_exist("nep.restart")) {
      check_nep_txt(
        "nep.txt", true, "Correct nep.in, or remove nep.restart to start a new training run.");
    } else {
      check_nep_txt("nep.txt", false, "Warning: nep.txt will be overwritten by this training run.");
    }
  }

  // nep.restart is read only when resuming a training run, not when predicting
  if (train_mode == 0 && prediction == 0 && does_file_exist("nep.restart")) {
    check_nep_restart();
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

  if (is_charge_mode_set) {
    if (charge_mode == 1) {
      printf("    (input)   use NEP-Charge and include both real-space and k-space.\n");
    } else if (charge_mode == 2) {
      printf("    (input)   use NEP-Charge and include k-space only.\n");
    }
    printf("        lambda_q = %g.\n", lambda_q);
    printf("        lambda_z = %g.\n", lambda_z);

    if (has_multiple_cutoffs) {
      PRINT_INPUT_ERROR("Can only use uniform cutoff for qNEP.");
    }
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

  if (is_atomic_v_set) {
    printf("    (input)   atomic_v = %d.\n", atomic_v);
  } else {
    printf("    (default) atomic_v = %d.\n", atomic_v);
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

  // some calcuated parameters:
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
    number_of_variables_ann * (train_mode == 2 ? 2 : 1));
  printf(
    "    number of descriptor parameters to be optimized = %d.\n", number_of_variables_descriptor);
  printf("    total number of parameters to be optimized = %d.\n", number_of_variables);
}

void Parameters::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const char* param[179];
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
  } else if (strcmp(param[0], "lambda_q") == 0) {
    parse_lambda_q(param, num_param);
  } else if (strcmp(param[0], "lambda_z") == 0) {
    parse_lambda_z(param, num_param);
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
  } else if (strcmp(param[0], "atomic_v") == 0) {
    parse_atomic_v(param, num_param);
  } else if (strcmp(param[0], "use_typewise_cutoff_zbl") == 0) {
    parse_use_typewise_cutoff_zbl(param, num_param);
  } else if (strcmp(param[0], "output_descriptor") == 0) {
    parse_output_descriptor(param, num_param);
  } else if (strcmp(param[0], "charge_mode") == 0) {
    parse_charge_mode(param, num_param);
  } else if (strcmp(param[0], "fine_tune") == 0) {
    parse_fine_tune(param, num_param);
  } else if (strcmp(param[0], "save_potential") == 0) {
    parse_save_potential(param, num_param);
  } else if (strcmp(param[0], "output_interval") == 0) {
    parse_output_interval(param, num_param);
  } else if (strcmp(param[0], "import_q_scaler") == 0) {
    parse_import_q_scaler(param, num_param);
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
  if (version != 4) {
    PRINT_INPUT_ERROR("version should = 4.");
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
  } else if (zbl_rc_outer > 4.0f) {
    PRINT_INPUT_ERROR("outer cutoff for ZBL should <= 4.0 A.");
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
  } else if (n_max_radial > 12) {
    PRINT_INPUT_ERROR("n_max_radial should <= 12.");
  }
  if (n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max_angular should >= 0.");
  } else if (n_max_angular > 8) {
    PRINT_INPUT_ERROR("n_max_angular should <= 8.");
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
  } else if (basis_size_radial > 16) {
    PRINT_INPUT_ERROR("basis_size_radial should <= 16.");
  }
  if (basis_size_angular < 0) {
    PRINT_INPUT_ERROR("basis_size_angular should >= 0.");
  } else if (basis_size_angular > 12) {
    PRINT_INPUT_ERROR("basis_size_angular should <= 12.");
  }
}

void Parameters::parse_l_max(const char** param, int num_param)
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

void Parameters::parse_neuron(const char** param, int num_param)
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
    if (charge_mode != 0) {
      PRINT_INPUT_ERROR("Can only use one hidden layer for qNEP.");
    }

    if (!is_valid_int(param[2], &num_neurons2)) {
      PRINT_INPUT_ERROR("number of neurons2 in the output layer should be an integer.\n");
    }
    if (num_neurons2 < 0) {
      PRINT_INPUT_ERROR("number of neurons2 in the output layer should >= 0.");
    } else if (num_neurons2 > 120) {
      PRINT_INPUT_ERROR("number of neurons2 in the output layer should <= 120.");
    }
    num_hidden_layers = 2;

    if (num_neurons2 == 0) {
      num_hidden_layers = 1;
    }
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

void Parameters::parse_lambda_q(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_q should have 1 parameter.\n");
  }

  double lambda_q_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_q_tmp)) {
    PRINT_INPUT_ERROR("Charge loss weight should be a number.\n");
  }
  lambda_q = lambda_q_tmp;

  if (lambda_q < 0.0f) {
    PRINT_INPUT_ERROR("Charge loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_z(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_z should have 1 parameter.\n");
  }

  double lambda_z_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_z_tmp)) {
    PRINT_INPUT_ERROR("BEC loss weight should be a number.\n");
  }
  lambda_z = lambda_z_tmp;

  if (lambda_z < 0.0f) {
    PRINT_INPUT_ERROR("BEC loss weight should >= 0.");
  }
}

void Parameters::parse_atomic_v(const char** param, int num_param)
{
  is_atomic_v_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("atomic_v should have 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &atomic_v)) {
    PRINT_INPUT_ERROR("atomic_v should be an integer.\n");
  }

  if (atomic_v != 0 && atomic_v != 1) {
    PRINT_INPUT_ERROR("atomic_v should = 0 or 1.");
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
  CHECK(gpuGetDeviceCount(&deviceCount));
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

void Parameters::parse_use_typewise_cutoff_zbl(const char** param, int num_param)
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

void Parameters::parse_output_descriptor(const char** param, int num_param)
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

void Parameters::parse_charge_mode(const char** param, int num_param)
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

  if (num_hidden_layers == 2) {
    PRINT_INPUT_ERROR("Can only use one hidden layer for qNEP.");
  }
}

void Parameters::parse_fine_tune(const char** param, int num_param)
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

void Parameters::parse_save_potential(const char** param, int num_param)
{
  is_save_potential_set = true;

  if (num_param != 4) {
    PRINT_INPUT_ERROR("save_potential should have 3 parameters.\n");
  }
  if (!is_valid_int(param[1], &save_potential)) {
    PRINT_INPUT_ERROR("save_potential interval should be an integer.\n");
  }
  if (save_potential < 0) {
    PRINT_INPUT_ERROR("save_potential interval should be >= 0.");
  }
  if (!is_valid_int(param[2], &save_potential_format)) {
    PRINT_INPUT_ERROR("save_potential format should be an integer.\n");
  }
  if (save_potential_format != 0 && save_potential_format != 1) {
    PRINT_INPUT_ERROR("save_potential format should be 0 or 1.");
  }  
  if (!is_valid_int(param[3], &save_potential_restart)) {
    PRINT_INPUT_ERROR("save_potential save restart should be an integer.\n");
  }
  if (save_potential_restart != 0 && save_potential_restart != 1) {
    PRINT_INPUT_ERROR("save_potential save restart should be 0 or 1.");
  }
}

void Parameters::parse_output_interval(const char** param, int num_param)
{
  is_output_interval_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("output_interval should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval should be an integer.\n");
  }
  if (output_interval <= 0) {
    PRINT_INPUT_ERROR("output_interval should be > 0.");
  }
}

void Parameters::parse_import_q_scaler(const char** param, int num_param)
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