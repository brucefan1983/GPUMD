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

#include "nep_io.cuh"
#include "nep_elements.cuh"
#include "nep_parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_parameters.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

struct NepTxtHeader {
  std::string model_token;
  int version;
  bool enable_zbl;
  bool flexible_zbl;
  bool use_typewise_cutoff_zbl;
  float zbl_rc_inner;
  float zbl_rc_outer;
  float typewise_cutoff_zbl_factor;
  int num_types;
  std::vector<std::string> elements;
  bool has_multiple_cutoffs;
  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
  int n_max_radial;
  int n_max_angular;
  int basis_size_radial;
  int basis_size_angular;
  int L_max;
  int has_q_222; // as written to nep.txt, that is 2 or 0
  int has_q_1111;
  int has_q_112;
  int has_q_123;
  int has_q_233;
  int has_q_134;
  int num_neurons1;
  int num_neurons2;
  int number_of_header_lines; // 6 without a zbl line, 7 with one
};

// The shared codec reads the NEP prefix and optional ZBL header. The owning
// model supplies the complete expected file tag; suffixes are not dispatched here.
static bool parse_model_token(const std::string& token, NepTxtHeader& header)
{
  if (token.size() < 4 || token.compare(0, 3, "nep") != 0 ||
      !isdigit(static_cast<unsigned char>(token[3]))) {
    return false;
  }
  header.version = token[3] - '0';
  header.enable_zbl = token.compare(4, 4, "_zbl") == 0;
  return true;
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

static bool read_nep_txt_header(const std::string& filename, NepTxtHeader& header, std::string& error)
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

static void nep_parameters_to_channel_major(const NEP_Parameters& para, float* parameters)
{
  const int descriptor_offset = para.number_of_variables_ann * para.ann_count;
#ifdef USE_CJ
  const int num_channels = para.num_types;
#else
  const int num_channels = para.num_types * para.num_types;
#endif
  descriptor_parameters_to_channel_major(
    parameters,
    descriptor_offset,
    num_channels,
    para.n_max_radial,
    para.n_max_angular,
    para.basis_size_radial,
    para.basis_size_angular);
}

static void nep_parameters_to_basis_major(const NEP_Parameters& para, float* parameters)
{
  const int descriptor_offset = para.number_of_variables_ann * para.ann_count;
#ifdef USE_CJ
  const int num_channels = para.num_types;
#else
  const int num_channels = para.num_types * para.num_types;
#endif
  descriptor_parameters_to_basis_major(
    parameters,
    descriptor_offset,
    num_channels,
    para.n_max_radial,
    para.n_max_angular,
    para.basis_size_radial,
    para.basis_size_angular);
}

static void read_nep_txt(const std::string& filename, NEP_Parameters& para, float* parameters)
{
  NepTxtHeader header;
  std::string error;
  if (!read_nep_txt_header(filename, header, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (header.model_token != para.file_tag) {
    PRINT_INPUT_ERROR(("Expected " + para.file_tag + " in " + filename +
      ", found " + header.model_token + ".").c_str());
  }
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR(("Failed to open " + filename + ".").c_str());
  }
  std::vector<std::string> tokens;
  for (int n = 0; n < header.number_of_header_lines; ++n) {
    tokens = get_tokens(input);
  }
  for (int n = 0; n < para.number_of_variables; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_parameters_to_channel_major(para, parameters);
  for (int d = 0; d < para.dim; ++d) {
    tokens = get_tokens(input);
    para.q_scaler_cpu[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
}

void read_nep_model(
  const std::string& filename, NEP_Parameters& para, float* parameters)
{
  read_nep_txt(filename, para, parameters);
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_from_host(para.q_scaler_cpu.data());
}

bool read_nep_restart(
  const char* filename,
  const NEP_Parameters& para,
  std::vector<float>& mu,
  std::vector<float>& sigma)
{
  FILE* fid_restart = fopen(filename, "r");
  if (fid_restart == NULL) {
    return false;
  }
  for (int n = 0; n < para.number_of_variables; ++n) {
    int count = fscanf(fid_restart, "%f%f", &mu[n], &sigma[n]);
    PRINT_SCANF_ERROR(count, 2, "Reading error for nep.restart.");
  }
  nep_parameters_to_channel_major(para, mu.data());
  nep_parameters_to_channel_major(para, sigma.data());
  fclose(fid_restart);
  return true;
}

void read_nep_foundation_restart(
  const std::string& filename, std::vector<float>& mu, std::vector<float>& sigma)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Cannot open the foundation model file " << filename << std::endl;
    exit(1);
  }
  std::vector<std::string> tokens;

  for (size_t n = 0; n < mu.size(); ++n) {
    tokens = get_tokens(input);
    if (tokens.size() != 2) {
      std::cout << "Foundation model file should have two columns.\n";
      exit(1);
    }
    mu[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    sigma[n] = get_double_from_token(tokens[1], __FILE__, __LINE__);
  }
  input.close();
}

void write_nep_restart(
  const char* filename,
  const NEP_Parameters& para,
  const std::vector<float>& mu,
  const std::vector<float>& sigma)
{
  std::vector<float> mu_file = mu;
  std::vector<float> sigma_file = sigma;
  nep_parameters_to_basis_major(para, mu_file.data());
  nep_parameters_to_basis_major(para, sigma_file.data());
  FILE* fid_restart = my_fopen(filename, "w");
  for (int n = 0; n < para.number_of_variables; ++n) {
    fprintf(fid_restart, "%15.7e %15.7e\n", mu_file[n], sigma_file[n]);
  }
  fclose(fid_restart);
}

static std::string source_of_value(const bool is_set) { return is_set ? " (input)" : " (default)"; }

static std::string float_to_string(const float value)
{
  char buffer[32];
  snprintf(buffer, sizeof(buffer), "%g", value);
  return std::string(buffer);
}

void read_nep_q_scaler(NEP_Parameters& para)
{
  if (para.fine_tune) {
    std::ifstream input(para.fine_tune_nep_txt);
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open foundation model file.");
    }
    std::vector<std::string> tokens;
    const int NUM89 = 89;
    const int num_ann =
      NUM89 * para.number_of_variables_ann_1 + para.global_parameters;
#ifdef USE_CJ
    const int num_cnk_radial = NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
    const int num_cnk_angular = NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
#else
    const int num_cnk_radial = NUM89 * NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
    const int num_cnk_angular = NUM89 * NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
#endif
    const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
    for (int n = 0; n < num_tot + para.number_of_nep_txt_header_lines; ++n) {
      tokens = get_tokens(input); // not used
    }
    for (int n = 0; n < para.q_scaler_cpu.size(); ++n) {
      tokens = get_tokens(input);
      para.q_scaler_cpu[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    input.close();
  } else if (para.import_q_scaler) {
    std::ifstream input("nep.txt");
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open nep.txt for q_scaler import.");
    }
    std::vector<std::string> tokens;
    for (int n = 0; n < para.number_of_nep_txt_header_lines + para.number_of_variables; ++n) {
      tokens = get_tokens(input); // not used
    }
    for (int n = 0; n < para.q_scaler_cpu.size(); ++n) {
      tokens = get_tokens(input);
      para.q_scaler_cpu[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    input.close();
  }
}

static void write_nep_txt(
  FILE* fid_nep,
  const NEP_Parameters& para,
  const float* elite,
  const int max_NN_radial,
  const int max_NN_angular)
{
  fprintf(fid_nep, "%s %d ", para.file_tag.c_str(), para.num_types);

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_nep, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_nep, "\n");
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_nep, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(fid_nep, "zbl %g %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(fid_nep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }

  fprintf(fid_nep, "cutoff %g %g ", para.rc_radial[0], para.rc_angular[0]);
  if (para.has_multiple_cutoffs) {
    for (int n = 1; n < para.num_types; ++n) {
      fprintf(fid_nep, "%g %g ", para.rc_radial[n], para.rc_angular[n]);
    }
  }
  fprintf(fid_nep, "%d %d\n", max_NN_radial, max_NN_angular);

  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_nep, "l_max %d %d %d ", para.L_max, (para.has_q_222 ? 2 : 0), para.has_q_1111);
  if (para.has_q_112 || para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_112);
  }
  if (para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_123);
  }
  if (para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_233);
  }
  if (para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_134);
  }
  fprintf(fid_nep, "\n");

  if (para.num_hidden_layers == 2) {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, para.num_neurons2);
  } else {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  }

  std::vector<float> parameters_file(elite, elite + para.number_of_variables);
  nep_parameters_to_basis_major(para, parameters_file.data());
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", parameters_file[m]);
  }
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_nep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_nep, "%15.7e\n", para.zbl_para[d]);
    }
  }
}

void write_nep_model(
  FILE* file, NEP_Parameters& para, const float* parameters, int max_radial, int max_angular)
{
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  write_nep_txt(file, para, parameters, max_radial, max_angular);
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

static void compare_with_nep_txt_fine_tune(
  NEP_Parameters& para, const std::string& filename, std::vector<std::string>& mismatches)
{
  NepTxtHeader header;
  std::string error;
  if (!read_nep_txt_header(filename, header, error)) {
    mismatches.push_back(error);
    return;
  }
  para.number_of_nep_txt_header_lines = header.number_of_header_lines;

  std::vector<std::string> elements_nep89;
  for (int n = 0; n < NUM_ELEMENTS; ++n) {
    if (NEP_ELEMENTS[n] != "Po" && NEP_ELEMENTS[n] != "At" && NEP_ELEMENTS[n] != "Rn" &&
        NEP_ELEMENTS[n] != "Fr" && NEP_ELEMENTS[n] != "Ra") {
      elements_nep89.push_back(NEP_ELEMENTS[n]);
    }
  }
  if (header.num_types != int(elements_nep89.size()) || header.elements != elements_nep89) {
    mismatches.push_back(
      "type: the fine-tune foundation model must contain the canonical 89 elements in the "
      "expected order.");
  }

  std::vector<int> foundation_type_index(para.num_types, -1);
  for (int n = 0; n < para.num_types; ++n) {
    const auto it = std::find(header.elements.begin(), header.elements.end(), para.elements[n]);
    if (it == header.elements.end()) {
      mismatches.push_back(
        "type: element " + para.elements[n] + " is not available in the foundation model.");
    } else {
      foundation_type_index[n] = int(it - header.elements.begin());
    }
  }

  if (header.model_token != para.file_tag) {
    mismatches.push_back("model: nep.in implies " + para.file_tag + ", " + filename +
      " holds " + header.model_token + ".");
  }
  compare_int(
    "version", para.version, para.is_version_set, header.version, filename, mismatches);

  if (para.enable_zbl != header.enable_zbl) {
    mismatches.push_back(
      std::string("zbl: nep.in has ZBL ") + (para.enable_zbl ? "enabled" : "disabled") +
      ", " + filename + " has it " + (header.enable_zbl ? "enabled" : "disabled") + ".");
  } else if (para.enable_zbl) {
    if (para.flexible_zbl != header.flexible_zbl) {
      mismatches.push_back(
        std::string("zbl: nep.in requests a ") +
        (para.flexible_zbl ? "flexible" : "universal") + " ZBL potential, " + filename +
        " holds a " + (header.flexible_zbl ? "flexible" : "universal") + " one.");
    } else if (!para.flexible_zbl) {
      compare_float(
        "zbl (inner cutoff)",
        para.zbl_rc_inner,
        para.is_zbl_set,
        header.zbl_rc_inner,
        filename,
        mismatches);
      compare_float(
        "zbl (outer cutoff)",
        para.zbl_rc_outer,
        para.is_zbl_set,
        header.zbl_rc_outer,
        filename,
        mismatches);
      compare_float(
        "use_typewise_cutoff_zbl",
        para.typewise_cutoff_zbl_factor,
        para.is_use_typewise_cutoff_zbl_set,
        header.typewise_cutoff_zbl_factor,
        filename,
        mismatches);
    }
  }

  for (int n = 0; n < para.num_types; ++n) {
    if (foundation_type_index[n] < 0) {
      continue;
    }
    const int m = header.has_multiple_cutoffs ? foundation_type_index[n] : 0;
    const std::string suffix = (para.num_types > 1) ? " for " + para.elements[n] : "";
    compare_float(
      "cutoff (radial)" + suffix,
      para.rc_radial[n],
      para.is_cutoff_set,
      header.rc_radial[m],
      filename,
      mismatches);
    compare_float(
      "cutoff (angular)" + suffix,
      para.rc_angular[n],
      para.is_cutoff_set,
      header.rc_angular[m],
      filename,
      mismatches);
  }

  compare_int(
    "n_max_radial",
    para.n_max_radial,
    para.is_n_max_set,
    header.n_max_radial,
    filename,
    mismatches);
  compare_int(
    "n_max_angular",
    para.n_max_angular,
    para.is_n_max_set,
    header.n_max_angular,
    filename,
    mismatches);
  compare_int(
    "basis_size_radial",
    para.basis_size_radial,
    para.is_basis_size_set,
    header.basis_size_radial,
    filename,
    mismatches);
  compare_int(
    "basis_size_angular",
    para.basis_size_angular,
    para.is_basis_size_set,
    header.basis_size_angular,
    filename,
    mismatches);

  compare_int(
    "L_max", para.L_max, para.is_l_max_set, header.L_max, filename, mismatches);
  compare_int(
    "L_max_4body",
    para.has_q_222 ? 2 : 0,
    para.is_l_max_set,
    header.has_q_222,
    filename,
    mismatches);
  compare_int(
    "L_max_5body",
    para.has_q_1111,
    para.is_l_max_set,
    header.has_q_1111,
    filename,
    mismatches);
  compare_int(
    "has_q_112",
    para.has_q_112,
    para.is_l_max_set,
    header.has_q_112,
    filename,
    mismatches);
  compare_int(
    "has_q_123",
    para.has_q_123,
    para.is_l_max_set,
    header.has_q_123,
    filename,
    mismatches);
  compare_int(
    "has_q_233",
    para.has_q_233,
    para.is_l_max_set,
    header.has_q_233,
    filename,
    mismatches);
  compare_int(
    "has_q_134",
    para.has_q_134,
    para.is_l_max_set,
    header.has_q_134,
    filename,
    mismatches);

  compare_int(
    "neuron",
    para.num_neurons1,
    para.is_neuron_set,
    header.num_neurons1,
    filename,
    mismatches);
  compare_int(
    "neuron (second hidden layer)",
    (para.num_hidden_layers == 2) ? para.num_neurons2 : 0,
    para.is_neuron_set,
    header.num_neurons2,
    filename,
    mismatches);
}

static void compare_with_nep_txt(
  NEP_Parameters& para, const std::string& filename, std::vector<std::string>& mismatches)
{
  NepTxtHeader header;
  std::string error;
  if (!read_nep_txt_header(filename, header, error)) {
    mismatches.push_back(error);
    return;
  }
  para.number_of_nep_txt_header_lines = header.number_of_header_lines;

  if (header.model_token != para.file_tag) {
    mismatches.push_back("model: nep.in implies " + para.file_tag + ", " + filename +
      " holds " + header.model_token + ".");
  }
  compare_int("version", para.version, para.is_version_set, header.version, filename, mismatches);
  compare_int(
    "type (number of types)", para.num_types, para.is_type_set, header.num_types, filename, mismatches);
  if (para.num_types == header.num_types && para.elements != header.elements) {
    std::string from_nep_in, from_file;
    for (int n = 0; n < para.num_types; ++n) {
      from_nep_in += (n == 0 ? "" : " ") + para.elements[n];
      from_file += (n == 0 ? "" : " ") + header.elements[n];
    }
    mismatches.push_back(
      "type (elements and their order): nep.in gives " + from_nep_in + ", " + filename + " gives " +
      from_file + ".");
  }

  if (para.enable_zbl != header.enable_zbl) {
    mismatches.push_back(
      std::string("zbl: nep.in has ZBL ") + (para.enable_zbl ? "enabled" : "disabled") + ", " +
      filename + " has it " + (header.enable_zbl ? "enabled" : "disabled") + ".");
  } else if (para.enable_zbl) {
    if (para.flexible_zbl != header.flexible_zbl) {
      mismatches.push_back(
        std::string("zbl: nep.in requests a ") + (para.flexible_zbl ? "flexible" : "universal") +
        " ZBL potential, " + filename + " holds a " +
        (header.flexible_zbl ? "flexible" : "universal") + " one.");
    } else if (!para.flexible_zbl) {
      compare_float(
        "zbl (inner cutoff)",
        para.zbl_rc_inner,
        para.is_zbl_set,
        header.zbl_rc_inner,
        filename,
        mismatches);
      compare_float(
        "zbl (outer cutoff)",
        para.zbl_rc_outer,
        para.is_zbl_set,
        header.zbl_rc_outer,
        filename,
        mismatches);
      compare_float(
        "use_typewise_cutoff_zbl",
        para.typewise_cutoff_zbl_factor,
        para.is_use_typewise_cutoff_zbl_set,
        header.typewise_cutoff_zbl_factor,
        filename,
        mismatches);
    }
  }

  // a single cutoff in the file applies to every type
  for (int n = 0; n < para.num_types && n < header.num_types; ++n) {
    const int m = header.has_multiple_cutoffs ? n : 0;
    const std::string suffix = (para.num_types > 1) ? " for " + para.elements[n] : "";
    compare_float(
      "cutoff (radial)" + suffix,
      para.rc_radial[n],
      para.is_cutoff_set,
      header.rc_radial[m],
      filename,
      mismatches);
    compare_float(
      "cutoff (angular)" + suffix,
      para.rc_angular[n],
      para.is_cutoff_set,
      header.rc_angular[m],
      filename,
      mismatches);
  }

  compare_int(
    "n_max_radial", para.n_max_radial, para.is_n_max_set, header.n_max_radial, filename, mismatches);
  compare_int(
    "n_max_angular", para.n_max_angular, para.is_n_max_set, header.n_max_angular, filename, mismatches);
  compare_int(
    "basis_size_radial",
    para.basis_size_radial,
    para.is_basis_size_set,
    header.basis_size_radial,
    filename,
    mismatches);
  compare_int(
    "basis_size_angular",
    para.basis_size_angular,
    para.is_basis_size_set,
    header.basis_size_angular,
    filename,
    mismatches);

  compare_int("L_max", para.L_max, para.is_l_max_set, header.L_max, filename, mismatches);
  // the 4-body flag is written as 2 or 0 rather than as 1 or 0
  compare_int(
    "L_max_4body", para.has_q_222 ? 2 : 0, para.is_l_max_set, header.has_q_222, filename, mismatches);
  compare_int("L_max_5body", para.has_q_1111, para.is_l_max_set, header.has_q_1111, filename, mismatches);
  compare_int("has_q_112", para.has_q_112, para.is_l_max_set, header.has_q_112, filename, mismatches);
  compare_int("has_q_123", para.has_q_123, para.is_l_max_set, header.has_q_123, filename, mismatches);
  compare_int("has_q_233", para.has_q_233, para.is_l_max_set, header.has_q_233, filename, mismatches);
  compare_int("has_q_134", para.has_q_134, para.is_l_max_set, header.has_q_134, filename, mismatches);

  compare_int("neuron", para.num_neurons1, para.is_neuron_set, header.num_neurons1, filename, mismatches);
  compare_int(
    "neuron (second hidden layer)",
    (para.num_hidden_layers == 2) ? para.num_neurons2 : 0,
    para.is_neuron_set,
    header.num_neurons2,
    filename,
    mismatches);
}

static void check_nep_txt(
  NEP_Parameters& para, const std::string& filename, const char* remedy)
{
  std::vector<std::string> mismatches;
  compare_with_nep_txt(para, filename, mismatches);
  if (mismatches.empty()) {
    return;
  }
  printf("The model in nep.in is inconsistent with %s:\n", filename.c_str());
  for (const auto& mismatch : mismatches) {
    printf("    %s\n", mismatch.c_str());
  }
  printf("%s\n", remedy);
  PRINT_INPUT_ERROR(("nep.in is inconsistent with " + filename + ".").c_str());
}

static void check_nep_restart(NEP_Parameters& para)
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

  if (number_of_rows != para.number_of_variables) {
    printf(
      "nep.restart holds %d rows but nep.in implies %d parameters.\n",
      number_of_rows,
      para.number_of_variables);
    printf(
      "The difference of %d rows means that one or more model hyperparameters in nep.in differ "
      "from the model that wrote nep.restart.\n",
      (number_of_rows > para.number_of_variables) ? number_of_rows - para.number_of_variables
                                               : para.number_of_variables - number_of_rows);
    printf("Correct nep.in, or remove nep.restart to start a new training run.\n");
    PRINT_INPUT_ERROR("nep.restart does not match the model implied by nep.in.");
  }
}

static bool does_file_exist(const char* filename)
{
  std::ifstream input(filename);
  return input.is_open();
}

void check_existing_nep_model(NEP_Parameters& para)
{
  if (para.fine_tune) {
    std::vector<std::string> mismatches;
    compare_with_nep_txt_fine_tune(para, para.fine_tune_nep_txt, mismatches);
    if (!mismatches.empty()) {
      printf(
        "The model in nep.in is inconsistent with the fine-tune foundation model %s:\n",
        para.fine_tune_nep_txt.c_str());
      for (const auto& mismatch : mismatches) {
        printf("    %s\n", mismatch.c_str());
      }
      printf("Correct nep.in or use the matching fine-tune template.\n");
      PRINT_INPUT_ERROR(
        ("nep.in is inconsistent with " + para.fine_tune_nep_txt + ".").c_str());
    }
    return; // the restart file to fine-tune from is named by the fine_tune keyword
  }

  if (para.import_q_scaler) {
    check_nep_txt(para, "nep.txt", "Correct nep.in, or switch off import_q_scaler.");
  } else if (
    (para.prediction == 1 || does_file_exist("nep.restart")) && does_file_exist("nep.txt")) {
    if (para.prediction == 1) {
      check_nep_txt(para, "nep.txt", "Correct nep.in to match the model to predict with.");
    } else {
      check_nep_txt(
        para, "nep.txt", "Correct nep.in, or remove nep.restart to start a new training run.");
    }
  }

  // nep.restart is read only when resuming a training run, not when predicting
  if (para.prediction == 0 && does_file_exist("nep.restart")) {
    check_nep_restart(para);
  }
}
