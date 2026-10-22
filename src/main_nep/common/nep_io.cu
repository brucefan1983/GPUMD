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
#include "utilities/nep_model_file.cuh"
#include "utilities/nep_parameters.cuh"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <fstream>

static bool read_nep_file(
  const std::string& filename,
  NEP_Model_File& model,
  std::string& error,
  std::vector<float>* sigma = nullptr)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    error = "Failed to open " + filename + ".";
    return false;
  }
  std::string token;
  input >> token;
  for (char& c : token) {
    c = std::tolower(static_cast<unsigned char>(c));
  }
  if (token != "nep") {
    error = filename + ": the nep executable accepts only the new NEP format. "
                       "Legacy potential files remain supported by gpumd for MD.";
    return false;
  }
  input.clear();
  input.seekg(0);
  const bool ok =
    sigma == nullptr ? model.read(input, error) : model.read_restart(input, *sigma, error);
  if (!ok) {
    error = filename + ": " + error;
  }
  return ok;
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

static std::vector<float> get_nep_parameters(
  const NEP_Model_File& model, const std::vector<float>& values)
{
#ifdef USE_CJ
  const int ann_size = model.layout.number_of_variables_ann * model.layout.ann_count;
  const int num_types = model.num_types;
  const int num_basis = model.layout.number_of_variables_descriptor / (num_types * num_types);
  std::vector<float> parameters(ann_size + num_basis * num_types);
  std::copy(values.begin(), values.begin() + ann_size, parameters.begin());
  for (int nk = 0; nk < num_basis; ++nk) {
    for (int t2 = 0; t2 < num_types; ++t2) {
      const float value = values[ann_size + nk * num_types * num_types + t2];
      for (int t1 = 1; t1 < num_types; ++t1) {
        if (values[ann_size + (nk * num_types + t1) * num_types + t2] != value) {
          PRINT_INPUT_ERROR("This model requires type-pair descriptors. Rebuild without USE_CJ.");
        }
      }
      parameters[ann_size + nk * num_types + t2] = value;
    }
  }
  return parameters;
#else
  return values;
#endif
}

static void read_nep_parameters(
  const NEP_Model_File& model, const NEP_Parameters& para,
  const std::vector<float>& values, float* parameters)
{
  const std::vector<float> parsed = get_nep_parameters(model, values);
  if (
    model.name != para.model_name || model.elements != para.elements ||
    model.layout.dim != para.dim || model.layout.ann_count != para.ann_count ||
    model.layout.number_of_variables_ann != para.number_of_variables_ann ||
    model.layout.number_of_variables_ann_1 != para.number_of_variables_ann_1 ||
    model.layout.global_parameters != para.global_parameters ||
    parsed.size() != static_cast<std::size_t>(para.number_of_variables)) {
    PRINT_INPUT_ERROR("The NEP model parameter layout does not match nep.in.");
  }
  std::copy(parsed.begin(), parsed.end(), parameters);
  nep_parameters_to_channel_major(para, parameters);
}

void read_nep_model(
  const std::string& filename, NEP_Parameters& para, float* parameters)
{
  NEP_Model_File model;
  std::string error;
  if (!read_nep_file(filename, model, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  read_nep_parameters(model, para, model.parameters, parameters);
  para.q_scaler_cpu = model.q_scaler;
  para.energy_reference = model.energy_reference;
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_from_host(para.q_scaler_cpu.data());
}

static bool does_file_exist(const char* filename)
{
  errno = 0;
  std::ifstream input(filename);
  if (input.is_open()) {
    return true;
  }
  if (errno != ENOENT) {
    PRINT_INPUT_ERROR(("Failed to open " + std::string(filename) + ".").c_str());
  }
  return false;
}

bool read_nep_restart(
  const char* filename,
  const NEP_Parameters& para,
  std::vector<float>& mu,
  std::vector<float>& sigma)
{
  if (!does_file_exist(filename)) {
    return false;
  }
  NEP_Model_File model;
  std::vector<float> sigma_file;
  std::string error;
  if (!read_nep_file(filename, model, error, &sigma_file)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.energy_reference != para.energy_reference) {
    PRINT_INPUT_ERROR("The energy references in nep.restart have changed since initialization.");
  }
  read_nep_parameters(model, para, model.parameters, mu.data());
  read_nep_parameters(model, para, sigma_file, sigma.data());
  return true;
}

void read_nep_foundation_restart(
  const std::string& filename, std::vector<float>& mu, std::vector<float>& sigma)
{
  NEP_Model_File model;
  std::vector<float> sigma_file;
  std::string error;
  if (!read_nep_file(filename, model, error, &sigma_file)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  const std::vector<float> parsed_mu = get_nep_parameters(model, model.parameters);
  const std::vector<float> parsed_sigma = get_nep_parameters(model, sigma_file);
  if (parsed_mu.size() != mu.size() || parsed_sigma.size() != sigma.size()) {
    PRINT_INPUT_ERROR("The foundation restart parameter count does not match nep.in.");
  }
  mu = parsed_mu;
  sigma = parsed_sigma;
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
  if (!para.fine_tune && !para.import_q_scaler) {
    return;
  }
  const std::string filename = para.fine_tune ? para.fine_tune_nep_txt : "nep.txt";
  NEP_Model_File model;
  std::string error;
  if (!read_nep_file(filename, model, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.q_scaler.size() != para.q_scaler_cpu.size()) {
    PRINT_INPUT_ERROR("The q_scaler size does not match nep.in.");
  }
  para.q_scaler_cpu = model.q_scaler;
}

void read_nep_prediction_zbl(NEP_Parameters& para)
{
  NEP_Model_File model;
  std::string error;
  if (!read_nep_file("nep.txt", model, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  para.flexible_zbl = model.flexible_zbl;
  if (model.flexible_zbl) {
    if (model.zbl_parameters.size() > para.zbl_para.size()) {
      PRINT_INPUT_ERROR("The flexible ZBL model exceeds the supported parameter count.");
    }
    std::copy(model.zbl_parameters.begin(), model.zbl_parameters.end(), para.zbl_para.begin());
  }
}

static void write_nep_header(
  FILE* file, const NEP_Parameters& para, int max_radial, int max_angular)
{
  fprintf(file, "nep %d", para.num_types);
  for (const std::string& element : para.elements) {
    fprintf(file, " %s", element.c_str());
  }
  fprintf(file, "\nformat_version 1\nmodel %s\n", para.model_name.c_str());
  fprintf(file, "cutoff");
  const int count = para.has_multiple_cutoffs ? para.num_types : 1;
  for (int t = 0; t < count; ++t) {
    fprintf(file, " %.9g %.9g", para.rc_radial[t], para.rc_angular[t]);
  }
  fprintf(file, " %d %d\n", max_radial, max_angular);
  fprintf(file, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(file, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(
    file, "l_max %d %d %d %d %d %d %d\n", para.L_max, para.has_q_222 ? 2 : 0,
    para.has_q_1111, para.has_q_112, para.has_q_123, para.has_q_233, para.has_q_134);
  fprintf(file, "ann %d %d\n", para.num_neurons1,
    para.num_hidden_layers == 2 ? para.num_neurons2 : 0);
  if (!para.energy_reference.empty()) {
    fprintf(file, "energy_reference");
    for (const double reference : para.energy_reference) {
      fprintf(file, " %.17g", reference);
    }
    fprintf(file, "\n");
  }
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(file, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(file, "zbl %.9g %.9g %.9g\n",
        para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(file, "zbl %.9g %.9g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }
}

static void write_nep_block(FILE* file, const char* name, const float* values, int count)
{
  fprintf(file, "@ %s %d\n", name, count);
  for (int n = 0; n < count; ++n) {
    fprintf(file, "%.9g\n", values[n]);
  }
}

static void write_nep_parameters(
  FILE* file, const char* name, const NEP_Parameters& para, const float* parameters)
{
  std::vector<float> values(parameters, parameters + para.number_of_variables);
  nep_parameters_to_basis_major(para, values.data());
#ifdef USE_CJ
  const int ann_size = para.number_of_variables_ann * para.ann_count;
  const int num_basis = para.number_of_variables_descriptor / para.num_types;
  std::vector<float> expanded(ann_size + num_basis * para.num_types * para.num_types);
  std::copy(values.begin(), values.begin() + ann_size, expanded.begin());
  for (int nk = 0; nk < num_basis; ++nk) {
    for (int t1 = 0; t1 < para.num_types; ++t1) {
      for (int t2 = 0; t2 < para.num_types; ++t2) {
        expanded[ann_size + (nk * para.num_types + t1) * para.num_types + t2] =
          values[ann_size + nk * para.num_types + t2];
      }
    }
  }
  values.swap(expanded);
#endif
  write_nep_block(file, name, values.data(), values.size());
}

static void write_nep_zbl(FILE* file, const NEP_Parameters& para)
{
  if (para.flexible_zbl) {
    write_nep_block(file, "zbl_parameters", para.zbl_para.data(),
      5 * para.num_types * (para.num_types + 1));
  }
}

void write_nep_restart(
  const char* filename,
  const NEP_Parameters& para,
  const std::vector<float>& mu,
  const std::vector<float>& sigma)
{
  FILE* file = my_fopen(filename, "w");
  write_nep_header(file, para, 0, 0);
  write_nep_parameters(file, "parameters", para, mu.data());
  write_nep_parameters(file, "sigma", para, sigma.data());
  write_nep_zbl(file, para);
  const bool failed = ferror(file) != 0;
  if (fclose(file) != 0 || failed) {
    PRINT_INPUT_ERROR(("Failed to write " + std::string(filename) + ".").c_str());
  }
}

void write_nep_model(
  FILE* file, NEP_Parameters& para, const float* parameters, int max_radial, int max_angular)
{
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  write_nep_header(file, para, max_radial, max_angular);
  write_nep_parameters(file, "parameters", para, parameters);
  write_nep_block(file, "q_scaler", para.q_scaler_cpu.data(), para.dim);
  write_nep_zbl(file, para);
  if (ferror(file)) {
    PRINT_INPUT_ERROR("Failed to write the NEP model.");
  }
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
  const float scale = fabs(from_nep_in) + fabs(from_file) + 1.0e-10f;
  if (fabs(from_nep_in - from_file) > 1.0e-5f * scale) {
    mismatches.push_back(
      name + ": nep.in gives " + float_to_string(from_nep_in) + source_of_value(is_set) + ", " +
      filename + " gives " + float_to_string(from_file) + ".");
  }
}

static void compare_with_nep_txt_fine_tune(
  const NEP_Parameters& para, const NEP_Model_File& header,
  const std::string& filename, std::vector<std::string>& mismatches)
{
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

  const std::string& expected_model = para.model_name;
  if (header.name != expected_model) {
    mismatches.push_back("model: nep.in implies " + expected_model + ", " + filename +
      " holds " + header.name + ".");
  }

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
        header.use_typewise_cutoff_zbl ? header.typewise_cutoff_zbl_factor : -1.0f,
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
    header.has_q_222 ? 2 : 0,
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
  const NEP_Parameters& para, const NEP_Model_File& header,
  const std::string& filename, std::vector<std::string>& mismatches)
{
  const std::string& expected_model = para.model_name;
  if (header.name != expected_model) {
    mismatches.push_back("model: nep.in implies " + expected_model + ", " + filename +
      " holds " + header.name + ".");
  }
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
        header.use_typewise_cutoff_zbl ? header.typewise_cutoff_zbl_factor : -1.0f,
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
    "L_max_4body", para.has_q_222 ? 2 : 0, para.is_l_max_set,
    header.has_q_222 ? 2 : 0, filename, mismatches);
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

static NEP_Model_File check_nep_file(
  const NEP_Parameters& para, const std::string& filename, bool restart, bool fine_tune)
{
  NEP_Model_File model;
  std::vector<float> sigma;
  std::string error;
  if (!read_nep_file(filename, model, error, restart ? &sigma : nullptr)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  std::vector<std::string> mismatches;
  if (fine_tune) {
    compare_with_nep_txt_fine_tune(para, model, filename, mismatches);
  } else {
    compare_with_nep_txt(para, model, filename, mismatches);
  }
  if (!mismatches.empty()) {
    printf("The model in nep.in is inconsistent with %s:\n", filename.c_str());
    for (const std::string& mismatch : mismatches) {
      printf("    %s\n", mismatch.c_str());
    }
    PRINT_INPUT_ERROR(("nep.in is inconsistent with " + filename + ".").c_str());
  }
  return model;
}

void check_existing_nep_model(NEP_Parameters& para)
{
  if (para.prediction == 1) {
    check_nep_file(para, "nep.txt", false, false);
    return;
  }
  if (para.fine_tune) {
    check_nep_file(para, para.fine_tune_nep_txt, false, true);
    check_nep_file(para, para.fine_tune_nep_restart, true, true);
    return;
  }
  if (para.import_q_scaler) {
    check_nep_file(para, "nep.txt", false, false);
  }
  if (does_file_exist("nep.restart")) {
    const NEP_Model_File model = check_nep_file(para, "nep.restart", true, false);
    para.energy_reference = model.energy_reference;
  }
}
