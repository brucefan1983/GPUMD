/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#include "spin_io.cuh"
#include "utilities/nep_spin_model_file.cuh"
#include <sstream>
#include "main_nep/common/nep_io.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "utilities/nep_spin_legacy_parameters.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
namespace {
// Shared structural codec; model-specific metadata precedes this section.
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

bool read_nep_structural_header(
  std::istream& input, const std::string& filename, NepTxtHeader& header, std::string& error)
{
  std::vector<std::string> tokens;
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

  return true;
}


// parse the header of a nep.txt file; returns false and sets error if it cannot be parsed
static bool parse_model_token(const std::string& token, NepTxtHeader& header) {
  if (token != "nep4_spin3" && token != "nep4_spin3_zbl") return false;
  header.version = 4;
  header.enable_zbl = token == "nep4_spin3_zbl";
  return true;
}
bool read_nep_txt_header(const std::string& filename, NepTxtHeader& header, std::string& error)
{
  std::istringstream input(read_nep_spin_model_input(filename));
  if (!input.good()) {
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
  {
    tokens = get_tokens(input);
    if (tokens.size() != 3 || tokens[0] != "spin_mode" ||
        get_int_from_token(tokens[1], __FILE__, __LINE__) != 3) {
      error = "Invalid counted Spin3 header in " + filename + ".";
      return false;
    }
    const int count = get_int_from_token(tokens[2], __FILE__, __LINE__);
    if (count < 9 || count > 11) {
      error = "Invalid counted Spin3 header length in " + filename + ".";
      return false;
    }
    for (int n = 0; n < count; ++n) {
      if (get_tokens(input).empty()) {
        error = "Truncated Spin3 header in " + filename + ".";
        return false;
      }
    }
    header.number_of_header_lines += count + 1;
  }

  return read_nep_structural_header(input, filename, header, error);
}

void load_spin_checkpoint_metadata(Spin_Parameters& para)
{
  std::istringstream input(read_nep_spin_model_input("nep.txt"));
  if (!input.good()) {
    PRINT_INPUT_ERROR("Failed to open nep.txt for Spin NEP prediction.");
  }
  auto tokens = get_tokens(input);
  const std::string expected_tag = para.enable_zbl
    ? "nep4_spin3_zbl"
    : "nep4_spin3";
  if (tokens.size() != static_cast<std::size_t>(para.num_types + 2) ||
      tokens[0] != expected_tag ||
      get_int_from_token(tokens[1], __FILE__, __LINE__) != para.num_types) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint header does not match nep.in.");
  }
  for (int type = 0; type < para.num_types; ++type) {
    if (tokens[type + 2] != para.elements[type]) {
      PRINT_INPUT_ERROR("Spin NEP checkpoint atom types do not match nep.in.");
    }
  }
  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "spin_mode" ||
      get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_mode) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint has an invalid counted spin header.");
  }
  const int line_count = get_int_from_token(tokens[2], __FILE__, __LINE__);
  constexpr int minimum_lines = 9;
  if (line_count < minimum_lines || line_count > minimum_lines + 2) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint has an invalid counted header length.");
  }

  bool seen_baseline = false;
  bool seen_basis_size = false;
  bool seen_l_max = false;
  bool seen_compress = false;
  bool seen_cutoff = false;
  bool seen_scaler = false;
  bool seen_order = false;
  bool seen_soc = false;
  bool seen_projection_size = false;
  bool seen_dof = false;
  bool seen_env = false;
  std::vector<int> dof(para.num_types, 1);
  std::vector<int> env;
  auto parse_active = [&](const std::vector<std::string>& line_tokens) {
    std::vector<int> active(para.num_types, 0);
    if (line_tokens.size() < 2) {
      PRINT_INPUT_ERROR("Spin type-selection line must enable at least one type.");
    }
    for (std::size_t index = 1; index < line_tokens.size(); ++index) {
      auto found =
        std::find(para.elements.begin(), para.elements.end(), line_tokens[index]);
      if (found == para.elements.end()) {
        PRINT_INPUT_ERROR("Spin checkpoint contains an unknown atom type.");
      }
      const int type = static_cast<int>(found - para.elements.begin());
      if (active[type]) {
        PRINT_INPUT_ERROR("Spin checkpoint contains a duplicate atom type.");
      }
      active[type] = 1;
    }
    return active;
  };
  for (int line = 0; line < line_count; ++line) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Spin NEP checkpoint has a truncated spin header.");
    }
    const std::string& keyword = tokens[0];
    if (keyword == "spin_baseline") {
      if (seen_baseline ||
          tokens.size() != static_cast<std::size_t>(para.num_types + 1)) {
        PRINT_INPUT_ERROR("Invalid spin_baseline in Spin NEP checkpoint.");
      }
      seen_baseline = true;
      para.energy_reference.resize(para.num_types);
      for (int type = 0; type < para.num_types; ++type) {
        para.energy_reference[type] =
          get_double_from_token(tokens[type + 1], __FILE__, __LINE__);
      }
    } else if (keyword == "spin_basis_size") {
      if (seen_basis_size || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) !=
            para.spin_basis_size[0]) {
        PRINT_INPUT_ERROR("spin_basis_size in nep.txt does not match nep.in.");
      }
      seen_basis_size = true;
    } else if (keyword == "spin_l_max") {
      if (seen_l_max || tokens.size() != 2) {
        PRINT_INPUT_ERROR("Invalid spin_l_max in Spin NEP checkpoint.");
      }
      if (get_int_from_token(tokens[1], __FILE__, __LINE__) !=
          para.spin_l_max[0]) {
        PRINT_INPUT_ERROR("spin_l_max in nep.txt does not match nep.in.");
      }
      seen_l_max = true;
    } else if (keyword == "spin_compress") {
      if (seen_compress || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_compress) {
        PRINT_INPUT_ERROR("spin_compress in nep.txt does not match nep.in.");
      }
      seen_compress = true;
    } else if (keyword == "spin_cutoff") {
      if (seen_cutoff ||
          (tokens.size() != 2 &&
           tokens.size() != static_cast<std::size_t>(para.num_types + 1))) {
        PRINT_INPUT_ERROR("Invalid spin_cutoff in Spin NEP checkpoint.");
      }
      for (int type = 0; type < para.num_types; ++type) {
        const std::size_t value_index = tokens.size() == 2 ? 1 : type + 1;
        const double value =
          get_double_from_token(tokens[value_index], __FILE__, __LINE__);
        if (std::abs(value - para.spin_cutoff_by_type[type]) >
            1.0e-6 * std::max(1.0, std::abs(value))) {
          PRINT_INPUT_ERROR("spin_cutoff in nep.txt does not match nep.in.");
        }
      }
      seen_cutoff = true;
    } else if (keyword == "spin_order") {
      if (seen_order || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_order) {
        PRINT_INPUT_ERROR("spin_order in nep.txt does not match nep.in.");
      }
      seen_order = true;
    } else if (keyword == "spin_soc") {
      if (seen_soc || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != para.spin_soc) {
        PRINT_INPUT_ERROR("spin_soc in nep.txt does not match nep.in.");
      }
      seen_soc = true;
    } else if (keyword == "spin_projection_size") {
      if (seen_projection_size || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) !=
            para.number_of_variables_spin_projection) {
        PRINT_INPUT_ERROR("spin_projection_size in nep.txt does not match nep.in.");
      }
      seen_projection_size = true;
    } else if (keyword == "spin_scaler") {
      if (seen_scaler || tokens.size() != 2 ||
          get_int_from_token(tokens[1], __FILE__, __LINE__) != 1) {
        PRINT_INPUT_ERROR("spin_scaler in nep.txt must occur once and equal 1.");
      }
      seen_scaler = true;
    } else if (keyword == "spin_dof_type") {
      if (seen_dof) {
        PRINT_INPUT_ERROR("Duplicate spin_dof_type in Spin NEP checkpoint.");
      }
      dof = parse_active(tokens);
      seen_dof = true;
    } else if (keyword == "spin_env_type") {
      if (seen_env) {
        PRINT_INPUT_ERROR("Duplicate spin_env_type in Spin NEP checkpoint.");
      }
      env = parse_active(tokens);
      seen_env = true;
    } else {
      PRINT_INPUT_ERROR("Unknown line in counted Spin NEP checkpoint header.");
    }
  }
  if (!seen_baseline || !seen_basis_size || !seen_l_max ||
      !seen_compress || !seen_cutoff || !seen_scaler || !seen_order ||
      !seen_soc || !seen_projection_size) {
    PRINT_INPUT_ERROR("Spin NEP checkpoint is missing required metadata.");
  }
  if (!seen_env) {
    env = dof;
  }
  for (int type = 0; type < para.num_types; ++type) {
    if (dof[type] && !env[type]) {
      PRINT_INPUT_ERROR("spin_dof_type must be a subset of spin_env_type.");
    }
  }
  if (para.is_spin_dof_type_set && para.spin_dof_type_active != dof) {
    PRINT_INPUT_ERROR("spin_dof_type in nep.in conflicts with the Spin3 checkpoint.");
  }
  if (para.is_spin_env_type_set && para.spin_env_type_active != env) {
    PRINT_INPUT_ERROR("spin_env_type in nep.in conflicts with the Spin3 checkpoint.");
  }
  para.spin_dof_type_active = dof;
  para.spin_env_type_active = env;
  if (para.enable_zbl) {
    tokens = get_tokens(input);
    const std::size_t expected_size = para.use_typewise_cutoff_zbl ? 4 : 3;
    if (tokens.size() != expected_size || tokens[0] != "zbl") {
      PRINT_INPUT_ERROR("Spin3 ZBL checkpoint has an invalid zbl line.");
    }
    const double inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    const double outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (std::abs(inner - para.zbl_rc_inner) > 1.0e-6 ||
        std::abs(outer - para.zbl_rc_outer) > 1.0e-6 ||
        (para.use_typewise_cutoff_zbl &&
         std::abs(get_double_from_token(tokens[3], __FILE__, __LINE__) -
                  para.typewise_cutoff_zbl_factor) > 1.0e-6)) {
      PRINT_INPUT_ERROR("Spin3 ZBL checkpoint does not match nep.in.");
    }
  }
}
void write_checkpoint_metadata(FILE* fid_nep, const Spin_Parameters& para)
{
  // Serialize effective model state, including masks inherited from a checkpoint.
  const bool write_dof = std::any_of(
    para.spin_dof_type_active.begin(), para.spin_dof_type_active.end(),
    [](int active) { return active != 1; });
  const bool write_env = para.spin_env_type_active != para.spin_dof_type_active;
  fprintf(fid_nep, "spin_mode %d\n", para.spin_mode);
  fprintf(fid_nep, "spin_basis_size %d\n", para.spin_basis_size[0]);
  fprintf(fid_nep, "spin_l_max %d\n", para.spin_l_max[0]);
  fprintf(fid_nep, "spin_compress %d\n", para.spin_compress);
  fprintf(fid_nep, "spin_cutoff");
  const bool uniform_spin_cutoff = std::all_of(
    para.spin_cutoff_by_type.begin() + 1,
    para.spin_cutoff_by_type.end(),
    [&](float value) {
      return std::abs(value - para.spin_cutoff_by_type.front()) <= 1.0e-7f;
    });
  const int cutoff_count = uniform_spin_cutoff ? 1 : para.num_types;
  for (int type = 0; type < cutoff_count; ++type) {
    fprintf(fid_nep, " %.9g", para.spin_cutoff_by_type[type]);
  }
  fprintf(fid_nep, "\n");
  fprintf(fid_nep, "spin_order %d\n", para.spin_order);
  fprintf(fid_nep, "spin_soc %d\n", para.spin_soc);
  fprintf(
    fid_nep,
    "spin_projection_size %d\n",
    para.number_of_variables_spin_projection);
  fprintf(fid_nep, "spin_scaler 1\n");
  if (write_dof) {
    fprintf(fid_nep, "spin_dof_type");
    for (int type = 0; type < para.num_types; ++type) {
      if (para.spin_dof_type_active[type]) {
        fprintf(fid_nep, " %s", para.elements[type].c_str());
      }
    }
    fprintf(fid_nep, "\n");
  }
  if (write_env) {
    fprintf(fid_nep, "spin_env_type");
    for (int type = 0; type < para.num_types; ++type) {
      if (para.spin_env_type_active[type]) {
        fprintf(fid_nep, " %s", para.elements[type].c_str());
      }
    }
    fprintf(fid_nep, "\n");
  }
}
}

static void validate_model_header(Spin_Parameters& para) {
  NepTxtHeader h;
  std::string error;
  if (!read_nep_txt_header("nep.txt", h, error)) PRINT_INPUT_ERROR(error.c_str());
  para.number_of_nep_txt_header_lines = h.number_of_header_lines;
  if (h.model_token != para.file_tag || h.num_types != para.num_types || h.elements != para.elements ||
      h.n_max_radial != para.n_max_radial || h.n_max_angular != para.n_max_angular ||
      h.basis_size_radial != para.basis_size_radial || h.basis_size_angular != para.basis_size_angular ||
      h.L_max != para.L_max || h.has_q_222 != (para.has_q_222 ? 2 : 0) ||
      h.has_q_1111 != para.has_q_1111 || h.has_q_112 != para.has_q_112 ||
      h.has_q_123 != para.has_q_123 || h.has_q_233 != para.has_q_233 || h.has_q_134 != para.has_q_134 ||
      h.num_neurons1 != para.num_neurons1 || h.num_neurons2 != 0 || h.has_multiple_cutoffs) {
    PRINT_INPUT_ERROR("nep.in is inconsistent with the Spin3 model in nep.txt.");
  }
  const auto same = [](float a, float b) { return std::abs(a-b) <= 1.0e-5f*(std::abs(a)+std::abs(b)+1.0e-10f); };
  if (!same(h.rc_radial[0], para.rc_radial[0]) || !same(h.rc_angular[0], para.rc_angular[0])) {
    PRINT_INPUT_ERROR("Structural cutoffs in nep.txt do not match nep.in.");
  }
  para.flexible_zbl = h.flexible_zbl;
  load_spin_checkpoint_metadata(para);
}

void read_spin_model(Spin_Parameters& para, float* parameters) {
  validate_model_header(para);
  std::istringstream input(read_nep_spin_model_input("nep.txt"));
  for (int i=0; i<para.number_of_nep_txt_header_lines; ++i) get_tokens(input);
  for (int i=0; i<para.number_of_variables; ++i) {
    auto tokens=get_tokens(input);
    if (tokens.size()!=1) PRINT_INPUT_ERROR("Missing or malformed Spin3 model parameter.");
    parameters[i]=get_double_from_token(tokens[0], __FILE__, __LINE__);
    if (!std::isfinite(parameters[i])) PRINT_INPUT_ERROR("Spin3 model parameter must be finite.");
  }
  descriptor_parameters_to_channel_major(parameters, para.number_of_variables_ann, para.num_types*para.num_types,
    para.n_max_radial, para.n_max_angular, para.basis_size_radial, para.basis_size_angular);
  for (int d=0; d<para.dim; ++d) {
    auto tokens=get_tokens(input);
    if (tokens.size()!=1) PRINT_INPUT_ERROR("Missing or malformed q_scaler in nep.txt.");
    para.q_scaler_cpu[d]=get_double_from_token(tokens[0], __FILE__, __LINE__);
    if (!std::isfinite(para.q_scaler_cpu[d]) || para.q_scaler_cpu[d]<=0)
      PRINT_INPUT_ERROR("Spin3 checkpoint q_scaler must be finite and positive.");
  }
  if (para.flexible_zbl) {
    const int count = 5 * para.num_types * (para.num_types + 1);
    if (count > para.zbl_para.size()) PRINT_INPUT_ERROR("Spin3 flexible ZBL parameter count exceeds capacity.");
    for (int n = 0; n < count; ++n) {
      const auto tokens = get_tokens(input);
      if (tokens.size() != 1) PRINT_INPUT_ERROR("Missing Spin3 flexible ZBL parameter.");
      para.zbl_para[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
      if (!std::isfinite(para.zbl_para[n])) PRINT_INPUT_ERROR("Invalid Spin3 flexible ZBL parameter.");
    }
  }
  if (!get_tokens(input).empty()) PRINT_INPUT_ERROR("Unexpected data after Spin3 scaler.");
  int devices; CHECK(gpuGetDeviceCount(&devices));
  for (int d=0; d<devices; ++d) {
    CHECK(gpuSetDevice(d));
    para.q_scaler_gpu[d].copy_from_host(para.q_scaler_cpu.data());
  }
  CHECK(gpuSetDevice(0));
}

void initialize_spin_checkpoint(Spin_Parameters& para) {
  para.spin_restart = !para.prediction && std::ifstream("nep.restart").good();
  para.q_scaler_cpu.assign(para.dim, 1.0e10f);
  int devices; CHECK(gpuGetDeviceCount(&devices));
  if (devices > 16) PRINT_INPUT_ERROR("At most 16 training devices are supported.");
  for (int d=0; d<devices; ++d) {
    CHECK(gpuSetDevice(d));
    para.q_scaler_gpu[d].resize(para.dim);
    para.q_scaler_gpu[d].copy_from_host(para.q_scaler_cpu.data());
    para.q_scaler_max[d].resize(para.dim); para.q_scaler_max[d].fill(-1.0e10f);
    para.q_scaler_min[d].resize(para.dim); para.q_scaler_min[d].fill(1.0e10f);
    para.q_scaler_square_sum[d].resize(para.dim); para.q_scaler_square_sum[d].fill(0.0f);
    para.q_scaler_count[d].resize(para.dim); para.q_scaler_count[d].fill(0);
  }
  CHECK(gpuSetDevice(0));
  if (para.prediction || para.spin_restart || para.import_q_scaler) {
    std::vector<float> parameters(para.number_of_variables);
    read_spin_model(para, parameters.data());
  }
  if (para.spin_restart) {
    std::vector<float> mu, sigma;
    read_spin_restart(para, mu, sigma);
  } else if (!para.prediction) {
    // import_q_scaler imports only the scaler; new training fits its own reference.
    para.energy_reference.clear();
  }
}

void write_spin_model(FILE* fid_nep, Spin_Parameters& para, const float* elite, int max_NN_radial, int max_NN_angular) {
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  write_nep_header(fid_nep, para, max_NN_radial, max_NN_angular);
  write_checkpoint_metadata(fid_nep, para);
  write_nep_model_parameters(fid_nep, para, elite);
}

bool read_spin_restart(Spin_Parameters& para, std::vector<float>& mu, std::vector<float>& sigma)
{
  std::ifstream input("nep.restart");
  if (!input.good()) return false;
  std::string first;
  input >> first;
  if (first == "nep_restart") return read_nep_restart("nep.restart", para, mu, sigma, &para.energy_reference);
  // Legacy restart stores basis-major mu/sigma, with references in nep.txt.
  input.clear(); input.seekg(0);
  mu.clear(); sigma.clear();
  std::string line;
  while (std::getline(input, line)) {
    const auto tokens = get_tokens(line);
    if (tokens.size() != 2) PRINT_INPUT_ERROR("Legacy Spin3 restart requires mu sigma pairs.");
    const double m = get_double_from_token(tokens[0], __FILE__, __LINE__);
    const double s = get_double_from_token(tokens[1], __FILE__, __LINE__);
    if (!std::isfinite(m) || !std::isfinite(s) || std::abs(m) > std::numeric_limits<float>::max() ||
        s < 0 || s > std::numeric_limits<float>::max()) PRINT_INPUT_ERROR("Invalid Spin3 restart value.");
    mu.push_back(m); sigma.push_back(s);
  }
  if (mu.size() != size_t(para.number_of_variables)) PRINT_INPUT_ERROR("Spin3 restart parameter count mismatch.");
  for (auto* values : {&mu, &sigma})
    descriptor_parameters_to_channel_major(values->data(), para.number_of_variables_ann,
      para.num_types * para.num_types, para.n_max_radial, para.n_max_angular,
      para.basis_size_radial, para.basis_size_angular);
  return true;
}
