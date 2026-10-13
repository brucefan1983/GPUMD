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

#include "fitness_spin.cuh"
#include "parameters.cuh"
#include "dataset.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>

namespace fitness_spin {

bool prepare_checkpoint(Parameters& para)
{
  const bool spin_restart = para.spin_restart;
  if (para.spin_mode && (para.prediction || spin_restart || para.import_q_scaler)) {
    load_spin_checkpoint_metadata(para);
  }
  return spin_restart;
}

void prepare_training_data(Parameters& para, std::vector<Structure>& structures_train, bool spin_restart)
{
  if (para.spin_mode && !para.prediction && !spin_restart) {
    fit_spin_energy_baseline(structures_train, para);
    printf("Spin energy baseline:");
    for (const float value : para.spin_baseline) {
      printf(" %.10g", value);
    }
    printf("\n");
  }
}

// Compare stored geometry/spins, independent of energy/force labels and group names.
static std::string response_frame_key(const Structure& structure)
{
  std::ostringstream key;
  key << std::hexfloat;
  for (int type : structure.type) key << type << ',';
  key << ';';
  for (float value : structure.box_original) key << value << ',';
  for (const auto* values : {&structure.x, &structure.y, &structure.z,
                            &structure.sx, &structure.sy, &structure.sz}) {
    key << ';';
    for (float value : *values) key << value << ',';
  }
  return key.str();
}

void prepare_response_data(const Parameters& para, std::vector<Structure>& response,
  const std::vector<Structure>& train, const std::vector<Structure>& test)
{
  for (const auto& structure : response) {
    if (!structure.has_spin_response_metadata || structure.spin_response_probe != "rotation") {
      PRINT_INPUT_ERROR("Every response.xyz frame must have complete rotation response metadata.");
    }
  }
  derive_spin_response_tangents(para, response);
  std::set<std::string> test_groups;
  std::set<std::string> test_frames;
  for (const auto& structure : test) {
    if (structure.has_spin_response_metadata) test_groups.insert(structure.spin_response_group);
    test_frames.insert(response_frame_key(structure));
  }
  for (const auto& structure : train) {
    if (structure.has_spin_response_metadata && test_groups.count(structure.spin_response_group)) {
      PRINT_INPUT_ERROR("response_group leakage between train.xyz and test.xyz.");
    }
  }
  for (const auto& structure : response) {
    if (test_groups.count(structure.spin_response_group)) {
      PRINT_INPUT_ERROR("response_group leakage between response.xyz and test.xyz.");
    }
    if (test_frames.count(response_frame_key(structure))) {
      PRINT_INPUT_ERROR("Response frame leakage between response.xyz and test.xyz.");
    }
  }
}

void write_mforce(FILE* fid_mforce, Dataset& dataset)
{
  dataset.mforce.copy_to_host(dataset.mforce_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    if (!dataset.structures[nc].has_mforce) {
      continue;
    }
    const int offset = dataset.Na_sum_cpu[nc];
    for (int atom = 0; atom < dataset.Na_cpu[nc]; ++atom) {
      const int index = offset + atom;
      fprintf(
        fid_mforce,
        "%g %g %g %g %g %g\n",
        dataset.mforce_cpu[index],
        dataset.mforce_cpu[dataset.N + index],
        dataset.mforce_cpu[2 * dataset.N + index],
        dataset.mforce_ref_cpu[index],
        dataset.mforce_ref_cpu[dataset.N + index],
        dataset.mforce_ref_cpu[2 * dataset.N + index]);
    }
  }
}

void print_loss_header()
{
  printf(
    "%-8s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s\n",
    "Step",
    "Total",
    "L1Reg",
    "L2Reg",
    "E-Train",
    "F-Train",
    "V-Train",
    "M-Train",
    "T-Train",
    "E-Test",
    "F-Test",
    "V-Test",
    "M-Test",
    "T-Test");
}

void write_loss(FILE* fid_loss_out, int generation, float loss_total, float loss_L1, float loss_L2,
  float rmse_energy_train, float rmse_force_train, float rmse_virial_train,
  float rmse_mforce_train, float rmse_tau_train, float rmse_energy_test,
  float rmse_force_test, float rmse_virial_test, float rmse_mforce_test, float rmse_tau_test)
{
  printf(
    "%-8d%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f\n",
    generation + 1,
    loss_total,
    loss_L1,
    loss_L2,
    rmse_energy_train,
    rmse_force_train,
    rmse_virial_train,
    rmse_mforce_train,
    rmse_tau_train,
    rmse_energy_test,
    rmse_force_test,
    rmse_virial_test,
    rmse_mforce_test,
    rmse_tau_test);
  fprintf(
    fid_loss_out,
    "%-8d%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f%-11.5f\n",
    generation + 1,
    loss_total,
    loss_L1,
    loss_L2,
    rmse_energy_train,
    rmse_force_train,
    rmse_virial_train,
    rmse_mforce_train,
    rmse_tau_train,
    rmse_energy_test,
    rmse_force_test,
    rmse_virial_test,
    rmse_mforce_test,
    rmse_tau_test);
}

void load_spin_checkpoint_metadata(Parameters& para)
{
  std::ifstream input("nep.txt");
  if (!input.is_open()) {
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
      para.spin_baseline.resize(para.num_types);
      for (int type = 0; type < para.num_types; ++type) {
        para.spin_baseline[type] =
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

void fit_spin_energy_baseline(
  const std::vector<Structure>& structures, Parameters& para)
{
  const int T = para.num_types;
  std::vector<double> gram(T * T, 0.0);
  std::vector<double> rhs(T, 0.0);
  std::vector<double> counts(T);
  for (const auto& structure : structures) {
    std::fill(counts.begin(), counts.end(), 0.0);
    for (const int type : structure.type) {
      counts[type] += 1.0;
    }
    const double total_energy =
      static_cast<double>(structure.energy) * structure.num_atom;
    for (int i = 0; i < T; ++i) {
      rhs[i] += counts[i] * total_energy;
      for (int j = 0; j < T; ++j) {
        gram[i * T + j] += counts[i] * counts[j];
      }
    }
  }

  std::vector<double> eigenvectors(T * T, 0.0);
  for (int i = 0; i < T; ++i) {
    eigenvectors[i * T + i] = 1.0;
  }
  const int max_sweeps = std::max(16, 32 * T * T);
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p = 0;
    int q = 0;
    double largest = 0.0;
    for (int i = 0; i < T; ++i) {
      for (int j = i + 1; j < T; ++j) {
        const double value = std::abs(gram[i * T + j]);
        if (value > largest) {
          largest = value;
          p = i;
          q = j;
        }
      }
    }
    double diagonal_scale = 0.0;
    for (int i = 0; i < T; ++i) {
      diagonal_scale = std::max(diagonal_scale, std::abs(gram[i * T + i]));
    }
    if (largest <=
        std::numeric_limits<double>::epsilon() * std::max(1.0, diagonal_scale)) {
      break;
    }
    const double app = gram[p * T + p];
    const double aqq = gram[q * T + q];
    const double apq = gram[p * T + q];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (int k = 0; k < T; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double akp = gram[k * T + p];
      const double akq = gram[k * T + q];
      gram[k * T + p] = gram[p * T + k] = c * akp - s * akq;
      gram[k * T + q] = gram[q * T + k] = s * akp + c * akq;
    }
    gram[p * T + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    gram[q * T + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    gram[p * T + q] = gram[q * T + p] = 0.0;
    for (int k = 0; k < T; ++k) {
      const double vkp = eigenvectors[k * T + p];
      const double vkq = eigenvectors[k * T + q];
      eigenvectors[k * T + p] = c * vkp - s * vkq;
      eigenvectors[k * T + q] = s * vkp + c * vkq;
    }
  }

  double largest_eigenvalue = 0.0;
  for (int i = 0; i < T; ++i) {
    largest_eigenvalue =
      std::max(largest_eigenvalue, std::max(0.0, gram[i * T + i]));
  }
  const double eigenvalue_tolerance =
    largest_eigenvalue *
    std::max(structures.size(), static_cast<std::size_t>(T)) *
    std::numeric_limits<double>::epsilon();
  para.spin_baseline.assign(T, 0.0f);
  for (int column = 0; column < T; ++column) {
    const double eigenvalue = gram[column * T + column];
    if (eigenvalue <= eigenvalue_tolerance) {
      continue;
    }
    double projected_rhs = 0.0;
    for (int row = 0; row < T; ++row) {
      projected_rhs += eigenvectors[row * T + column] * rhs[row];
    }
    const double scale = projected_rhs / eigenvalue;
    for (int row = 0; row < T; ++row) {
      para.spin_baseline[row] +=
        static_cast<float>(eigenvectors[row * T + column] * scale);
    }
  }
}


void derive_spin_response_tangents(
  const Parameters& para,
  std::vector<Structure>& structures)
{
  std::map<std::string, std::vector<Structure*>> groups;
  for (auto& structure : structures) {
    structure.has_spin_response = 0;
    structure.spin_tangent_x.clear();
    structure.spin_tangent_y.clear();
    structure.spin_tangent_z.clear();
    if (structure.has_spin_response_metadata &&
        structure.spin_response_probe == "rotation") {
      if (!std::isfinite(structure.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.");
      }
      groups[structure.spin_response_group].push_back(&structure);
    }
  }
  if (groups.empty()) {
    PRINT_INPUT_ERROR(
      "lambda_spin_response requires rotation response frames in response.xyz.\n");
  }

  const auto differs = [](const float left, const float right) {
    return !std::isfinite(left) || !std::isfinite(right) ||
      std::abs(static_cast<double>(left) - static_cast<double>(right)) > 1.0e-6;
  };
  for (auto& item : groups) {
    auto& members = item.second;
    std::sort(members.begin(), members.end(), [](const Structure* left, const Structure* right) {
      return left->spin_response_coordinate < right->spin_response_coordinate;
    });
    if (members.size() < 3) {
      PRINT_INPUT_ERROR(
        "Each rotation response_group needs at least three distinct "
        "response_coordinate values.\n");
    }
    const Structure& reference = *members.front();
    for (std::size_t k = 0; k < members.size(); ++k) {
      Structure& structure = *members[k];
      if (!std::isfinite(structure.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.\n");
      }
      if (k > 0 &&
          structure.spin_response_coordinate == members[k - 1]->spin_response_coordinate) {
        PRINT_INPUT_ERROR(
          "Each rotation response_group needs distinct response_coordinate values.\n");
      }
      if (!structure.has_mforce) {
        PRINT_INPUT_ERROR(
          "Each rotation response frame requires mforce:R:3.\n");
      }
      if (structure.num_atom != reference.num_atom ||
          structure.type != reference.type) {
        PRINT_INPUT_ERROR(
          "A rotation response_group cannot change atom count, type, or order.\n");
      }
      for (int d = 0; d < 9; ++d) {
        if (differs(structure.box_original[d], reference.box_original[d])) {
          PRINT_INPUT_ERROR(
            "A rotation response_group cannot change its cell.\n");
        }
      }
      for (int atom = 0; atom < structure.num_atom; ++atom) {
        if (differs(structure.x[atom], reference.x[atom]) ||
            differs(structure.y[atom], reference.y[atom]) ||
            differs(structure.z[atom], reference.z[atom])) {
          PRINT_INPUT_ERROR(
            "A rotation response_group cannot change atomic positions.\n");
        }
        if (!std::isfinite(structure.mfx[atom]) || !std::isfinite(structure.mfy[atom]) ||
            !std::isfinite(structure.mfz[atom])) {
          PRINT_INPUT_ERROR("Magnetic forces in response.xyz must be finite.");
        }
        if (!std::isfinite(structure.sx[atom]) ||
            !std::isfinite(structure.sy[atom]) ||
            !std::isfinite(structure.sz[atom])) {
          PRINT_INPUT_ERROR(
            "Converged DFT spins in a rotation response_group must be finite.\n");
        }
      }
      structure.spin_tangent_x.resize(structure.num_atom);
      structure.spin_tangent_y.resize(structure.num_atom);
      structure.spin_tangent_z.resize(structure.num_atom);
    }

    double tangent_power = 0.0;
    for (std::size_t k = 0; k < members.size(); ++k) {
      const std::size_t first = k == 0 ? 0 : (k + 1 == members.size() ? k - 2 : k - 1);
      const std::size_t second = first + 1;
      const std::size_t third = first + 2;
      const double x0 = members[first]->spin_response_coordinate;
      const double x1 = members[second]->spin_response_coordinate;
      const double x2 = members[third]->spin_response_coordinate;
      const double x = members[k]->spin_response_coordinate;
      const double w0 = (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2));
      const double w1 = (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2));
      const double w2 = (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1));
      Structure& output = *members[k];
      for (int atom = 0; atom < output.num_atom; ++atom) {
        const double tx =
          w0 * members[first]->sx[atom] +
          w1 * members[second]->sx[atom] +
          w2 * members[third]->sx[atom];
        const double ty =
          w0 * members[first]->sy[atom] +
          w1 * members[second]->sy[atom] +
          w2 * members[third]->sy[atom];
        const double tz =
          w0 * members[first]->sz[atom] +
          w1 * members[second]->sz[atom] +
          w2 * members[third]->sz[atom];
        const bool active = para.spin_dof_type_active[output.type[atom]] != 0;
        output.spin_tangent_x[atom] = active ? static_cast<float>(tx) : 0.0f;
        output.spin_tangent_y[atom] = active ? static_cast<float>(ty) : 0.0f;
        output.spin_tangent_z[atom] = active ? static_cast<float>(tz) : 0.0f;
        if (active) {
          tangent_power += tx * tx + ty * ty + tz * tz;
        }
      }
      output.has_spin_response = 1;
    }
    if (tangent_power == 0.0) {
      PRINT_INPUT_ERROR(
        "A rotation response_group must contain a changing converged DFT spin path.\n");
    }
  }
}

void ResponseLoss::append(
  const int device_id,
  Dataset& dataset)
{
  CHECK(gpuSetDevice(device_id));
  dataset.mforce.copy_to_host(dataset.mforce_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    const Structure& structure = dataset.structures[nc];
    if (!structure.has_spin_response) {
      continue;
    }
    double prediction = 0.0;
    double target = 0.0;
    const int atom_offset = dataset.Na_sum_cpu[nc];
    for (int na = 0; na < structure.num_atom; ++na) {
      const int atom = atom_offset + na;
      prediction +=
        dataset.mforce_cpu[atom] * structure.spin_tangent_x[na] +
        dataset.mforce_cpu[dataset.N + atom] * structure.spin_tangent_y[na] +
        dataset.mforce_cpu[2 * dataset.N + atom] * structure.spin_tangent_z[na];
      target +=
        structure.mfx[na] * structure.spin_tangent_x[na] +
        structure.mfy[na] * structure.spin_tangent_y[na] +
        structure.mfz[na] * structure.spin_tangent_z[na];
    }
    points_.push_back({
      structure.spin_response_group,
      structure.spin_response_coordinate,
      prediction,
      target});
  }
}

static double huber_loss(const double residual)
{
  const double absolute = std::abs(residual);
  return absolute <= 1.0 ? 0.5 * residual * residual : absolute - 0.5;
}

float ResponseLoss::value() const
{
  std::map<std::string, std::vector<const Point*>> groups;
  for (const auto& point : points_) {
    groups[point.group].push_back(&point);
  }
  if (groups.empty()) {
    return 0.0f;
  }

  double group_power_sum = 0.0;
  for (const auto& item : groups) {
    const auto& members = item.second;
    double target_power = 0.0;
    for (const auto* point : members) {
      target_power += point->target * point->target;
    }
    group_power_sum += target_power / members.size();
  }
  const double scale = std::max(
    std::sqrt(group_power_sum / groups.size()), 1.0e-6);
  double shape_loss = 0.0;
  double mean_loss = 0.0;
  for (const auto& item : groups) {
    const auto& members = item.second;
    double mean_prediction = 0.0;
    double mean_target = 0.0;
    for (const auto* point : members) {
      mean_prediction += point->prediction;
      mean_target += point->target;
    }
    mean_prediction /= members.size();
    mean_target /= members.size();
    double group_shape_loss = 0.0;
    for (const auto* point : members) {
      const double residual =
        ((point->prediction - mean_prediction) -
         (point->target - mean_target)) / scale;
      group_shape_loss += huber_loss(residual);
    }
    shape_loss += group_shape_loss / members.size();
    mean_loss += huber_loss((mean_prediction - mean_target) / scale);
  }
  shape_loss /= groups.size();
  mean_loss /= groups.size();
  return static_cast<float>(shape_loss + 0.25 * mean_loss);
}

void finalize_q_scaler(Parameters& para, int deviceCount)
{
  if (!para.spin_mode) {
    return;
  }

  std::vector<float> global_max(para.dim, -1.0e10f);
  std::vector<float> global_min(para.dim, 1.0e10f);
  std::vector<float> device_max(para.dim);
  std::vector<float> device_min(para.dim);
  std::vector<double> global_square_sum(para.dim, 0.0);
  std::vector<unsigned long long> global_count(para.dim, 0);
  std::vector<float> device_square_sum(para.dim);
  std::vector<unsigned long long> device_count(para.dim);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    para.q_scaler_max[device_id].copy_to_host(device_max.data());
    para.q_scaler_min[device_id].copy_to_host(device_min.data());
    para.q_scaler_square_sum[device_id].copy_to_host(device_square_sum.data());
    para.q_scaler_count[device_id].copy_to_host(device_count.data());
    for (int d = 0; d < para.dim; ++d) {
      global_max[d] = std::max(global_max[d], device_max[d]);
      global_min[d] = std::min(global_min[d], device_min[d]);
      global_square_sum[d] += device_square_sum[d];
      global_count[d] += device_count[d];
    }
  }

  int unresolved_channels = 0;
  for (int d = 0; d < para.dim; ++d) {
    const float range = global_max[d] - global_min[d];
    const float magnitude = std::max(
      1.0f, std::max(std::abs(global_max[d]), std::abs(global_min[d])));
    const float resolution =
      64.0f * std::numeric_limits<float>::epsilon() * magnitude;
    if (!std::isfinite(range)) {
      PRINT_INPUT_ERROR(
        "Cannot initialize Spin NEP q_scaler from a non-finite "
        "descriptor channel.\n");
    }
    if (d >= para.dim_struct) {
      const float rms = global_count[d] == 0
        ? 0.0f
        : static_cast<float>(std::sqrt(
            global_square_sum[d] / static_cast<double>(global_count[d])));
      if (!std::isfinite(rms)) {
        PRINT_INPUT_ERROR("Cannot initialize Spin3 q_scaler from non-finite RMS.\n");
      }
      para.q_scaler_cpu[d] = 1.0f / std::max(1.0f, std::max(range, rms));
      if (range <= resolution && rms <= resolution) {
        ++unresolved_channels;
      }
    } else {
      if (range <= 1.0e-10f) {
        PRINT_INPUT_ERROR(
          "Cannot initialize Spin NEP q_scaler from a constant "
          "descriptor channel.\n");
      }
      para.q_scaler_cpu[d] = 1.0f / range;
    }
  }
  if (unresolved_channels > 0) {
    printf(
      "Spin3 q_scaler kept at 1 for %d numerically unresolved "
      "descriptor channel(s).\n",
      unresolved_channels);
  }
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    para.q_scaler_gpu[device_id].copy_from_host(para.q_scaler_cpu.data());
  }
  CHECK(gpuSetDevice(0));
}

void write_checkpoint_metadata(FILE* fid_nep, const Parameters& para)
{
  const int spin_header_lines =
    9 +
    static_cast<int>(para.is_spin_dof_type_set) +
    static_cast<int>(para.is_spin_env_type_set);
  fprintf(fid_nep, "spin_mode %d %d\n", para.spin_mode, spin_header_lines);
  fprintf(fid_nep, "spin_baseline");
  for (const float value : para.spin_baseline) {
    fprintf(fid_nep, " %.16e", static_cast<double>(value));
  }
  fprintf(fid_nep, "\n");
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
    fprintf(fid_nep, " %.16e", static_cast<double>(para.spin_cutoff_by_type[type]));
  }
  fprintf(fid_nep, "\n");
  fprintf(fid_nep, "spin_order %d\n", para.spin_order);
  fprintf(fid_nep, "spin_soc %d\n", para.spin_soc);
  fprintf(
    fid_nep,
    "spin_projection_size %d\n",
    para.number_of_variables_spin_projection);
  fprintf(fid_nep, "spin_scaler 1\n");
  if (para.is_spin_dof_type_set) {
    fprintf(fid_nep, "spin_dof_type");
    for (int type = 0; type < para.num_types; ++type) {
      if (para.spin_dof_type_active[type]) {
        fprintf(fid_nep, " %s", para.elements[type].c_str());
      }
    }
    fprintf(fid_nep, "\n");
  }
  if (para.is_spin_env_type_set) {
    fprintf(fid_nep, "spin_env_type");
    for (int type = 0; type < para.num_types; ++type) {
      if (para.spin_env_type_active[type]) {
        fprintf(fid_nep, " %s", para.elements[type].c_str());
      }
    }
    fprintf(fid_nep, "\n");
  }
}

} // namespace fitness_spin
