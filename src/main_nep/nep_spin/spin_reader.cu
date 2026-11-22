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

/*----------------------------------------------------------------------------80
Major contributors:
  Chengbing Chen: Initial implementation.
------------------------------------------------------------------------------*/

#include "main_nep/xyz_reader.cuh"
#include "nep_spin.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

// Spin keeps the original double geometry independently of Structure's float data.
static void read_spin_geometry(const XYZ_Record& record, Spin_Data::Geometry& geometry)
{
  bool has_cell = false;
  for (size_t n = 0; n < record.tokens.size(); ++n) {
    const std::string key = "lattice=";
    if (record.tokens[n].compare(0, key.size(), key) != 0)
      continue;
    record.validate_array(n, 9);
    const int transpose[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    for (int k = 0; k < 9; ++k) {
      const auto& token = record.tokens[n + k];
      const size_t start = k == 0 ? key.size() + 1 : 0;
      geometry.cell[transpose[k]] = get_double_from_token(
        token.substr(start, token.size() - start - (k == 8)),
        record.filename.c_str(),
        record.comment_line_number);
    }
    has_cell = true;
  }
  if (!has_cell)
    PRINT_INPUT_ERROR("'lattice' is missing in the second line of a frame.");
  const int pos = record.get_property_offset("pos", 'R', 3);
  if (pos < 0)
    PRINT_INPUT_ERROR("'pos' is missing in properties.");
  geometry.position.resize(record.num_atoms);
  for (int a = 0; a < record.num_atoms; ++a)
    for (int d = 0; d < 3; ++d)
      geometry.position[a][d] = get_double_from_token(
        record.atoms[a][pos + d], record.filename.c_str(), record.comment_line_number + a + 1);
}

static bool read_spin_tensor(
  const XYZ_Record& record, size_t first, double scale, bool is_virial, float* result)
{
  record.validate_array(first, 9);
  double tensor[9];
  for (int k = 0; k < 9; ++k) {
    const auto& token = record.tokens[first + k];
    const size_t start = k == 0 ? token.find('"') + 1 : 0;
    tensor[k] = get_double_from_token(
      token.substr(start, token.size() - start - (k == 8)),
      record.filename.c_str(),
      record.comment_line_number);
  }
  // shortcut: symmetric virials use six labels, even if measured as a full tensor.
  const bool full =
    is_virial && (tensor[1] != tensor[3] || tensor[5] != tensor[7] || tensor[6] != tensor[2]);
  const double projected[9] = {
    tensor[0] * scale,
    tensor[4] * scale,
    tensor[8] * scale,
    (tensor[1] + tensor[3]) * (.5 * scale),
    (tensor[5] + tensor[7]) * (.5 * scale),
    (tensor[6] + tensor[2]) * (.5 * scale),
    (tensor[1] - tensor[3]) * (.5 * scale),
    (tensor[5] - tensor[7]) * (.5 * scale),
    (tensor[6] - tensor[2]) * (.5 * scale)};
  for (int d = 0; d < (full ? 9 : 6); ++d) {
    if (!std::isfinite(projected[d]) || std::abs(projected[d]) > std::numeric_limits<float>::max())
      PRINT_INPUT_ERROR("Per-atom virial values must be finite and within the float range.");
    result[d] = projected[d];
  }
  return full;
}

// A local displacement for differentiating a sampled spin path. The sphere log
// keeps fixed-length rotations tangent, while the separate radial difference
// retains genuine changes in the converged moment magnitude. Neighbouring
// samples must resolve the path (angular separation strictly below pi).
static bool rotation_displacement(
  const std::array<double, 3>& center,
  const std::array<double, 3>& neighbour,
  std::array<double, 3>& displacement)
{
  double r2 = 0.0, q2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    r2 += center[d] * center[d];
    q2 += neighbour[d] * neighbour[d];
  }
  const double r = std::sqrt(r2), q = std::sqrt(q2);
  // At zero moment there is no sphere tangent; retain the Cartesian limit.
  if (r <= 1.0e-12 || q <= 1.0e-12) {
    for (int d = 0; d < 3; ++d)
      displacement[d] = neighbour[d] - center[d];
    return true;
  }
  double cosine = 0.0;
  for (int d = 0; d < 3; ++d)
    cosine += (center[d] / r) * (neighbour[d] / q);
  cosine = std::max(-1.0, std::min(1.0, cosine));
  std::array<double, 3> tangent;
  double sine2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    tangent[d] = neighbour[d] / q - cosine * center[d] / r;
    sine2 += tangent[d] * tangent[d];
  }
  const double sine = std::sqrt(sine2);
  if (cosine < 0.0 && sine < 1.0e-10)
    return false;
  const double factor = sine > 1.0e-12 ? std::atan2(sine, cosine) / sine : 1.0;
  for (int d = 0; d < 3; ++d) {
    displacement[d] = r * factor * tangent[d] + (q - r) * center[d] / r;
  }
  return true;
}

static std::string frame_key(const Structure& geometry, const Spin_Data::FrameInput& label)
{
  std::ostringstream key;
  key << std::hexfloat;
  for (int t : geometry.type)
    key << t << ',';
  key << ';';
  auto append = [&](double x) { key << (x == 0.0 ? 0.0 : x) << ','; };
  for (double x : label.geometry.cell)
    append(x);
  for (const auto& position : label.geometry.position)
    for (double x : position)
      append(x);
  for (const auto* values : {&label.sx, &label.sy, &label.sz}) {
    key << ';';
    for (double x : *values)
      append(x);
  }
  return key.str();
}

void NEP_Spin_Model::Spin_Reader::read_structure(const XYZ_Record& record, Structure& structure)
{
  labels.emplace_back();
  auto& label = labels.back();
  read_spin_geometry(record, label.geometry);
  const auto& h = label.geometry.cell;
  const double volume = std::abs(
    h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
    h[2] * (h[3] * h[7] - h[4] * h[6]));
  float virial[9] = {}, stress[6] = {};
  bool has_virial = false, has_stress = false;
  // Common energy/force handling receives no model-specific tensor labels.
  XYZ_Record energy_record = record;
  for (size_t n = 0; n < energy_record.tokens.size();) {
    const auto& token = energy_record.tokens[n];
    const bool v = token.compare(0, 7, "virial=") == 0;
    const bool p = token.compare(0, 7, "stress=") == 0;
    if (!v && !p) {
      ++n;
      continue;
    }
    if (v) {
      label.has_full_virial =
        read_spin_tensor(energy_record, n, 1.0 / record.num_atoms, true, virial);
      has_virial = true;
    } else {
      read_spin_tensor(energy_record, n, -volume / record.num_atoms, false, stress);
      has_stress = true;
    }
    energy_record.tokens.erase(
      energy_record.tokens.begin() + n, energy_record.tokens.begin() + n + 9);
  }
  NEP_Energy_Reader::read_structure(energy_record, structure);
  if (has_virial && has_stress) {
    for (int d = 0; d < 6; ++d)
      if (std::abs(virial[d] - stress[d]) > 1e-3f && para.prediction == 0)
        PRINT_INPUT_ERROR("Virials and stresses for structure are inconsistent!");
    if (para.prediction == 0)
      printf("Structure has both defined virials and stresses. Will use virial information.\n");
  }
  structure.has_virial = has_virial || has_stress;
  for (int d = 0; d < 6; ++d)
    structure.virial[d] = has_virial ? virial[d] : (has_stress ? stress[d] : -1e6f);
  if (label.has_full_virial)
    std::copy_n(virial + 6, 3, label.antisymmetric_virial.begin());
  int spin_offset = -1, mforce_offset = -1, offset = 0;
  for (int k = 0; k < record.properties.size() / 3; ++k) {
    const auto& name = record.properties[3 * k];
    const int count = get_int_from_token(
      record.properties[3 * k + 2], record.filename.c_str(), record.comment_line_number);
    bool spin = name == "spin" || name == "spins" || name == "moment" || name == "moments";
    bool force = name == "mforce" || name == "mforces" || name == "force_mag" ||
                 name == "forces_mag" || name == "magnetic_force" || name == "magnetic_forces";
    if (name == "spin_tangent")
      PRINT_INPUT_ERROR(
        "spin_tangent is not an input label; it is derived from each final DFT spin path.");
    if (spin || force) {
      if (record.properties[3 * k + 1] != "r" || count != 3)
        PRINT_INPUT_ERROR("spin/mforce must have property type R:3.");
      int& slot = spin ? spin_offset : mforce_offset;
      if (slot >= 0)
        PRINT_INPUT_ERROR("Only one spin or magnetic-force property alias is allowed.");
      slot = offset;
    }
    offset += count;
  }
  if (spin_offset < 0)
    PRINT_INPUT_ERROR("NEP-spin requires spin/spins/moment/moments with property type R:3.");
  label.has_mforce = mforce_offset >= 0;
  for (auto* values : {&label.sx, &label.sy, &label.sz})
    values->resize(structure.num_atom, 0.0);
  for (auto* values : {&label.mfx, &label.mfy, &label.mfz})
    values->resize(structure.num_atom, 0.0f);
  for (int a = 0; a < structure.num_atom; ++a) {
    auto value = [&](int i) {
      return get_double_from_token(
        record.atoms[a][i], record.filename.c_str(), record.comment_line_number + a + 1);
    };
    // Excluded types supply geometry only. Keep their effective spins and
    // magnetic labels zero; the shared reader preserves energy/atomic forces.
    if (!parameters.spin_type_active[structure.type[a]])
      continue;
    label.sx[a] = value(spin_offset);
    label.sy[a] = value(spin_offset + 1);
    label.sz[a] = value(spin_offset + 2);
    if (label.has_mforce) {
      label.mfx[a] = value(mforce_offset);
      label.mfy[a] = value(mforce_offset + 1);
      label.mfz[a] = value(mforce_offset + 2);
    }
  }
  bool has_response_probe = false;
  bool has_response_group = false;
  bool has_response_coordinate = false;
  for (const auto& token : record.tokens) {
    const std::string probe = "response_probe=";
    const std::string group = "response_group=";
    const std::string coordinate = "response_coordinate=";
    if (token.substr(0, probe.length()) == probe) {
      has_response_probe = true;
      label.spin_response_probe = token.substr(probe.length(), token.length());
      if (label.spin_response_probe.empty()) {
        PRINT_INPUT_ERROR("response_probe should not be empty.\n");
      }
    } else if (token.substr(0, group.length()) == group) {
      has_response_group = true;
      label.spin_response_group = token.substr(group.length(), token.length());
      if (label.spin_response_group.empty()) {
        PRINT_INPUT_ERROR("response_group should not be empty.\n");
      }
    } else if (token.substr(0, coordinate.length()) == coordinate) {
      has_response_coordinate = true;
      label.spin_response_coordinate = get_double_from_token(
        token.substr(coordinate.length(), token.length()),
        record.filename.c_str(),
        record.comment_line_number);
    }
  }
  if (has_response_probe || has_response_group || has_response_coordinate) {
    if (!has_response_probe || !has_response_group || !has_response_coordinate) {
      PRINT_INPUT_ERROR("A magnetic response frame requires response_probe, response_group, "
                        "and response_coordinate.\n");
    }
    label.has_spin_response_metadata = 1;
  }
}

void NEP_Spin_Model::Spin_Reader::prepare_train(
  std::vector<Structure>& structures, int batch_size, int& batches, std::vector<int>& sizes)
{
  if (parameters.needs_spin_wall_bounds()) {
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> minima(parameters.num_types, infinity);
    std::vector<double> maxima(parameters.num_types, -infinity);
    for (const auto& structure : structures) {
      const auto& label = labels.at(structure.index_in_file);
      for (int a = 0; a < structure.num_atom; ++a) {
        const int type = structure.type[a];
        if (!parameters.spin_type_active[type])
          continue;
        const double norm = std::hypot(std::hypot(label.sx[a], label.sy[a]), label.sz[a]);
        if (!std::isfinite(norm))
          PRINT_INPUT_ERROR("Cannot resolve spin wall auto from a non-finite training moment.");
        minima[type] = std::min(minima[type], norm);
        maxima[type] = std::max(maxima[type], norm);
      }
    }
    parameters.resolve_spin_wall_bounds(minima, maxima);
  }
  for (const auto& structure : structures) {
    const auto& label = labels.at(structure.index_in_file);
    if (label.has_spin_response_metadata)
      train_groups.insert(label.spin_response_group);
  }
  NEP_Energy_Reader::prepare_train(structures, batch_size, batches, sizes);
}
void NEP_Spin_Model::Spin_Reader::prepare_test(
  const std::vector<Structure>& structures, bool has_test_set)
{
  for (const auto& structure : structures) {
    const auto& label = labels.at(structure.index_in_file);
    if (label.has_spin_response_metadata) {
      test_groups.insert(label.spin_response_group);
      if (parameters.lambda_spin_response > 0 && train_groups.count(label.spin_response_group))
        PRINT_INPUT_ERROR("response_group leakage between train.xyz and test.xyz.");
    }
    test_frames.insert(frame_key(structure, label));
  }
  NEP_Energy_Reader::prepare_test(structures, has_test_set);
}
void NEP_Spin_Model::Spin_Reader::prepare_dataset(Dataset& dataset)
{
  NEP_Energy_Reader::prepare_dataset(dataset);
  data.create(dataset).construct(parameters, dataset, labels);
}
const char* NEP_Spin_Model::Spin_Reader::energy_difference_conflict(
  const Structure& a, const Structure& b) const
{
  if (a.num_atom != b.num_atom || a.type != b.type || a.pbc != b.pbc)
    return nullptr;
  const auto& x = labels.at(a.index_in_file);
  const auto& y = labels.at(b.index_in_file);
  for (int d = 0; a.pbc && d < 9; ++d)
    if (std::abs(x.geometry.cell[d] - y.geometry.cell[d]) > 1e-5f)
      return nullptr;
  for (int i = 0; i < a.num_atom; ++i) {
    for (int d = 0; d < 3; ++d)
      if (std::abs(x.geometry.position[i][d] - y.geometry.position[i][d]) > 1e-5f)
        return nullptr;
    if (
      std::abs(x.sx[i] - y.sx[i]) > 1e-5f || std::abs(x.sy[i] - y.sy[i]) > 1e-5f ||
      std::abs(x.sz[i] - y.sz[i]) > 1e-5f)
      return nullptr;
  }
  return "have the same geometry and spins, for which the model predicts the same energy.";
}
void NEP_Spin_Model::Spin_Reader::read_response(
  std::vector<Structure>& structures, std::vector<Spin_Data::FrameInput>& response_labels)
{
  std::ifstream input("response.xyz");
  if (!input.is_open())
    PRINT_INPUT_ERROR("lambda_spin_response > 0 requires response.xyz.");
  XYZ_Reader reader(input, "response.xyz");
  XYZ_Record record;
  begin_reading();
  while (reader.read(record)) {
    structures.emplace_back();
    read_structure(record, structures.back());
    const auto& label = labels.back();
    if (!label.has_spin_response_metadata || label.spin_response_probe != "rotation")
      PRINT_INPUT_ERROR("Every response.xyz frame must have complete rotation response metadata.");
    if (test_groups.count(label.spin_response_group))
      PRINT_INPUT_ERROR("response_group leakage between response.xyz and test.xyz.");
    if (test_frames.count(frame_key(structures.back(), label)))
      PRINT_INPUT_ERROR("Response frame leakage between response.xyz and test.xyz.");
  }
  std::vector<Frame> frames;
  for (int i = 0; i < structures.size(); ++i)
    frames.push_back({structures[i], labels[i]});
  derive_spin_response_tangents(frames);
  response_labels = std::move(labels);
}

void NEP_Spin_Model::Spin_Reader::derive_spin_response_tangents(std::vector<Frame>& structures)
{
  const auto& para = parameters;
  std::map<std::string, std::vector<Frame*>> groups;
  for (auto& structure : structures) {
    structure.label.has_spin_response = 0;
    structure.label.spin_tangent_x.clear();
    structure.label.spin_tangent_y.clear();
    structure.label.spin_tangent_z.clear();
    if (
      structure.label.has_spin_response_metadata &&
      structure.label.spin_response_probe == "rotation") {
      if (!std::isfinite(structure.label.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.");
      }
      groups[structure.label.spin_response_group].push_back(&structure);
    }
  }
  if (groups.empty()) {
    PRINT_INPUT_ERROR("lambda_spin_response requires rotation response frames in response.xyz.\n");
  }

  const auto differs = [](double left, double right) {
    return !std::isfinite(left) || !std::isfinite(right) || std::abs(left - right) > 1.0e-6;
  };
  for (auto& item : groups) {
    auto& members = item.second;
    std::sort(members.begin(), members.end(), [](const Frame* left, const Frame* right) {
      return left->label.spin_response_coordinate < right->label.spin_response_coordinate;
    });
    if (members.size() < 3) {
      PRINT_INPUT_ERROR("Each rotation response_group needs at least three distinct "
                        "response_coordinate values.\n");
    }
    const Frame& reference = *members.front();
    for (std::size_t k = 0; k < members.size(); ++k) {
      Frame& structure = *members[k];
      if (!std::isfinite(structure.label.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.\n");
      }
      if (
        k > 0 && structure.label.spin_response_coordinate ==
                   members[k - 1]->label.spin_response_coordinate) {
        PRINT_INPUT_ERROR(
          "Each rotation response_group needs distinct response_coordinate values.\n");
      }
      if (!structure.label.has_mforce) {
        PRINT_INPUT_ERROR("Each rotation response frame requires mforce:R:3.\n");
      }
      if (
        structure.geometry.num_atom != reference.geometry.num_atom ||
        structure.geometry.type != reference.geometry.type) {
        PRINT_INPUT_ERROR("A rotation response_group cannot change atom count, type, or order.\n");
      }
      for (int d = 0; d < 9; ++d) {
        if (differs(structure.label.geometry.cell[d], reference.label.geometry.cell[d])) {
          PRINT_INPUT_ERROR("A rotation response_group cannot change its cell.\n");
        }
      }
      for (int atom = 0; atom < structure.geometry.num_atom; ++atom) {
        for (int d = 0; d < 3; ++d) {
          if (differs(
                structure.label.geometry.position[atom][d],
                reference.label.geometry.position[atom][d]))
            PRINT_INPUT_ERROR("A rotation response_group cannot change atomic positions.\n");
        }
        if (
          !std::isfinite(structure.label.mfx[atom]) || !std::isfinite(structure.label.mfy[atom]) ||
          !std::isfinite(structure.label.mfz[atom])) {
          PRINT_INPUT_ERROR("Magnetic forces in response.xyz must be finite.");
        }
        if (
          !std::isfinite(structure.label.sx[atom]) || !std::isfinite(structure.label.sy[atom]) ||
          !std::isfinite(structure.label.sz[atom])) {
          PRINT_INPUT_ERROR("Converged DFT spins in a rotation response_group must be finite.\n");
        }
      }
      structure.label.spin_tangent_x.resize(structure.geometry.num_atom);
      structure.label.spin_tangent_y.resize(structure.geometry.num_atom);
      structure.label.spin_tangent_z.resize(structure.geometry.num_atom);
    }

    double tangent_power = 0.0;
    for (std::size_t k = 0; k < members.size(); ++k) {
      const std::size_t first = k == 0 ? 0 : (k + 1 == members.size() ? k - 2 : k - 1);
      const std::size_t second = first + 1;
      const std::size_t third = first + 2;
      const double x0 = members[first]->label.spin_response_coordinate;
      const double x1 = members[second]->label.spin_response_coordinate;
      const double x2 = members[third]->label.spin_response_coordinate;
      const double x = members[k]->label.spin_response_coordinate;
      const double w0 = (2.0 * x - x1 - x2) / ((x0 - x1) * (x0 - x2));
      const double w1 = (2.0 * x - x0 - x2) / ((x1 - x0) * (x1 - x2));
      const double w2 = (2.0 * x - x0 - x1) / ((x2 - x0) * (x2 - x1));
      Frame& output = *members[k];
      for (int atom = 0; atom < output.geometry.num_atom; ++atom) {
        const bool active = para.spin_type_active[output.geometry.type[atom]] != 0;
        std::array<double, 3> tangent = {{0.0, 0.0, 0.0}};
        if (active) {
          const std::array<double, 3> center = {
            {output.label.sx[atom], output.label.sy[atom], output.label.sz[atom]}};
          const std::size_t nodes[] = {first, second, third};
          const double weights[] = {w0, w1, w2};
          for (int j = 0; j < 3; ++j) {
            const auto& node = *members[nodes[j]];
            std::array<double, 3> delta;
            if (!rotation_displacement(
                  center,
                  {{node.label.sx[atom], node.label.sy[atom], node.label.sz[atom]}},
                  delta)) {
              PRINT_INPUT_ERROR("Antipodal spins in response rotation stencil; add intermediate "
                                "rotation frames.\n");
            }
            for (int d = 0; d < 3; ++d)
              tangent[d] += weights[j] * delta[d];
          }
        }
        output.label.spin_tangent_x[atom] = static_cast<float>(tangent[0]);
        output.label.spin_tangent_y[atom] = static_cast<float>(tangent[1]);
        output.label.spin_tangent_z[atom] = static_cast<float>(tangent[2]);
        if (active) {
          for (double value : tangent)
            tangent_power += value * value;
        }
      }
      output.label.has_spin_response = 1;
    }
    if (tangent_power == 0.0) {
      PRINT_INPUT_ERROR(
        "A rotation response_group must contain a changing converged DFT spin path.\n");
    }
  }
}
