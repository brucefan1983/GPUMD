/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#include "nep_spin.cuh"
#include "rotation_response.cuh"
#include <map>
#include "main_nep/xyz_reader.cuh"
#include "utilities/read_file.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

static std::string frame_key(const Structure& geometry, const Spin_Data::Labels& label) {
  std::ostringstream key; key << std::hexfloat;
  for (int t:geometry.type) key << t << ',';
  key << ';'; for (float x:geometry.box_original) key << x << ',';
  for (const auto* values:{&geometry.x,&geometry.y,&geometry.z,&label.sx,&label.sy,&label.sz}) {
    key << ';'; for (float x:*values) key << x << ',';
  }
  return key.str();
}

void NEP_Spin_Model::Spin_Reader::read_structure(const XYZ_Record& record, Structure& structure) {
  NEP_Energy_Reader::read_structure(record,structure);
  labels.emplace_back();
  auto& label=labels.back();
  int spin_offset=-1, mforce_offset=-1, offset=0;
  for (int k=0; k<record.properties.size()/3; ++k) {
    const auto& name=record.properties[3*k];
    const int count=get_int_from_token(record.properties[3*k+2],record.filename.c_str(),record.comment_line_number);
    bool spin = name=="spin" || name=="spins" || name=="moment" || name=="moments";
    bool force = name=="mforce" || name=="mforces" || name=="force_mag" || name=="forces_mag" || name=="magnetic_force" || name=="magnetic_forces";
    if (name=="spin_tangent") PRINT_INPUT_ERROR("spin_tangent is not an input label; it is derived from each final DFT spin path.");
    if (spin || force) {
      if (record.properties[3*k+1]!="r" || count!=3) PRINT_INPUT_ERROR("spin/mforce must have property type R:3.");
      int& slot=spin ? spin_offset : mforce_offset;
      if (slot>=0) PRINT_INPUT_ERROR("Only one spin or magnetic-force property alias is allowed.");
      slot=offset;
    }
    offset+=count;
  }
  if (spin_offset<0) PRINT_INPUT_ERROR("spin_version 1 requires spin/spins/moment/moments with property type R:3.");
  label.has_mforce=mforce_offset>=0;
  for (auto* values:{&label.sx,&label.sy,&label.sz,&label.mfx,&label.mfy,&label.mfz}) values->resize(structure.num_atom,0.0f);
  for (int a=0;a<structure.num_atom;++a) {
    auto value=[&](int i) { return get_double_from_token(record.atoms[a][i],record.filename.c_str(),record.comment_line_number+a+1); };
    // Excluded types supply geometry only. Keep their effective spins and
    // magnetic labels zero; the shared reader preserves energy/atomic forces.
    if (!parameters.spin_type_active[structure.type[a]]) continue;
    label.sx[a]=value(spin_offset); label.sy[a]=value(spin_offset+1); label.sz[a]=value(spin_offset+2);
    if (label.has_mforce) { label.mfx[a]=value(mforce_offset);label.mfy[a]=value(mforce_offset+1);label.mfz[a]=value(mforce_offset+2); }
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
      label.spin_response_probe =
        token.substr(probe.length(), token.length());
      if (label.spin_response_probe.empty()) {
        PRINT_INPUT_ERROR("response_probe should not be empty.\n");
      }
    } else if (token.substr(0, group.length()) == group) {
      has_response_group = true;
      label.spin_response_group =
        token.substr(group.length(), token.length());
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
      PRINT_INPUT_ERROR(
        "A magnetic response frame requires response_probe, response_group, "
        "and response_coordinate.\n");
    }
    label.has_spin_response_metadata = 1;
  }

}

void NEP_Spin_Model::Spin_Reader::prepare_train(std::vector<Structure>& structures, int batch_size, int& batches, std::vector<int>& sizes) {
  for (const auto& structure:structures) {
    const auto& label=labels.at(structure.index_in_file);
    if (label.has_spin_response_metadata) train_groups.insert(label.spin_response_group);
  }
  NEP_Energy_Reader::prepare_train(structures,batch_size,batches,sizes);
}
void NEP_Spin_Model::Spin_Reader::prepare_test(const std::vector<Structure>& structures, bool has_test_set) {
  for (const auto& structure:structures) {
    const auto& label=labels.at(structure.index_in_file);
    if (label.has_spin_response_metadata) {
      test_groups.insert(label.spin_response_group);
      if (parameters.lambda_spin_response>0 && train_groups.count(label.spin_response_group))
        PRINT_INPUT_ERROR("response_group leakage between train.xyz and test.xyz.");
    }
    test_frames.insert(frame_key(structure,label));
  }
  NEP_Energy_Reader::prepare_test(structures,has_test_set);
}
void NEP_Spin_Model::Spin_Reader::prepare_dataset(Dataset& dataset) {
  NEP_Energy_Reader::prepare_dataset(dataset);
  data.create(dataset).construct(parameters,dataset,labels);
}
const char* NEP_Spin_Model::Spin_Reader::energy_difference_conflict(const Structure& a, const Structure& b) const {
  const char* conflict=NEP_Energy_Reader::energy_difference_conflict(a,b);
  if (!conflict) return nullptr;
  const auto& x=labels.at(a.index_in_file);const auto& y=labels.at(b.index_in_file);
  for (int i=0;i<a.num_atom;++i) {
    if (std::abs(x.sx[i]-y.sx[i])>1e-5f || std::abs(x.sy[i]-y.sy[i])>1e-5f || std::abs(x.sz[i]-y.sz[i])>1e-5f) return nullptr;
  }
  return "have the same geometry and spins, for which the model predicts the same energy.";
}
void NEP_Spin_Model::Spin_Reader::read_response(std::vector<Structure>& structures, std::vector<Spin_Data::Labels>& response_labels) {
  std::ifstream input("response.xyz");
  if (!input.is_open()) PRINT_INPUT_ERROR("lambda_spin_response > 0 requires response.xyz.");
  XYZ_Reader reader(input,"response.xyz"); XYZ_Record record;
  begin_reading();
  while(reader.read(record)) {
    structures.emplace_back(); read_structure(record,structures.back());
    const auto& label=labels.back();
    if (!label.has_spin_response_metadata || label.spin_response_probe!="rotation")
      PRINT_INPUT_ERROR("Every response.xyz frame must have complete rotation response metadata.");
    if (test_groups.count(label.spin_response_group)) PRINT_INPUT_ERROR("response_group leakage between response.xyz and test.xyz.");
    if (test_frames.count(frame_key(structures.back(),label))) PRINT_INPUT_ERROR("Response frame leakage between response.xyz and test.xyz.");
  }
  std::vector<Frame> frames;
  for (int i=0;i<structures.size();++i) frames.push_back({structures[i],labels[i]});
  derive_spin_response_tangents(frames);
  response_labels=std::move(labels);
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
    if (structure.label.has_spin_response_metadata &&
        structure.label.spin_response_probe == "rotation") {
      if (!std::isfinite(structure.label.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.");
      }
      groups[structure.label.spin_response_group].push_back(&structure);
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
    std::sort(members.begin(), members.end(), [](const Frame* left, const Frame* right) {
      return left->label.spin_response_coordinate < right->label.spin_response_coordinate;
    });
    if (members.size() < 3) {
      PRINT_INPUT_ERROR(
        "Each rotation response_group needs at least three distinct "
        "response_coordinate values.\n");
    }
    const Frame& reference = *members.front();
    for (std::size_t k = 0; k < members.size(); ++k) {
      Frame& structure = *members[k];
      if (!std::isfinite(structure.label.spin_response_coordinate)) {
        PRINT_INPUT_ERROR("response_coordinate must be finite.\n");
      }
      if (k > 0 &&
          structure.label.spin_response_coordinate == members[k - 1]->label.spin_response_coordinate) {
        PRINT_INPUT_ERROR(
          "Each rotation response_group needs distinct response_coordinate values.\n");
      }
      if (!structure.label.has_mforce) {
        PRINT_INPUT_ERROR(
          "Each rotation response frame requires mforce:R:3.\n");
      }
      if (structure.geometry.num_atom != reference.geometry.num_atom ||
          structure.geometry.type != reference.geometry.type) {
        PRINT_INPUT_ERROR(
          "A rotation response_group cannot change atom count, type, or order.\n");
      }
      for (int d = 0; d < 9; ++d) {
        if (differs(structure.geometry.box_original[d], reference.geometry.box_original[d])) {
          PRINT_INPUT_ERROR(
            "A rotation response_group cannot change its cell.\n");
        }
      }
      for (int atom = 0; atom < structure.geometry.num_atom; ++atom) {
        if (differs(structure.geometry.x[atom], reference.geometry.x[atom]) ||
            differs(structure.geometry.y[atom], reference.geometry.y[atom]) ||
            differs(structure.geometry.z[atom], reference.geometry.z[atom])) {
          PRINT_INPUT_ERROR(
            "A rotation response_group cannot change atomic positions.\n");
        }
        if (!std::isfinite(structure.label.mfx[atom]) || !std::isfinite(structure.label.mfy[atom]) ||
            !std::isfinite(structure.label.mfz[atom])) {
          PRINT_INPUT_ERROR("Magnetic forces in response.xyz must be finite.");
        }
        if (!std::isfinite(structure.label.sx[atom]) ||
            !std::isfinite(structure.label.sy[atom]) ||
            !std::isfinite(structure.label.sz[atom])) {
          PRINT_INPUT_ERROR(
            "Converged DFT spins in a rotation response_group must be finite.\n");
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
          const std::array<double, 3> center = {{
            output.label.sx[atom], output.label.sy[atom], output.label.sz[atom]}};
          const std::size_t nodes[] = {first, second, third};
          const double weights[] = {w0, w1, w2};
          for (int j = 0; j < 3; ++j) {
            const auto& node = *members[nodes[j]];
            std::array<double, 3> delta;
            if (!fitness_spin::rotation_displacement(center, {{node.label.sx[atom], node.label.sy[atom], node.label.sz[atom]}}, delta)) {
              PRINT_INPUT_ERROR(
                "Antipodal spins in response rotation stencil; add intermediate rotation frames.\n");
            }
            for (int d = 0; d < 3; ++d) tangent[d] += weights[j] * delta[d];
          }
        }
        output.label.spin_tangent_x[atom] = static_cast<float>(tangent[0]);
        output.label.spin_tangent_y[atom] = static_cast<float>(tangent[1]);
        output.label.spin_tangent_z[atom] = static_cast<float>(tangent[2]);
        if (active) {
          for (double value : tangent) tangent_power += value * value;
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
