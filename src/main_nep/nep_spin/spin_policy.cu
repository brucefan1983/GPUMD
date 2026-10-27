/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#include "spin_policy.cuh"
#include "rotation_response.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
namespace fitness_spin {
void finalize_q_scaler(Spin_Parameters& para, int deviceCount)
{
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
void print_loss_header(bool split_mforce)
{
  printf(
    "%-8s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s",
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
  if (split_mforce) printf(" M-parallel-Train M-perp-Train M-parallel-Test M-perp-Test");
  printf("\n");
}
void derive_spin_response_tangents(
  const Spin_Parameters& para,
  std::vector<Spin_Frame>& structures)
{
  std::map<std::string, std::vector<Spin_Frame*>> groups;
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
    std::sort(members.begin(), members.end(), [](const Spin_Frame* left, const Spin_Frame* right) {
      return left->label.spin_response_coordinate < right->label.spin_response_coordinate;
    });
    if (members.size() < 3) {
      PRINT_INPUT_ERROR(
        "Each rotation response_group needs at least three distinct "
        "response_coordinate values.\n");
    }
    const Spin_Frame& reference = *members.front();
    for (std::size_t k = 0; k < members.size(); ++k) {
      Spin_Frame& structure = *members[k];
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
      Spin_Frame& output = *members[k];
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
            if (!rotation_displacement(center, {{node.label.sx[atom], node.label.sy[atom], node.label.sz[atom]}}, delta)) {
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
void ResponseLoss::append(
  const int device_id,
  Dataset& dataset, Spin_Data& spin_data)
{
  CHECK(gpuSetDevice(device_id));
  spin_data.mforce.copy_to_host(spin_data.mforce_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    const auto& structure = spin_data.labels[nc];
    if (!structure.has_spin_response) {
      continue;
    }
    double prediction = 0.0;
    double target = 0.0;
    const int atom_offset = dataset.Na_sum_cpu[nc];
    for (int na = 0; na < dataset.structures[nc].num_atom; ++na) {
      const int atom = atom_offset + na;
      prediction +=
        spin_data.mforce_cpu[atom] * structure.spin_tangent_x[na] +
        spin_data.mforce_cpu[dataset.N + atom] * structure.spin_tangent_y[na] +
        spin_data.mforce_cpu[2 * dataset.N + atom] * structure.spin_tangent_z[na];
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

  double shape_loss = 0.0;
  double mean_loss = 0.0;
  for (const auto& item : groups) {
    const auto& members = item.second;
    // Equal group weighting also requires a group-local target scale: a strong
    // exchange scan must not suppress a weak anisotropy scan. Retain the
    // absolute floor for zero/near-zero reference responses.
    double target_power = 0.0;
    for (const auto* point : members) target_power += point->target * point->target;
    const double scale = std::max(std::sqrt(target_power / members.size()), 1.0e-6);
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
}
