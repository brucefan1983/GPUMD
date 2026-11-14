/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "nep_spin.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
void NEP_Spin_Model::finalize_q_scaler()
{
  auto& para = *model_parameters;
  const int deviceCount = training->device_count;
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
    const float magnitude =
      std::max(1.0f, std::max(std::abs(global_max[d]), std::abs(global_min[d])));
    const float resolution = 64.0f * std::numeric_limits<float>::epsilon() * magnitude;
    if (!std::isfinite(range)) {
      PRINT_INPUT_ERROR("Cannot initialize Spin NEP q_scaler from a non-finite "
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
        PRINT_INPUT_ERROR("Cannot initialize Spin NEP q_scaler from a constant "
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
void NEP_Spin_Model::ResponseLoss::append(
  const int device_id, Dataset& dataset, Spin_Data& spin_data)
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
      prediction += spin_data.mforce_cpu[atom] * structure.spin_tangent_x[na] +
                    spin_data.mforce_cpu[dataset.N + atom] * structure.spin_tangent_y[na] +
                    spin_data.mforce_cpu[2 * dataset.N + atom] * structure.spin_tangent_z[na];
      target += structure.mfx[na] * structure.spin_tangent_x[na] +
                structure.mfy[na] * structure.spin_tangent_y[na] +
                structure.mfz[na] * structure.spin_tangent_z[na];
    }
    points_.push_back(
      {structure.spin_response_group, structure.spin_response_coordinate, prediction, target});
  }
}
static double huber_loss(const double residual)
{
  const double absolute = std::abs(residual);
  return absolute <= 1.0 ? 0.5 * residual * residual : absolute - 0.5;
}
float NEP_Spin_Model::ResponseLoss::value() const
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
    for (const auto* point : members)
      target_power += point->target * point->target;
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
        ((point->prediction - mean_prediction) - (point->target - mean_target)) / scale;
      group_shape_loss += huber_loss(residual);
    }
    shape_loss += group_shape_loss / members.size();
    mean_loss += huber_loss((mean_prediction - mean_target) / scale);
  }
  shape_loss /= groups.size();
  mean_loss /= groups.size();
  return static_cast<float>(shape_loss + 0.25 * mean_loss);
}
