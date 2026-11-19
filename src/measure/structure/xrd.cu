/*
    Copyright 2026 Ke Li and GPUMD development team
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

// Original XRD implementation by Ke Li.
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "integrate/integrate.cuh"
#include "measure/parse_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "xrd.cuh"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>

namespace
{
constexpr double pi = 3.14159265358979323846;
constexpr int block_size = 128;
constexpr int batch_size = 65536;
constexpr int max_mesh_nodes = 10000000;

__global__ void xrd_intensities(
  const XrdNode* nodes,
  const int number_atoms,
  const int number_selected,
  const int number_factors,
  const int* selected,
  const int* types,
  const XrdFormFactor* factors,
  const double* positions,
  double* intensities)
{
  extern __shared__ double weights[];
  __shared__ double real_parts[block_size];
  __shared__ double imag_parts[block_size];
  const int lane = threadIdx.x;
  const XrdNode node = nodes[blockIdx.x];

  for (int t = lane; t < number_factors; t += block_size) {
    const auto factor = factors[t];
    double weight = factor.c;
    for (int j = 0; j < 4; ++j) {
      weight += factor.a[j] * exp(-factor.b[j] * node.s_squared);
    }
    weights[t] = weight;
  }
  __syncthreads();

  double real = 0.0;
  double imag = 0.0;
  for (int i = lane; i < number_selected; i += block_size) {
    const int j = selected ? selected[i] : i;
    const double k_dot_r = node.kx * positions[j] +
                           node.ky * positions[j + static_cast<size_t>(number_atoms)] +
                           node.kz * positions[j + 2 * static_cast<size_t>(number_atoms)];
    double sine, cosine;
    sincos(2.0 * pi * k_dot_r, &sine, &cosine);
    const double weight = weights[types[j]];
    real += weight * cosine;
    imag += weight * sine;
  }
  real_parts[lane] = real;
  imag_parts[lane] = imag;
  __syncthreads();
  for (int stride = block_size / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      real_parts[lane] += real_parts[lane + stride];
      imag_parts[lane] += imag_parts[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    intensities[blockIdx.x] =
      node.lp * (real_parts[0] * real_parts[0] + imag_parts[0] * imag_parts[0]) / number_selected;
  }
}

double positive_number(const std::string& value)
{
  double result;
  if (!is_valid_real(value, &result) || !std::isfinite(result) || result <= 0.0) {
    PRINT_INPUT_ERROR("compute_xrd requires a finite, positive wavelength and mesh values.");
  }
  return result;
}

double finite_coefficient(const std::string& value)
{
  double result;
  if (!is_valid_real(value, &result) || !std::isfinite(result)) {
    PRINT_INPUT_ERROR("compute_xrd form_factor requires finite coefficients.");
  }
  return result;
}
} // namespace

XRD::XRD(const std::vector<std::string>& tokens, const std::vector<Group>& groups)
{
  action_name = "compute_xrd";
  if (tokens.size() < 3 || !is_valid_int(tokens[1], &interval_) || interval_ <= 0) {
    PRINT_INPUT_ERROR("Expected compute_xrd <positive interval> <wavelength> [options].");
  }
  wavelength_ = positive_number(tokens[2]);

  std::set<std::string> options;
  const int count = tokens.size();
  for (int i = 3; i < count; ++i) {
    const auto& key = tokens[i];
    if (key != "form_factor" && !options.insert(key).second) {
      PRINT_INPUT_ERROR("Repeated compute_xrd option.");
    }
    if (key == "group") {
      parse_group(tokens, false, groups, i, grouping_method_, group_id_);
    } else if ((key == "spacing" || key == "scale") && i + 3 < count) {
      manual_ = key == "spacing";
      for (auto& value : mesh_) {
        value = positive_number(tokens[++i]);
      }
    } else if (key == "two_theta" && i + 2 < count) {
      if (
        !is_valid_real(tokens[++i], &angle_min_) || !is_valid_real(tokens[++i], &angle_max_) ||
        !(angle_min_ >= 0 && angle_min_ < angle_max_ && angle_max_ <= 180)) {
        PRINT_INPUT_ERROR("compute_xrd requires 0 <= min 2theta < max 2theta <= 180 in degrees.");
      }
    } else if (key == "lp" && i + 1 < count) {
      const auto& value = tokens[++i];
      if (value != "0" && value != "1") {
        PRINT_INPUT_ERROR("compute_xrd lp expects 0 or 1.");
      }
      use_lp_ = value == "1";
    } else if (key == "form_factor" && i + 10 < count) {
      const auto& element = tokens[++i];
      XrdFormFactor factor;
      for (int j = 0; j < 4; ++j) {
        factor.a[j] = finite_coefficient(tokens[++i]);
        factor.b[j] = finite_coefficient(tokens[++i]);
      }
      factor.c = finite_coefficient(tokens[++i]);
      if (!input_factors_.emplace(element, factor).second) {
        PRINT_INPUT_ERROR("Repeated compute_xrd form_factor for an element.");
      }
    } else if (key == "file" && i + 1 < count) {
      filename_ = tokens[++i];
    } else {
      const std::string error = "Invalid or incomplete compute_xrd option: " + key;
      PRINT_INPUT_ERROR(error.c_str());
    }
  }
  if (options.count("scale") && options.count("spacing")) {
    PRINT_INPUT_ERROR("Specify one of compute_xrd scale or spacing.");
  }
  printf(
    "Compute X-ray diffraction every %d steps at wavelength %.8g A.\n", interval_, wavelength_);
}

void XRD::pre_run(
  int number_of_steps, double, Integrate&, std::vector<Group>& groups, Atom& atom, Box& box, Force&)
{
  if (interval_ > number_of_steps) {
    PRINT_INPUT_ERROR("compute_xrd interval exceeds the number of run steps.");
  }

  number_selected_ = atom.number_of_atoms;
  if (grouping_method_ >= 0) {
    const auto& group = groups[grouping_method_];
    number_selected_ = group.cpu_size[group_id_];
  }
  if (number_selected_ <= 0) {
    PRINT_INPUT_ERROR("compute_xrd requires a nonempty atom selection.");
  }

  cpu_factors_.resize(atom.cpu_type_size.size());
  factor_initialized_.assign(cpu_factors_.size(), false);
  factors_.resize(cpu_factors_.size());
  update_factors(atom, groups);

  update_mesh(box);
  output_.open(filename_, std::ios::app);
  if (!output_) {
    const std::string error = "Cannot open compute_xrd output: " + filename_;
    PRINT_INPUT_ERROR(error.c_str());
  }
  output_ << std::setprecision(17);
}

void XRD::update_factors(const Atom& atom, const std::vector<Group>& groups)
{
  const int* indices = nullptr;
  if (grouping_method_ >= 0) {
    const auto& group = groups[grouping_method_];
    indices = group.cpu_contents.data() + group.cpu_size_sum[group_id_];
  }
  // MC can introduce an atom type not previously present in the sampled group.
  // The kernel uses the current device types, including accepted MC swaps.
  bool changed = false;
  for (int i = 0; i < number_selected_; ++i) {
    const int j = indices ? indices[i] : i;
    const int type = atom.cpu_type[j];
    if (!factor_initialized_[type]) {
      const auto& element = atom.cpu_atom_symbol[j];
      const auto factor = input_factors_.find(element);
      if (factor == input_factors_.end()) {
        const std::string error = "Missing compute_xrd form_factor for element " + element + ".";
        PRINT_INPUT_ERROR(error.c_str());
      }
      cpu_factors_[type] = factor->second;
      factor_initialized_[type] = true;
      changed = true;
    }
  }
  if (changed) {
    factors_.copy_from_host(cpu_factors_.data());
  }
}

void XRD::update_mesh(const Box& box)
{
  // The reciprocal mesh and phase convention follow the kinematic diffraction equations
  // in Coleman et al., doi:10.1088/0965-0393/21/5/055020.
  // At pre_run, the cached box fast-path flag need not have been set yet.
  // Read the current matrix here and on each sampled step.
  for (int i : {1, 2, 3, 5, 6, 7}) {
    if (box.cpu_h[i] != 0.0) {
      PRINT_INPUT_ERROR("compute_xrd requires an axis-aligned orthogonal cell.");
    }
  }
  auto spacing = mesh_;
  if (!manual_) {
    const int periodic[] = {box.pbc_x, box.pbc_y, box.pbc_z};
    int periodic_count = 0;
    double inverse_sum = 0.0;
    for (int i = 0; i < 3; ++i) {
      if (periodic[i]) {
        const double length = box.cpu_h[4 * i];
        if (!std::isfinite(length) || length <= 0.0) {
          PRINT_INPUT_ERROR("Invalid periodic cell length in compute_xrd.");
        }
        inverse_sum += 1.0 / length;
        ++periodic_count;
      }
    }
    if (periodic_count == 0) {
      PRINT_INPUT_ERROR("Nonperiodic compute_xrd requires explicit spacing.");
    }
    for (int i = 0; i < 3; ++i) {
      spacing[i] *= periodic[i] ? 1.0 / box.cpu_h[4 * i] : inverse_sum / periodic_count;
    }
  }
  const double radius = 2.0 * std::sin(angle_max_ * pi / 360.0) / wavelength_;
  // Allow for roundoff in the integer bounds at the surface of the sphere.
  const double bound = radius * (1.0 + 16.0 * std::numeric_limits<double>::epsilon());
  const double radius_squared = bound * bound;
  if (!std::isfinite(radius_squared) || radius_squared == 0.0) {
    PRINT_INPUT_ERROR("compute_xrd wavelength gives an unrepresentable reciprocal radius.");
  }
  std::array<int, 3> extent;
  for (int i = 0; i < 3; ++i) {
    const double limit = bound / spacing[i];
    if (
      !std::isfinite(spacing[i]) || !std::isfinite(limit) || spacing[i] <= 0.0 ||
      limit > max_mesh_nodes) {
      PRINT_INPUT_ERROR(
        "compute_xrd mesh is too dense. Increase spacing or narrow the angle range.");
    }
    extent[i] = static_cast<int>(limit);
  }
  if (spacing == spacing_) {
    return;
  }
  spacing_ = spacing;

  nodes_.clear();
  size_t candidate_count = 0;
  for (int h = -extent[0]; h <= extent[0]; ++h) {
    const double x = h * spacing[0];
    const double yz_squared = std::max(0.0, radius_squared - x * x);
    const int kmax = std::min(extent[1], static_cast<int>(std::sqrt(yz_squared) / spacing[1]));
    for (int k = -kmax; k <= kmax; ++k) {
      const double y = k * spacing[1];
      const double z_squared = std::max(0.0, yz_squared - y * y);
      const int lmax = std::min(extent[2], static_cast<int>(std::sqrt(z_squared) / spacing[2]));
      candidate_count += 2 * lmax + 1;
      if (candidate_count > max_mesh_nodes) {
        PRINT_INPUT_ERROR("compute_xrd mesh exceeds 10000000 candidate nodes. Increase spacing.");
      }
      for (int l = -lmax; l <= lmax; ++l) {
        const double z = l * spacing[2];
        const double s_squared = (x * x + y * y + z * z) * 0.25;
        const double sine = wavelength_ * std::sqrt(s_squared);
        if (!(sine > 0.0 && sine < 1.0)) {
          continue; // omit forward scattering and the backscattering singularity
        }
        const double two_theta = 2.0 * std::asin(sine) * 180.0 / pi;
        if (two_theta < angle_min_ - 1.0e-12 || two_theta > angle_max_ + 1.0e-12) {
          continue;
        }
        double lp = 1.0;
        if (use_lp_) {
          const double cos_two_theta = 1.0 - 2.0 * sine * sine;
          lp = (1.0 + cos_two_theta * cos_two_theta) / (std::sqrt(1.0 - sine * sine) * sine * sine);
        }
        nodes_.push_back({h, k, l, x, y, z, two_theta, s_squared, lp});
      }
    }
  }
  if (nodes_.empty()) {
    return;
  }
  const size_t buffer_size = std::min(nodes_.size(), static_cast<size_t>(batch_size));
  if (intensities_.size() < buffer_size) {
    intensities_.resize(buffer_size);
    gpu_nodes_.resize(buffer_size);
    gpu_intensities_.resize(buffer_size);
  }
  if (nodes_.size() <= batch_size) {
    gpu_nodes_.copy_from_host(nodes_.data(), nodes_.size());
  }
}

void XRD::end_of_step(
  int,
  int step,
  int,
  int,
  double,
  double,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>&,
  Atom& atom,
  Force&)
{
  if ((step + 1) % interval_) {
    return;
  }
  update_mesh(box);
  update_factors(atom, groups);
  output_ << "# step " << step + 1 << " nodes " << nodes_.size() << " atoms " << number_selected_
          << " wavelength_A " << wavelength_ << " lp " << use_lp_ << " spacing_A^-1 " << spacing_[0]
          << ' ' << spacing_[1] << ' ' << spacing_[2] << '\n'
          << "# h k l two_theta_deg intensity\n";
  const auto& positions =
    is_pimd(integrate.get_type()) ? atom.position_beads[0] : atom.position_per_atom;
  const int* selected = nullptr;
  if (grouping_method_ >= 0) {
    const auto& group = groups[grouping_method_];
    selected = group.contents.data() + group.cpu_size_sum[group_id_];
  }
  for (size_t start = 0; start < nodes_.size(); start += batch_size) {
    const int count = std::min(nodes_.size() - start, static_cast<size_t>(batch_size));
    if (nodes_.size() > batch_size) {
      gpu_nodes_.copy_from_host(nodes_.data() + start, count);
    }
    xrd_intensities<<<count, block_size, cpu_factors_.size() * sizeof(double)>>>(
      gpu_nodes_.data(),
      atom.number_of_atoms,
      number_selected_,
      cpu_factors_.size(),
      selected,
      atom.type.data(),
      factors_.data(),
      positions.data(),
      gpu_intensities_.data());
    GPU_CHECK_KERNEL
    gpu_intensities_.copy_to_host(intensities_.data(), count);
    for (int i = 0; i < count; ++i) {
      const auto& node = nodes_[start + i];
      if (!std::isfinite(intensities_[i])) {
        PRINT_INPUT_ERROR(
          "compute_xrd produced a nonfinite intensity. Check coordinates and wavelength.");
      }
      output_ << node.h << ' ' << node.k << ' ' << node.l << ' ' << node.two_theta << ' '
              << intensities_[i] << '\n';
    }
  }
  output_.flush();
  if (!output_) {
    PRINT_INPUT_ERROR("Failed to write compute_xrd output.");
  }
}
