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

#include "main_nep/structure.cuh"
#include "nep_energy_reference.cuh"
#include "utilities/common.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

static void add_reference_row(
  std::vector<double>& r,
  std::vector<double>& z,
  std::vector<double>& row,
  double energy,
  const int n)
{
  for (int j = 0; j < n; ++j) {
    if (row[j] == 0.0) {
      continue;
    }
    const double diagonal = std::hypot(r[j * n + j], row[j]);
    const double c = r[j * n + j] / diagonal;
    const double s = row[j] / diagonal;
    r[j * n + j] = diagonal;
    for (int k = j + 1; k < n; ++k) {
      const double old_r = r[j * n + k];
      r[j * n + k] = c * old_r + s * row[k];
      row[k] = -s * old_r + c * row[k];
    }
    const double old_z = z[j];
    z[j] = c * old_z + s * energy;
    energy = -s * old_z + c * energy;
  }
}

static double reference_column_norm(const std::vector<double>& r, const int n, const int column)
{
  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm = std::hypot(norm, r[i * n + column]);
  }
  return norm;
}

static bool orthogonalize_reference_columns(
  std::vector<double>& r, std::vector<double>& v, const int n, const double relative_cutoff)
{
  const double tolerance = 8.0 * std::numeric_limits<double>::epsilon();
  double largest_column = 0.0;
  for (int j = 0; j < n; ++j) {
    largest_column = std::max(largest_column, reference_column_norm(r, n, j));
  }

  const double zero_tolerance = relative_cutoff * largest_column / std::sqrt(double(n));
  for (int sweep = 0; sweep < 100; ++sweep) {
    bool changed = false;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double norm_p = reference_column_norm(r, n, p);
        const double norm_q = reference_column_norm(r, n, q);
        if (
          norm_p == 0.0 || norm_q == 0.0 ||
          (norm_p <= zero_tolerance && norm_q <= zero_tolerance)) {
          continue;
        }
        double correlation = 0.0;
        for (int i = 0; i < n; ++i) {
          correlation += (r[i * n + p] / norm_p) * (r[i * n + q] / norm_q);
        }
        if (std::abs(correlation) <= tolerance) {
          continue;
        }

        const double scale = std::max(norm_p, norm_q);
        const double scaled_p = norm_p / scale;
        const double scaled_q = norm_q / scale;
        const double alpha = scaled_p * scaled_p;
        const double beta = scaled_q * scaled_q;
        const double gamma = correlation * scaled_p * scaled_q;

        const double delta = beta - alpha;
        const double t =
          2.0 * gamma / (delta + std::copysign(std::hypot(delta, 2.0 * gamma), delta));
        if (t == 0.0) {
          continue;
        }

        if ((norm_p > zero_tolerance && norm_q > zero_tolerance) || std::abs(t) > tolerance) {
          changed = true;
        }
        const double c = 1.0 / std::hypot(1.0, t);
        const double s = t * c;
        for (int i = 0; i < n; ++i) {
          double x = r[i * n + p];
          double y = r[i * n + q];
          r[i * n + p] = c * x - s * y;
          r[i * n + q] = s * x + c * y;
          x = v[i * n + p];
          y = v[i * n + q];
          v[i * n + p] = c * x - s * y;
          v[i * n + q] = s * x + c * y;
        }
      }
    }
    if (!changed) {
      return true;
    }
  }
  return false;
}

bool fit_nep_energy_reference(
  const std::vector<Structure>& structures,
  const int num_types,
  std::vector<double>& reference,
  int& rank,
  std::string& error)
{
  reference.clear();
  rank = 0;
  error.clear();
  if (num_types < 1 || num_types > NUM_ELEMENTS) {
    error = "Energy reference fit requires 1 to NUM_ELEMENTS atom types.";
    return false;
  }
  if (structures.empty()) {
    error = "Energy reference fit requires a nonempty training set.";
    return false;
  }

  const int n = num_types;
  std::vector<double> r(n * n, 0.0);
  std::vector<double> z(n, 0.0);
  std::vector<double> row(n, 0.0);
  size_t num_rows = 0;
  for (size_t nc = 0; nc < structures.size(); ++nc) {
    const Structure& structure = structures[nc];
    const std::string location = "Energy reference fit: structure " + std::to_string(nc + 1);
    if (structure.num_atom < 1 || structure.type.size() != size_t(structure.num_atom)) {
      error = location + " has an invalid atom count.";
      return false;
    }
    if (
      !std::isfinite(structure.energy_total) || !std::isfinite(structure.weight) ||
      !std::isfinite(structure.energy_weight) || structure.weight < 0.0f ||
      structure.energy_weight < 0.0f) {
      error = location + " requires finite energy and finite nonnegative weights.";
      return false;
    }
    std::fill(row.begin(), row.end(), 0.0);
    for (const int type : structure.type) {
      if (type < 0 || type >= n) {
        error = location + " has an atom type outside the model's element list.";
        return false;
      }
      row[type] += 1.0;
    }
    const double weight = static_cast<double>(structure.weight) *
                          std::sqrt(static_cast<double>(structure.energy_weight));
    if (weight == 0.0) {
      continue;
    }
    const double scale = weight / structure.num_atom;
    for (int t = 0; t < n; ++t) {
      row[t] *= scale;
    }
    const double energy = (structure.energy_total / structure.num_atom) * weight;
    if (!std::isfinite(energy)) {
      error = location + " has a weighted energy outside the double range.";
      return false;
    }
    add_reference_row(r, z, row, energy, n);
    ++num_rows;
  }

  std::vector<double> v(n * n, 0.0);
  for (int t = 0; t < n; ++t) {
    v[t * n + t] = 1.0;
  }
  const double relative_cutoff =
    std::numeric_limits<double>::epsilon() * std::max(num_rows, size_t(n));
  if (!orthogonalize_reference_columns(r, v, n, relative_cutoff)) {
    error = "Energy reference fit did not converge.";
    return false;
  }
  std::vector<double> singular_values(n, 0.0);
  for (int j = 0; j < n; ++j) {
    singular_values[j] = reference_column_norm(r, n, j);
  }
  const double largest = *std::max_element(singular_values.begin(), singular_values.end());
  const double cutoff = relative_cutoff * largest;
  std::vector<double> solution(n, 0.0);
  int fitted_rank = 0;
  for (int j = 0; j < n; ++j) {
    const double singular_value = singular_values[j];
    if (singular_value <= cutoff) {
      continue;
    }
    ++fitted_rank;
    double projection = 0.0;
    for (int i = 0; i < n; ++i) {
      projection += (r[i * n + j] / singular_value) * z[i];
    }
    const double coefficient = projection / singular_value;
    for (int t = 0; t < n; ++t) {
      solution[t] += v[t * n + j] * coefficient;
    }
  }
  for (const double value : solution) {
    if (!std::isfinite(value)) {
      error = "Energy reference fit produced a value outside the double range.";
      return false;
    }
  }
  reference.swap(solution);
  rank = fitted_rank;
  return true;
}
