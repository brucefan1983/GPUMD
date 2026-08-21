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

/*----------------------------------------------------------------------------80
The Eco-PIMD frequencies minimise Eq. (17) of

Z. Zeng and D. E. Manolopoulos, "Economised path integrals",
arXiv:2607.06414 (2026).

The shifted Newton step and line search follow the reference Fortran program
supplied with that paper. The symmetric Hessian is diagonalised with GPUMD's
existing cuSOLVER/hipSOLVER wrapper rather than LAPACK DSYEV. A small-x series
avoids cancellation in the objective kernel, and a relative convergence test
stops iterations in the flat valley of an already-converged fit.
------------------------------------------------------------------------------*/

#include "eco_pimd.cuh"
#include <cstddef>
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace
{
struct Objective
{
  double value = 0.0;
  std::vector<double> gradient;
  std::vector<double> hessian;
};

int matrix_index(const int row, const int column, const int size)
{
  return row + column * size;
}

Objective evaluate_objective(
  const std::vector<double>& target,
  const std::vector<double>& grid,
  const std::vector<double>& weights,
  const std::vector<double>& frequencies)
{
  const int number_of_grid_points = static_cast<int>(grid.size());
  const int number_of_modes = static_cast<int>(frequencies.size());

  Objective objective;
  objective.gradient.assign(number_of_modes, 0.0);
  objective.hessian.assign(number_of_modes * number_of_modes, 0.0);

  std::vector<double> first_derivatives(number_of_modes, 0.0);
  std::vector<double> second_derivatives(number_of_modes, 0.0);

  for (int j = 0; j < number_of_grid_points; ++j) {
    const double x_squared = grid[j] * grid[j];
    double residual = -1.0;

    for (int k = 0; k < number_of_modes; ++k) {
      const double denominator_inverse =
        1.0 / (frequencies[k] * frequencies[k] + x_squared);
      const double contribution = target[j] * weights[k] * denominator_inverse;
      residual += contribution;
      first_derivatives[k] =
        -2.0 * denominator_inverse * contribution * frequencies[k];
      second_derivatives[k] = 2.0 * denominator_inverse * denominator_inverse * contribution *
                              (3.0 * frequencies[k] * frequencies[k] - x_squared);
    }

    objective.value += 0.5 * residual * residual;

    for (int k = 0; k < number_of_modes; ++k) {
      objective.gradient[k] += residual * first_derivatives[k];
      for (int i = 0; i <= k; ++i) {
        objective.hessian[matrix_index(i, k, number_of_modes)] +=
          first_derivatives[i] * first_derivatives[k];
      }
      objective.hessian[matrix_index(k, k, number_of_modes)] +=
        residual * second_derivatives[k];
    }
  }

  const double normalization = 1.0 / static_cast<double>(number_of_grid_points);
  objective.value *= normalization;
  for (int k = 0; k < number_of_modes; ++k) {
    objective.gradient[k] *= normalization;
    for (int i = 0; i <= k; ++i) {
      const int upper_index = matrix_index(i, k, number_of_modes);
      objective.hessian[upper_index] *= normalization;
      objective.hessian[matrix_index(k, i, number_of_modes)] =
        objective.hessian[upper_index];
    }
  }

  return objective;
}

void validate_eigendecomposition(
  const std::vector<double>& hessian,
  const std::vector<double>& eigenvalues,
  const std::vector<double>& eigenvectors)
{
  const int size = static_cast<int>(eigenvalues.size());
  double hessian_norm_squared = 0.0;
  double residual_norm_squared = 0.0;
  double orthogonality_error_squared = 0.0;

  for (int i = 0; i < size; ++i) {
    if (!std::isfinite(eigenvalues[i])) {
      PRINT_INPUT_ERROR("Non-finite eigenvalue in the Eco-PIMD Newton step.");
    }
    if (i > 0 && eigenvalues[i] < eigenvalues[i - 1]) {
      PRINT_INPUT_ERROR("Unordered eigenvalues in the Eco-PIMD Newton step.");
    }

    for (int j = 0; j < size; ++j) {
      const double hessian_element = hessian[matrix_index(i, j, size)];
      const double eigenvector_element = eigenvectors[matrix_index(i, j, size)];
      if (!std::isfinite(eigenvector_element)) {
        PRINT_INPUT_ERROR("Non-finite eigenvector in the Eco-PIMD Newton step.");
      }
      hessian_norm_squared += hessian_element * hessian_element;

      double hessian_times_vector = 0.0;
      for (int k = 0; k < size; ++k) {
        hessian_times_vector +=
          hessian[matrix_index(i, k, size)] * eigenvectors[matrix_index(k, j, size)];
      }
      const double residual =
        hessian_times_vector - eigenvalues[j] * eigenvector_element;
      residual_norm_squared += residual * residual;
    }
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      double overlap = 0.0;
      for (int k = 0; k < size; ++k) {
        overlap += eigenvectors[matrix_index(k, i, size)] *
                   eigenvectors[matrix_index(k, j, size)];
      }
      if (i == j) {
        overlap -= 1.0;
      }
      orthogonality_error_squared += overlap * overlap;
    }
  }

  const double residual_error =
    std::sqrt(residual_norm_squared) / std::max(1.0, std::sqrt(hessian_norm_squared));
  const double orthogonality_error =
    std::sqrt(orthogonality_error_squared) / std::max(1, size);
  if (residual_error > 1.0e-8 || orthogonality_error > 1.0e-8) {
    PRINT_INPUT_ERROR("Inaccurate eigendecomposition in the Eco-PIMD Newton step.");
  }
}

std::vector<double> find_newton_direction(const Objective& objective)
{
  const int number_of_modes = static_cast<int>(objective.gradient.size());
  std::vector<double> eigenvalues(number_of_modes, 0.0);
  std::vector<double> eigenvectors(number_of_modes * number_of_modes, 0.0);
  std::vector<double> hessian_for_solver = objective.hessian;

  eigenvectors_symmetric_Jacobi(
    number_of_modes,
    hessian_for_solver.data(),
    eigenvalues.data(),
    eigenvectors.data());
  validate_eigendecomposition(objective.hessian, eigenvalues, eigenvectors);

  std::vector<double> gradient_in_eigenbasis(number_of_modes, 0.0);
  for (int column = 0; column < number_of_modes; ++column) {
    for (int row = 0; row < number_of_modes; ++row) {
      gradient_in_eigenbasis[column] +=
        eigenvectors[matrix_index(row, column, number_of_modes)] * objective.gradient[row];
    }
  }

  const double shift =
    std::max(1.0e-16 * eigenvalues.back(), -2.0 * eigenvalues.front());
  for (int k = 0; k < number_of_modes; ++k) {
    const double shifted_eigenvalue = eigenvalues[k] + shift;
    if (!(shifted_eigenvalue > 0.0) || !std::isfinite(shifted_eigenvalue)) {
      PRINT_INPUT_ERROR("Invalid shifted Hessian eigenvalue in the Eco-PIMD Newton step.");
    }
    gradient_in_eigenbasis[k] /= -shifted_eigenvalue;
  }

  std::vector<double> direction(number_of_modes, 0.0);
  for (int row = 0; row < number_of_modes; ++row) {
    for (int column = 0; column < number_of_modes; ++column) {
      direction[row] += eigenvectors[matrix_index(row, column, number_of_modes)] *
                        gradient_in_eigenbasis[column];
    }
  }
  return direction;
}

bool take_newton_step(
  const std::vector<double>& target,
  const std::vector<double>& grid,
  const std::vector<double>& weights,
  std::vector<double>& frequencies,
  Objective& objective)
{
  const int number_of_modes = static_cast<int>(frequencies.size());
  const std::vector<double> direction = find_newton_direction(objective);

  double directional_derivative = 0.0;
  double gradient_norm_squared = 0.0;
  double direction_norm_squared = 0.0;
  for (int k = 0; k < number_of_modes; ++k) {
    directional_derivative += objective.gradient[k] * direction[k];
    gradient_norm_squared += objective.gradient[k] * objective.gradient[k];
    direction_norm_squared += direction[k] * direction[k];
  }
  const double descent_tolerance =
    1.0e-12 * std::sqrt(gradient_norm_squared * direction_norm_squared);
  if (directional_derivative > descent_tolerance) {
    PRINT_INPUT_ERROR("Eco-PIMD Newton direction is not a descent direction.");
  }

  double step_size = 2.0;
  for (int iteration = 0; iteration < 60; ++iteration) {
    step_size *= 0.5;
    std::vector<double> trial_frequencies(number_of_modes, 0.0);
    for (int k = 0; k < number_of_modes; ++k) {
      trial_frequencies[k] = frequencies[k] + step_size * direction[k];
    }

    if (!(trial_frequencies.front() > 0.0) ||
        !std::is_sorted(trial_frequencies.begin(), trial_frequencies.end())) {
      continue;
    }

    Objective trial_objective =
      evaluate_objective(target, grid, weights, trial_frequencies);
    if (trial_objective.value <= objective.value) {
      frequencies.swap(trial_frequencies);
      objective = std::move(trial_objective);
      return true;
    }
  }

  return false;
}

void validate_initial_frequencies(
  const std::vector<double>& frequencies, const int number_of_modes)
{
  if (static_cast<int>(frequencies.size()) != number_of_modes) {
    PRINT_INPUT_ERROR("Incorrect number of initial frequencies for Eco-PIMD.");
  }
  if (!(frequencies.front() > 0.0) ||
      !std::is_sorted(frequencies.begin(), frequencies.end())) {
    PRINT_INPUT_ERROR("Eco-PIMD initial frequencies must be positive and ordered.");
  }
  for (const double frequency : frequencies) {
    if (!std::isfinite(frequency)) {
      PRINT_INPUT_ERROR("Eco-PIMD initial frequencies must be finite.");
    }
  }
}
} // namespace

Eco_PIMD_Result find_eco_pimd_frequencies(
  const int number_of_beads,
  const double x_max,
  const std::vector<double>& initial_frequencies)
{
  if (number_of_beads < 2) {
    PRINT_INPUT_ERROR("Eco-PIMD requires at least two beads.");
  }
  if (!(x_max > 0.0) || !std::isfinite(x_max)) {
    PRINT_INPUT_ERROR("Eco-PIMD requires a positive finite x_max.");
  }

  const int number_of_modes = number_of_beads / 2;
  const int number_of_grid_points =
    std::max(100, static_cast<int>(std::floor(10.0 * x_max + 0.5)));
  const double grid_spacing = x_max / static_cast<double>(number_of_grid_points);
  const double pi = std::acos(-1.0);

  std::vector<double> grid(number_of_grid_points, 0.0);
  std::vector<double> target(number_of_grid_points, 0.0);
  for (int j = 0; j < number_of_grid_points; ++j) {
    grid[j] = (static_cast<double>(j) + 0.5) * grid_spacing;
    const double half_x = 0.5 * grid[j];
    if (half_x < 0.25) {
      const double half_x_squared = half_x * half_x;
      target[j] = 4.0 /
                  (1.0 / 3.0 - half_x_squared / 45.0 +
                   2.0 * half_x_squared * half_x_squared / 945.0 -
                   half_x_squared * half_x_squared * half_x_squared / 4725.0);
    } else {
      target[j] = grid[j] * grid[j] / (half_x / std::tanh(half_x) - 1.0);
    }
  }

  std::vector<double> weights(number_of_modes, 2.0);
  if (2 * number_of_modes == number_of_beads) {
    weights.back() = 1.0;
  }

  Eco_PIMD_Result result;
  std::vector<double> frequencies(number_of_modes, 0.0);

  for (int k = 0; k < number_of_modes; ++k) {
    frequencies[k] = 2.0 * static_cast<double>(number_of_beads) *
                     std::sin(static_cast<double>(k + 1) * pi / number_of_beads);
  }
  Objective trotter_objective = evaluate_objective(target, grid, weights, frequencies);
  result.rmse_trotter = std::sqrt(2.0 * trotter_objective.value);

  for (int k = 0; k < number_of_modes; ++k) {
    frequencies[k] = 2.0 * static_cast<double>(k + 1) * pi;
  }
  Objective matsubara_objective = evaluate_objective(target, grid, weights, frequencies);
  result.rmse_matsubara = std::sqrt(2.0 * matsubara_objective.value);

  if (!initial_frequencies.empty()) {
    frequencies = initial_frequencies;
    validate_initial_frequencies(frequencies, number_of_modes);
  }
  Objective objective = evaluate_objective(target, grid, weights, frequencies);

  bool converged = false;
  for (int iteration = 1; iteration <= 500; ++iteration) {
    const double previous_value = objective.value;
    if (!take_newton_step(target, grid, weights, frequencies, objective)) {
      break;
    }
    result.number_of_iterations = iteration;
    if (previous_value - objective.value <= 1.0e-12 * previous_value) {
      converged = true;
      break;
    }
  }

  double maximum_gradient = 0.0;
  for (const double gradient : objective.gradient) {
    maximum_gradient = std::max(maximum_gradient, std::fabs(gradient));
  }
  if (!converged && objective.value > 1.0e-8 && maximum_gradient > 1.0e-6) {
    PRINT_INPUT_ERROR("Eco-PIMD frequency optimisation did not converge.");
  }

  result.rmse_eco = std::sqrt(2.0 * objective.value);
  result.independent_frequencies = frequencies;
  result.mode_factors.assign(number_of_beads, 0.0);
  for (int k = 1; k <= number_of_modes; ++k) {
    result.mode_factors[k] = frequencies[k - 1] / static_cast<double>(number_of_beads);
  }
  for (int k = 1; k <= (number_of_beads - 1) / 2; ++k) {
    result.mode_factors[number_of_beads - k] = result.mode_factors[k];
  }

  return result;
}
