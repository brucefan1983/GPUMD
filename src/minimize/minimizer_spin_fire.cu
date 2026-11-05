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

/*
Spin minimizer spin fire implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com
*/

#include "force/force.cuh"
#include "minimizer_spin_fire.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr int BLOCK = 256;
// E, |g|^2, |v|^2, g.v, max_i |g_i|^2, invalid state, relative spin update.
constexpr int NUM_STATS = 7;

__global__ void initialize_radius(int n, const double* spin, double* radius)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    radius[i] =
      sqrt(spin[i] * spin[i] + spin[n + i] * spin[n + i] + spin[2 * n + i] * spin[2 * n + i]);
}

__global__ void project_force(
  int n,
  bool fixed,
  const double* spin,
  const double* radius,
  const double* force,
  double* gradient,
  double* velocity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double sg = 0, sv = 0, s2 = 0;
  for (int c = 0; c < 3; ++c) {
    const int j = c * n + i;
    sg += spin[j] * force[j];
    sv += spin[j] * velocity[j];
    s2 += spin[j] * spin[j];
  }
  for (int c = 0; c < 3; ++c) {
    const int j = c * n + i;
    gradient[j] = fixed ? (radius[i] > 0 ? force[j] - spin[j] * sg / s2 : 0.0) : force[j];
    if (fixed)
      velocity[j] = radius[i] > 0 ? velocity[j] - spin[j] * sv / s2 : 0.0;
  }
}

__global__ void collect_stats(
  int n,
  const double* spin,
  const double* force,
  const double* energy,
  const double* gradient,
  const double* velocity,
  const double* previous_spin,
  double* result)
{
  __shared__ double sums[NUM_STATS][BLOCK];
  const int t = threadIdx.x;
  double e = 0, g2 = 0, v2 = 0, power = 0, gmax2 = 0, invalid = 0, change = 0;
  for (int i = t; i < n; i += BLOCK) {
    e += energy[i];
    if (!isfinite(energy[i]))
      invalid = 1;
    double gi2 = 0;
    for (int c = 0; c < 3; ++c) {
      const int j = c * n + i;
      const double g = gradient[j], v = velocity[j];
      if (!isfinite(spin[j]) || !isfinite(force[j]) || !isfinite(g) || !isfinite(v))
        invalid = 1;
      gi2 += g * g;
      v2 += v * v;
      power += g * v;
      const double scale = fmax(1.0, fmax(fabs(spin[j]), fabs(previous_spin[j])));
      change = fmax(change, fabs(spin[j] - previous_spin[j]) / scale);
    }
    g2 += gi2;
    gmax2 = fmax(gmax2, gi2);
  }
  sums[0][t] = e;
  sums[1][t] = g2;
  sums[2][t] = v2;
  sums[3][t] = power;
  sums[4][t] = gmax2;
  sums[5][t] = invalid;
  sums[6][t] = change;
  __syncthreads();
  for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
    if (t < offset) {
      for (int k = 0; k < 4; ++k)
        sums[k][t] += sums[k][t + offset];
      for (int k = 4; k < NUM_STATS; ++k)
        sums[k][t] = fmax(sums[k][t], sums[k][t + offset]);
    }
    __syncthreads();
  }
  if (t == 0)
    for (int k = 0; k < NUM_STATS; ++k)
      result[k] = sums[k][0];
}

__global__ void kick(int n, double dt, const double* gradient, double* velocity)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < 3 * n)
    velocity[j] += dt * gradient[j];
}

__global__ void move_spins(
  int n,
  bool fixed,
  double dt,
  double alpha,
  double ratio,
  double max_step,
  const double* radius,
  const double* gradient,
  double* velocity,
  double* spin)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || (fixed && radius[i] == 0))
    return;
  double v[3], norm2 = 0;
  for (int c = 0; c < 3; ++c) {
    v[c] = (1 - alpha) * velocity[c * n + i] + alpha * ratio * gradient[c * n + i];
    norm2 += v[c] * v[c];
  }
  const double norm = sqrt(norm2);
  if (norm == 0)
    return;
  // Convert the angular limit to an arc length for fixed-magnitude spins.
  const double limit = fixed ? max_step * radius[i] : max_step;
  const double scale = fmin(1.0, limit / (dt * norm));
  if (!fixed) {
    for (int c = 0; c < 3; ++c) {
      velocity[c * n + i] = v[c] * scale;
      spin[c * n + i] += dt * v[c] * scale;
    }
    return;
  }
  // Sphere exponential map, with parallel transport of the fictitious velocity.
  const double theta = dt * norm * scale / radius[i], co = cos(theta), si = sin(theta);
  double next[3], next_norm2 = 0;
  for (int c = 0; c < 3; ++c) {
    const double unit = spin[c * n + i] / radius[i], direction = v[c] / norm;
    next[c] = co * unit + si * direction;
    next_norm2 += next[c] * next[c];
    velocity[c * n + i] = norm * scale * (co * direction - si * unit);
  }
  const double rescale = radius[i] / sqrt(next_norm2);
  for (int c = 0; c < 3; ++c)
    spin[c * n + i] = next[c] * rescale;
}
// Atomic force/velocity statistics and the full (possibly nonsymmetric) virial.
__global__ void collect_lattice_stats(
  int n,
  const double* position,
  const double* previous,
  const double* force,
  const double* velocity,
  const double* virial,
  double* result)
{
  __shared__ double sums[15][BLOCK];
  const int t = threadIdx.x;
  double values[15] = {};
  for (int i = t; i < n; i += BLOCK) {
    for (int c = 0; c < 3; ++c) {
      const int j = c * n + i;
      const double f = force[j], v = velocity[j];
      values[0] += f * f;
      values[1] += v * v;
      values[2] += f * v;
      values[3] = fmax(values[3], fabs(f));
      if (!isfinite(f) || !isfinite(v) || !isfinite(position[j]))
        values[4] = 1;
      const double scale = fmax(1.0, fmax(fabs(position[j]), fabs(previous[j])));
      values[5] = fmax(values[5], fabs(position[j] - previous[j]) / scale);
    }
    for (int c = 0; c < 9; ++c) {
      const double w = virial[c * n + i];
      if (!isfinite(w))
        values[4] = 1;
      values[6 + c] += w;
    }
  }
  for (int k = 0; k < 15; ++k)
    sums[k][t] = values[k];
  __syncthreads();
  for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
    if (t < offset)
      for (int k = 0; k < 15; ++k)
        sums[k][t] = (k >= 3 && k <= 5) ? fmax(sums[k][t], sums[k][t + offset])
                                        : sums[k][t] + sums[k][t + offset];
    __syncthreads();
  }
  if (t == 0)
    for (int k = 0; k < 15; ++k)
      result[k] = sums[k][0];
}

__global__ void move_atoms(
  int n,
  double dt,
  double alpha,
  double ratio,
  double max_step,
  const double* force,
  const double* strain,
  double* velocity,
  double* position)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double r[3], v[3], norm2 = 0;
  for (int c = 0; c < 3; ++c) {
    r[c] = position[c * n + i];
    v[c] = (1 - alpha) * velocity[c * n + i] + alpha * ratio * force[c * n + i];
    norm2 += v[c] * v[c];
  }
  const double scale = norm2 > 0 ? fmin(1.0, max_step / (dt * sqrt(norm2))) : 1.0;
  for (int c = 0; c < 3; ++c) {
    velocity[c * n + i] = v[c] * scale;
    position[c * n + i] = r[c] + dt * v[c] * scale;
    for (int k = 0; k < 3; ++k)
      position[c * n + i] += strain[3 * c + k] * r[k];
  }
}

constexpr double EV_A3_TO_GPA = 160.2176621;
} // namespace

Minimizer_Spin_FIRE::Minimizer_Spin_FIRE(
  const std::vector<std::string>& tokens, int fixed_group, const Atom& atom, const Box& box)
  : Minimizer(-1, 0, atom.number_of_atoms, 0, 0.0)
{
  const int num_param = tokens.size();
  double tolerance;
  int steps, variable_magnitude, box_change = 0, hydrostatic_strain = 0;
  // Omitting optional flags retains the spin-only shorthand.
  int fixed_atoms = 1;
  if (num_param < 5 || num_param > 8)
    PRINT_INPUT_ERROR("minimize spin_fire requires <tolerance> <max_steps> <variable_magnitude: "
                      "0|1> [fixed_atoms: 0|1 [box_change: 0|1 [hydrostatic_strain: 0|1]]].");
  if (!is_valid_real(tokens[2], &tolerance) || !std::isfinite(tolerance) || tolerance <= 0)
    PRINT_INPUT_ERROR("Spin FIRE tolerance must be finite and positive.");
  if (!is_valid_int(tokens[3], &steps) || steps <= 0)
    PRINT_INPUT_ERROR("Spin FIRE max_steps must be a positive integer.");
  if (
    !is_valid_int(tokens[4], &variable_magnitude) ||
    (variable_magnitude != 0 && variable_magnitude != 1))
    PRINT_INPUT_ERROR("Spin FIRE variable_magnitude must be 0 (fixed) or 1 (variable).");
  if (!atom.has_spin)
    PRINT_INPUT_ERROR("Spin FIRE requires a Spin potential and spin:R:3 on every atom.");
  if (
    num_param >= 6 &&
    (!is_valid_int(tokens[5], &fixed_atoms) || (fixed_atoms != 0 && fixed_atoms != 1)))
    PRINT_INPUT_ERROR("Spin FIRE fixed_atoms must be 0 or 1.");
  if (
    num_param >= 7 &&
    (!is_valid_int(tokens[6], &box_change) || (box_change != 0 && box_change != 1)))
    PRINT_INPUT_ERROR("Spin FIRE box_change must be 0 or 1.");
  if (
    num_param == 8 && (!is_valid_int(tokens[7], &hydrostatic_strain) ||
                       (hydrostatic_strain != 0 && hydrostatic_strain != 1)))
    PRINT_INPUT_ERROR("Spin FIRE hydrostatic_strain must be 0 or 1.");
  if (fixed_atoms && box_change)
    PRINT_INPUT_ERROR("Spin FIRE fixed_atoms = 1 requires box_change = 0.");
  if (hydrostatic_strain && !box_change)
    PRINT_INPUT_ERROR("Spin FIRE hydrostatic_strain requires box_change = 1.");
  if (box_change && !(box.pbc_x && box.pbc_y && box.pbc_z))
    PRINT_INPUT_ERROR("Spin FIRE box_change requires periodic boundaries in all directions.");
  if (!fixed_atoms && fixed_group >= 0)
    PRINT_INPUT_ERROR("Joint Spin FIRE does not support fix groups.");
  number_of_steps_ = steps;
  force_tolerance_ = tolerance;
  fixed_magnitude_ = variable_magnitude == 0;
  relax_atoms_ = !fixed_atoms;
  box_change_ = box_change;
  hydrostatic_strain_ = hydrostatic_strain;
}

void Minimizer_Spin_FIRE::compute(
  Force& force, Box& box, Atom& atom, GPU_Vector<double>& position, std::vector<Group>& group)
{
  CHECK(gpuSetDevice(0));
  const int n = number_of_atoms_, blocks = (n + BLOCK - 1) / BLOCK;
  if (n <= 0 || atom.spin_per_atom.size() != 3 * n || atom.mforce_per_atom.size() != 3 * n)
    PRINT_INPUT_ERROR("Spin FIRE requires complete spin and magnetic-force arrays.");
  GPU_Vector<double> radius(n), gradient(3 * n), velocity(3 * n, 0.0), saved(3 * n),
    stats_gpu(NUM_STATS);
  double stats[NUM_STATS];
  GPU_Vector<double> lattice_position, saved_position, lattice_velocity, lattice_stats_gpu,
    strain_gpu;
  if (relax_atoms_) {
    lattice_position.resize(3 * n);
    saved_position.resize(3 * n);
    lattice_velocity.resize(3 * n, 0.0);
    lattice_stats_gpu.resize(15);
    strain_gpu.resize(9, 0.0);
  }
  double lattice_stats[15] = {}, cell_force[9] = {}, cell_velocity[9] = {};
  double max_force = 0, max_stress = 0;
  Box working_box = box, saved_box = box;
  const double cell_scale = std::cbrt(box.get_volume());
  if (box_change_ && (!(cell_scale > 0) || !std::isfinite(cell_scale)))
    PRINT_INPUT_ERROR("Spin FIRE requires a finite, nonsingular cell.");
  if (relax_atoms_) {
    atom.enable_unwrapped_position();
    atom.update_unwrapped_position(box);
    atom.unwrapped_position.copy_to_device(lattice_position.data());
    lattice_position.copy_to_device(saved_position.data());
  }
  // Force dispatch wraps its position buffer. Use a private copy to preserve the
  // caller's positions, periodic images and lattice velocities exactly.
  position.copy_to_device(position_per_atom_temp_.data());
  initialize_radius<<<blocks, BLOCK>>>(n, atom.spin_per_atom.data(), radius.data());
  GPU_CHECK_KERNEL
  GPU_Vector<double> original_position(3 * n);
  GPU_Vector<int> original_images;
  if (atom.position_image.size())
    original_images.resize(atom.position_image.size());
  auto evaluate = [&]() {
    if (relax_atoms_)
      lattice_position.copy_to_device(position_per_atom_temp_.data());
    // Atom owns the public evaluation state. Save and restore the caller's
    // coordinates and images while evaluating the trial configuration.
    original_position.copy_from_device(atom.position_per_atom.data());
    if (atom.position_image.size())
      original_images.copy_from_device(atom.position_image.data());
    atom.position_per_atom.copy_from_device(position_per_atom_temp_.data());
    if (atom.position_image.size())
      atom.position_image.fill(0);
    force.compute(working_box, atom, group);
    position_per_atom_temp_.copy_from_device(atom.position_per_atom.data());
    atom.position_per_atom.copy_from_device(original_position.data());
    if (atom.position_image.size())
      atom.position_image.copy_from_device(original_images.data());
    CHECK(gpuSetDevice(0));
    project_force<<<blocks, BLOCK>>>(
      n,
      fixed_magnitude_,
      atom.spin_per_atom.data(),
      radius.data(),
      atom.mforce_per_atom.data(),
      gradient.data(),
      velocity.data());
    GPU_CHECK_KERNEL
  };
  auto measure = [&]() {
    collect_stats<<<1, BLOCK>>>(
      n,
      atom.spin_per_atom.data(),
      atom.mforce_per_atom.data(),
      atom.potential_per_atom.data(),
      gradient.data(),
      velocity.data(),
      saved.data(),
      stats_gpu.data());
    GPU_CHECK_KERNEL
    stats_gpu.copy_to_host(stats);
    if (relax_atoms_) {
      collect_lattice_stats<<<1, BLOCK>>>(
        n,
        lattice_position.data(),
        saved_position.data(),
        atom.force_per_atom.data(),
        lattice_velocity.data(),
        atom.virial_per_atom.data(),
        lattice_stats_gpu.data());
      GPU_CHECK_KERNEL
      lattice_stats_gpu.copy_to_host(lattice_stats);
      for (double value : lattice_stats)
        if (!std::isfinite(value))
          stats[5] = 1;
      stats[1] += lattice_stats[0];
      stats[2] += lattice_stats[1];
      stats[3] += lattice_stats[2];
      stats[5] = std::max(stats[5], lattice_stats[4]);
      stats[6] = std::max(stats[6], lattice_stats[5]);
      max_force = lattice_stats[3];
      max_stress = 0;
      for (int k = 0; k < 9; ++k)
        max_stress = std::max(
          max_stress, std::abs(lattice_stats[6 + k]) / working_box.get_volume() * EV_A3_TO_GPA);
      if (box_change_) {
        // W_ab = -sum r_a dE/dr_b. For r'=(I+eps)r, -dE/deps_ab=W_ba.
        // Transpose explicitly: spin-orbit interactions need not give symmetric W.
        const int transpose_order[9] = {0, 6, 7, 3, 1, 8, 4, 5, 2};
        const double trace = (lattice_stats[6] + lattice_stats[7] + lattice_stats[8]) / 3;
        max_stress = 0;
        for (int k = 0; k < 9; ++k) {
          const double w =
            hydrostatic_strain_ ? (k % 4 == 0 ? trace : 0) : lattice_stats[6 + transpose_order[k]];
          max_stress = std::max(max_stress, std::abs(w) / working_box.get_volume() * EV_A3_TO_GPA);
          cell_force[k] = w / cell_scale;
          stats[1] += cell_force[k] * cell_force[k];
          stats[2] += cell_velocity[k] * cell_velocity[k];
          stats[3] += cell_force[k] * cell_velocity[k];
          stats[6] = std::max(
            stats[6],
            std::abs(working_box.cpu_h[k] - saved_box.cpu_h[k]) /
              std::max(
                1.0, std::max(std::abs(working_box.cpu_h[k]), std::abs(saved_box.cpu_h[k]))));
        }
      }
    }
    for (int k = 0; k < NUM_STATS; ++k)
      if (!std::isfinite(stats[k]))
        stats[5] = 1;
  };
  auto report = [&](int step, double dt) {
    printf(
      "spin_fire step %d energy %.16e max_mforce %.9e dt %.6e", step, stats[0], sqrt(stats[4]), dt);
    if (relax_atoms_)
      printf(" max_force %.9e max_stress_GPa %.9e", max_force, max_stress);
    printf("\n");
  };
  auto residual = [&]() {
    double result = sqrt(stats[4]) / force_tolerance_;
    if (relax_atoms_)
      result = std::max(result, max_force / force_tolerance_);
    if (box_change_)
      result = std::max(result, max_stress / stress_tolerance_gpa);
    return result;
  };
  auto reset_velocity = [&]() {
    velocity.fill(0.0);
    if (relax_atoms_)
      lattice_velocity.fill(0.0);
    for (double& v : cell_velocity)
      v = 0;
  };
  double dt = dt_0, alpha = alpha_start;
  const double dt_max = relax_atoms_ ? dt_max_joint : dt_max_spin;
  int positive = 0, nonfinite_retries = 0, power_resets = 0, steps = 0, tiny_updates = 0;
  const int interval = std::max(1, number_of_steps_ / 20);
  atom.spin_per_atom.copy_to_device(saved.data());
  evaluate();
  measure();
  if (stats[5])
    PRINT_INPUT_ERROR("Spin FIRE encountered a non-finite initial state.");
  printf(
    "\nSpin FIRE: %s magnitude; %s; tolerance %.9e eV/mu_B.\n",
    fixed_magnitude_ ? "fixed" : "variable",
    relax_atoms_ ? (box_change_ ? "joint atoms/spins/cell" : "joint atoms/spins; fixed cell")
                 : "fixed lattice",
    force_tolerance_);
  if (relax_atoms_)
    printf(
      "Atomic force tolerance %.9e eV/A; cell stress tolerance %.9e GPa; hydrostatic_strain %d.\n",
      force_tolerance_,
      stress_tolerance_gpa,
      int(hydrostatic_strain_));
  report(0, dt);
  double progress_residual = residual();
  int last_progress = 0;
  const char* reason = "maximum_steps";
  const char* stagnation = "none";
  for (; steps < number_of_steps_;) {
    if (residual() <= 1) {
      reason = "converged";
      break;
    }
    if (dt < dt_min) {
      reason = "stalled";
      stagnation = "time_step_floor";
      break;
    }
    atom.spin_per_atom.copy_to_device(saved.data());
    if (relax_atoms_)
      lattice_position.copy_to_device(saved_position.data());
    saved_box = working_box;
    ++steps;
    // Semi-implicit Euler FIRE: kick, test power, mix and move. Energy is
    // diagnostic only: adding a constant to the potential must not affect FIRE.
    kick<<<(3 * n + BLOCK - 1) / BLOCK, BLOCK>>>(n, dt, gradient.data(), velocity.data());
    GPU_CHECK_KERNEL
    if (relax_atoms_) {
      kick<<<(3 * n + BLOCK - 1) / BLOCK, BLOCK>>>(
        n, dt, atom.force_per_atom.data(), lattice_velocity.data());
      GPU_CHECK_KERNEL
      for (int k = 0; k < 9; ++k)
        cell_velocity[k] += dt * cell_force[k];
    }
    measure();
    if (stats[5])
      PRINT_INPUT_ERROR("Spin FIRE encountered non-finite fictitious velocities.");
    if (stats[3] <= 0) {
      reset_velocity();
      dt *= f_dec;
      alpha = alpha_start;
      positive = 0;
      ++power_resets;
    } else {
      const double ratio = sqrt(stats[2] / stats[1]);
      if (relax_atoms_) {
        double strain[9] = {}, strain2 = 0;
        if (box_change_) {
          for (int k = 0; k < 9; ++k) {
            cell_velocity[k] = (1 - alpha) * cell_velocity[k] + alpha * ratio * cell_force[k];
            strain[k] = dt * cell_velocity[k] / cell_scale;
            strain2 += strain[k] * strain[k];
          }
          const double scale = strain2 > 0 ? std::min(1.0, max_cell_strain / sqrt(strain2)) : 1.0;
          for (int k = 0; k < 9; ++k) {
            strain[k] *= scale;
            cell_velocity[k] *= scale;
          }
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
              working_box.cpu_h[3 * i + j] = saved_box.cpu_h[3 * i + j];
              for (int k = 0; k < 3; ++k)
                working_box.cpu_h[3 * i + j] += strain[3 * i + k] * saved_box.cpu_h[3 * k + j];
            }
          working_box.get_inverse();
          working_box.set_is_orthogonal();
        }
        strain_gpu.copy_from_host(strain);
        move_atoms<<<blocks, BLOCK>>>(
          n,
          dt,
          alpha,
          ratio,
          max_atom_displacement,
          atom.force_per_atom.data(),
          strain_gpu.data(),
          lattice_velocity.data(),
          lattice_position.data());
        GPU_CHECK_KERNEL
      }
      move_spins<<<blocks, BLOCK>>>(
        n,
        fixed_magnitude_,
        dt,
        alpha,
        sqrt(stats[2] / stats[1]),
        fixed_magnitude_ ? max_spin_angle : max_spin_displacement,
        radius.data(),
        gradient.data(),
        velocity.data(),
        atom.spin_per_atom.data());
      GPU_CHECK_KERNEL
      evaluate();
      measure();
      if (stats[5]) {
        saved.copy_to_device(atom.spin_per_atom.data());
        if (relax_atoms_)
          saved_position.copy_to_device(lattice_position.data());
        working_box = saved_box;
        reset_velocity();
        dt *= f_dec;
        alpha = alpha_start;
        positive = 0;
        ++nonfinite_retries;
        tiny_updates = 0;
        // Restore all endpoint caches as well as spins, including magnetic forces.
        evaluate();
        measure();
        if (stats[5])
          PRINT_INPUT_ERROR("Spin FIRE could not restore a finite state.");
      } else {
        tiny_updates =
          stats[6] <= tiny_update_epsilon_factor * std::numeric_limits<double>::epsilon()
            ? tiny_updates + 1
            : 0;
        if (++positive > N_min) {
          dt = std::min(dt * f_inc, dt_max);
          alpha *= f_alpha;
        }
      }
    }
    const double current_residual = residual();
    // Accumulated improvement counts, rather than requiring a 1% improvement
    // on each step. This conservative limit also catches cycling at a model's
    // force precision floor; it never relaxes the requested force tolerance.
    if (current_residual < residual_progress_ratio * progress_residual) {
      progress_residual = current_residual;
      last_progress = steps;
    }
    if (steps % interval == 0)
      report(steps, dt);
    if (current_residual <= 1) {
      reason = "converged";
      break;
    }
    if (tiny_updates >= max_tiny_updates || steps - last_progress >= max_steps_without_progress) {
      reason = "stalled";
      stagnation = tiny_updates >= max_tiny_updates
                     ? (relax_atoms_ ? "unresolved_updates" : "unresolved_spin_updates")
                     : "no_residual_progress";
      break;
    }
  }
  if (residual() <= 1)
    reason = "converged";
  report(steps, dt);
  printf(
    "Spin FIRE finished: %s; steps %d; power_resets %d; nonfinite_retries %d; stagnation %s.\n",
    reason,
    steps,
    power_resets,
    nonfinite_retries,
    stagnation);
  if (relax_atoms_) {
    box = working_box;
    lattice_position.copy_to_device(position.data());
    // Publish unwrapped optimization coordinates, then rebuild periodic images.
    // Physical atom/spin velocities have never been used as FIRE velocities.
    if (atom.position_image.size())
      atom.position_image.fill(0);
    force.compute(box, atom, group);
    CHECK(gpuSetDevice(0));
    atom.update_unwrapped_position(box);
    position.copy_to_host(atom.cpu_position_per_atom.data());
  }
  // Keep host-side spin state consistent for later commands. Physical velocities
  // are not FIRE's fictitious velocities and remain unchanged.
  atom.cpu_spin_per_atom.resize(3 * n);
  atom.spin_per_atom.copy_to_host(atom.cpu_spin_per_atom.data());
}
