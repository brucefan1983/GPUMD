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
    deposit.cu
    Implementation of the minimal "deposit" command for GPUMD.

    This module performs periodic atom deposition from a specified region
    onto the existing simulation box. Each deposition inserts one atom
    with a fixed velocity vector at uniform random coordinates within the
    given (x, y, z) range.

    Key features:
      - Activated by run.cu via Deposit::compute(step, atom, group)
      - Adds one atom every <interval> steps (no retry / distance check)
      - Updates atom arrays and all group data structures on both host
        and device
      - Prints deposition information to stdout
------------------------------------------------------------------------------*/

#include "deposit.cuh"
#include "utilities/read_file.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<double> append_atom(
  const std::vector<double>& data,
  int N,
  int num_components,
  const double* new_atom_values)
{
  std::vector<double> new_data((N + 1) * num_components);

  for (int comp = 0; comp < num_components; ++comp) {
    const int offset     = N * comp;         
    const int new_offset = (N + 1) * comp;   

    for (int atom_idx = 0; atom_idx < N; ++atom_idx) {
      new_data[new_offset + atom_idx] = data[offset + atom_idx];
    }

    new_data[new_offset + N] = new_atom_values[comp];
  }

  return new_data;
}

static double random_position(double left, double right)
{
  if (left == right) {
    return left;
  }
  const double r = rand() / (RAND_MAX + 1.0);  
  return left + (right - left) * r;
}

void Deposit::zero_total_linear_momentum(
  std::vector<double>& velocity, const std::vector<double>& mass, int number_of_atoms)
{
  if (number_of_atoms <= 0) {
    return;
  }
  double momentum[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < number_of_atoms; ++n) {
    double m = mass[n];
    total_mass += m;
    for (int d = 0; d < 3; ++d) {
      momentum[d] += m * velocity[n + number_of_atoms * d];
    }
  }
  if (total_mass <= 0.0) {
    return;
  }
  for (int d = 0; d < 3; ++d) {
    momentum[d] /= total_mass;
  }
  for (int n = 0; n < number_of_atoms; ++n) {
    for (int d = 0; d < 3; ++d) {
      velocity[n + number_of_atoms * d] -= momentum[d];
    }
  }
}

void Deposit::parse(
  const char** param, int num_param, const Atom& atom, const std::vector<Group>& group)
{
  if (active_) {
    PRINT_INPUT_ERROR("deposit command has already been defined.\n");
  }
  if (!(num_param == 16 || num_param == 19)) {
    PRINT_INPUT_ERROR(
      "deposit requires 15 parameters with an optional group clause: deposit <symbol> x <x1> <x2> y "
      "<y1> <y2> z <z1> <z2> v <vx> <vy> <vz> <interval> [group <group_method> <group_label>].\n");
  }

  symbol_ = param[1];
  if (strcmp(param[2], "x") != 0 || strcmp(param[5], "y") != 0 || strcmp(param[8], "z") != 0 ||
      strcmp(param[11], "v") != 0) {
    PRINT_INPUT_ERROR(
      "deposit syntax should be: deposit <symbol> x <x1> <x2> y <y1> <y2> z <z1> <z2> v <vx> <vy> "
      "<vz> <interval>.\n");
  }

  for (int d = 0; d < 3; ++d) {
    const int base = 3 * d + 3;
    if (!is_valid_real(param[base], &range_[d][0])) {
      PRINT_INPUT_ERROR("deposit region bounds should be real numbers.\n");
    }
    if (!is_valid_real(param[base + 1], &range_[d][1])) {
      PRINT_INPUT_ERROR("deposit region bounds should be real numbers.\n");
    }
    if (range_[d][1] < range_[d][0]) {
      PRINT_INPUT_ERROR("deposit region upper bound should be >= lower bound.\n");
    }
  }

  for (int d = 0; d < 3; ++d) {
    if (!is_valid_real(param[12 + d], &velocity_input_[d])) {
      PRINT_INPUT_ERROR("deposit velocity components should be real numbers.\n");
    }
    velocity_natural_[d] = velocity_input_[d] * TIME_UNIT_CONVERSION;
  }

  if (!is_valid_int(param[15], &interval_)) {
    PRINT_INPUT_ERROR("deposit interval should be an integer.\n");
  }
  if (interval_ <= 0) {
    PRINT_INPUT_ERROR("deposit interval should be a positive integer.\n");
  }

  int reference_index = -1;
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    if (atom.cpu_atom_symbol[n] == symbol_) {
      reference_index = n;
      break;
    }
  }
  if (reference_index < 0) {
    PRINT_INPUT_ERROR("deposit species is not present in the current model.\n");
  }

  type_index_ = atom.cpu_type[reference_index];
  if (type_index_ < 0) {
    PRINT_INPUT_ERROR("deposit failed to determine atom type.\n");
  }
  if (type_index_ >= atom.cpu_type_size.size()) {
    PRINT_INPUT_ERROR("deposit detected inconsistent atom type information.\n");
  }
  mass_ = atom.cpu_mass[reference_index];
  charge_ = atom.cpu_charge[reference_index];

  has_group_target_ = false;
  group_method_target_ = -1;
  group_label_target_ = -1;
  if (num_param == 19) {
    if (strcmp(param[16], "group") != 0) {
      PRINT_INPUT_ERROR("deposit optional clause should start with 'group'.\n");
    }
    if (!is_valid_int(param[17], &group_method_target_)) {
      PRINT_INPUT_ERROR("deposit group method should be an integer.\n");
    }
    if (!is_valid_int(param[18], &group_label_target_)) {
      PRINT_INPUT_ERROR("deposit group label should be an integer.\n");
    }
    if (group_method_target_ < 0 || group_method_target_ >= group.size()) {
      PRINT_INPUT_ERROR("deposit group method should refer to an existing grouping method.\n");
    }
    if (group_method_target_ != 0) {
      PRINT_INPUT_ERROR("deposit currently only supports group method 0.\n");
    }
    if (group[group_method_target_].number <= 0) {
      PRINT_INPUT_ERROR("deposit group method 0 is not available.\n");
    }
    if (group_label_target_ < 0 || group_label_target_ >= group[group_method_target_].number) {
      PRINT_INPUT_ERROR("deposit group label should be within the available range.\n");
    }
    has_group_target_ = true;
  }

  next_step_ = interval_ - 1;
  active_ = true;

  printf("Deposit %s atom every %d steps.\n", symbol_.c_str(), interval_);
  printf("    x in [%g, %g] A.\n", range_[0][0], range_[0][1]);
  printf("    y in [%g, %g] A.\n", range_[1][0], range_[1][1]);
  printf("    z in [%g, %g] A.\n", range_[2][0], range_[2][1]);
  printf(
    "    velocity = (%g, %g, %g) A/fs.\n",
    velocity_input_[0],
    velocity_input_[1],
    velocity_input_[2]);
  if (has_group_target_) {
    printf("    assign to group method %d, label %d.\n", group_method_target_, group_label_target_);
  }
}

void Deposit::compute(int step, Atom& atom, std::vector<Group>& group)
{
  if (!active_) {
    return;
  }
  if (step != next_step_) {
    return;
  }

  const int N = atom.number_of_atoms;
  const int new_N = N + 1;

  if (atom.cpu_position_per_atom.size() != static_cast<size_t>(N * 3)) {
    atom.cpu_position_per_atom.resize(N * 3);
  }
  if (atom.cpu_velocity_per_atom.size() != static_cast<size_t>(N * 3)) {
    atom.cpu_velocity_per_atom.resize(N * 3);
  }
  if (N > 0) {
    atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }

  std::vector<double> force_host(N * 3);
  std::vector<double> virial_host(N * 9);
  std::vector<double> potential_host(N);
  if (N > 0) {
    atom.force_per_atom.copy_to_host(force_host.data());
    atom.virial_per_atom.copy_to_host(virial_host.data());
    atom.potential_per_atom.copy_to_host(potential_host.data());
  }

  double new_position[3];
  for (int d = 0; d < 3; ++d) {
    new_position[d] = random_position(range_[d][0], range_[d][1]);
  }

  std::vector<double> position_new = append_atom(atom.cpu_position_per_atom, N, 3, new_position);
  std::vector<double> velocity_new = append_atom(atom.cpu_velocity_per_atom, N, 3, velocity_natural_);

  double zero_force[3] = {0.0, 0.0, 0.0};
  std::vector<double> force_new = append_atom(force_host, N, 3, zero_force);
  double zero_virial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> virial_new = append_atom(virial_host, N, 9, zero_virial);
  double zero_potential[1] = {0.0};
  std::vector<double> potential_new = append_atom(potential_host, N, 1, zero_potential);

  atom.cpu_position_per_atom = std::move(position_new);
  atom.cpu_velocity_per_atom = std::move(velocity_new);
  atom.number_of_atoms = new_N;

  atom.cpu_type.push_back(type_index_);
  atom.cpu_mass.push_back(mass_);
  atom.cpu_charge.push_back(charge_);
  atom.cpu_atom_symbol.push_back(symbol_);
  atom.cpu_type_size[type_index_] += 1;

  zero_total_linear_momentum(atom.cpu_velocity_per_atom, atom.cpu_mass, new_N);

  atom.position_per_atom.resize(new_N * 3);
  atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());

  atom.velocity_per_atom.resize(new_N * 3);
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());

  atom.force_per_atom.resize(new_N * 3);
  atom.force_per_atom.copy_from_host(force_new.data());

  atom.virial_per_atom.resize(new_N * 9);
  atom.virial_per_atom.copy_from_host(virial_new.data());

  atom.potential_per_atom.resize(new_N);
  atom.potential_per_atom.copy_from_host(potential_new.data());

  atom.type.resize(new_N);
  atom.type.copy_from_host(atom.cpu_type.data());

  atom.mass.resize(new_N);
  atom.mass.copy_from_host(atom.cpu_mass.data());

  atom.charge.resize(new_N);
  atom.charge.copy_from_host(atom.cpu_charge.data());

  for (int gm = 0; gm < group.size(); ++gm) {
    auto& g = group[gm];
    if (g.number <= 0) {
      continue;
    }
    if (g.cpu_label.size() != static_cast<size_t>(N)) {
      g.cpu_label.resize(N);
      if (N > 0 && g.label.size() == static_cast<size_t>(N)) {
        g.label.copy_to_host(g.cpu_label.data());
      }
    }
    int label = 0;
    if (has_group_target_ && gm == group_method_target_) {
      label = group_label_target_;
    } else if (!g.cpu_label.empty()) {
      label = g.cpu_label.back();
    }
    if (label < 0 || label >= g.number) {
      label = g.number - 1;
    }
    g.cpu_label.push_back(label);

    g.cpu_size.assign(g.number, 0);
    for (const int lbl : g.cpu_label) {
      if (lbl < 0 || lbl >= g.number) {
        PRINT_INPUT_ERROR("group label should be within the valid range when depositing atoms.\n");
      }
      g.cpu_size[lbl] += 1;
    }

    g.cpu_size_sum.resize(g.number);
    int cumulative = 0;
    for (int m = 0; m < g.number; ++m) {
      g.cpu_size_sum[m] = cumulative;
      cumulative += g.cpu_size[m];
    }

    g.cpu_contents.resize(new_N);
    std::vector<int> offset(g.number, 0);
    for (int idx = 0; idx < new_N; ++idx) {
      int lbl = g.cpu_label[idx];
      int pos = g.cpu_size_sum[lbl] + offset[lbl];
      g.cpu_contents[pos] = idx;
      offset[lbl] += 1;
    }

    g.label.resize(new_N);
    g.label.copy_from_host(g.cpu_label.data());
    g.size.resize(g.number);
    g.size.copy_from_host(g.cpu_size.data());
    g.size_sum.resize(g.number);
    g.size_sum.copy_from_host(g.cpu_size_sum.data());
    g.contents.resize(new_N);
    g.contents.copy_from_host(g.cpu_contents.data());
  }

  printf(
    "Deposited 1 %s atom at step %d, position = (%g, %g, %g) A.\n",
    symbol_.c_str(),
    step + 1,
    new_position[0],
    new_position[1],
    new_position[2]);
  fflush(stdout);

  next_step_ += interval_;
}

void Deposit::finalize()
{
  active_ = false;
  next_step_ = -1;
  has_group_target_ = false;
  group_method_target_ = -1;
  group_label_target_ = -1;
}
