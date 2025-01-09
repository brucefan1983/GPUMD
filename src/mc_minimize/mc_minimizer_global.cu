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
#pragma once

#include "mc_minimizer_global.cuh"

MC_Minimizer_Global::MC_Minimizer_Global(
  const char** param, int num_param,
  double temperature_input,
  double force_tolerance_input,
  int max_relax_steps_input)
  : MC_Minimizer(param, num_param)
{
  temperature = temperature_input;
  force_tolerance = force_tolerance_input;
  max_relax_steps = max_relax_steps_input;
}

MC_Minimizer_Global::~MC_Minimizer_Global()
{
  //default destructor
}


// a kernel with a single thread <<<1, 1>>>
static __global__ void exchange(
  const int i,
  const int j,
  const int type_i,
  const int type_j,
  int* g_type,
  double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  g_type[i] = type_j;
  g_type[j] = type_i;

  double mass_i = g_mass[i];
  g_mass[i] = g_mass[j];
  g_mass[j] = mass_i;

  double vx_i = g_vx[i];
  g_vx[i] = g_vx[j];
  g_vx[j] = vx_i;

  double vy_i = g_vy[i];
  g_vy[i] = g_vy[j];
  g_vy[j] = vy_i;

  double vz_i = g_vz[i];
  g_vz[i] = g_vz[j];
  g_vz[j] = vz_i;
}

void MC_Minimizer_Global::compute(
  int trials,
  Force& force,
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  int grouping_method,
  int group_id)
{
  //get the swap index
  int group_size =
    grouping_method >= 0 ? group[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  std::uniform_int_distribution<int> r1(0, group_size - 1);

  int num_accepted = 0;
  int N = atom.number_of_atoms;
  for (int step = 0; step < trials; ++step) {

    int i = grouping_method >= 0
              ? group[grouping_method]
                  .cpu_contents[group[grouping_method].cpu_size_sum[group_id] + r1(rng)]
              : r1(rng);
    int type_i = atom.cpu_type[i];
    int j = 0, type_j = type_i;
    while (type_i == type_j) {
      j = grouping_method >= 0
            ? group[grouping_method]
                .cpu_contents[group[grouping_method].cpu_size_sum[group_id] + r1(rng)]
            : r1(rng);
      type_j = atom.cpu_type[j];
    }
  //initialize
  if (step == 0)
  {
    energy_last_step = 0;
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom);
    std::vector<double> pe_before_cpu(N);
    atom.potential_per_atom.copy_to_host(pe_before_cpu.data(), N);
    for (int n = 0; n < N; ++n) {
      energy_last_step += pe_before_cpu[n];
    }
  }

  //construct a copy
  Atom atom_copy;
  atom_copy.number_of_atoms = N;
  atom_copy.mass.resize(N);
  atom_copy.type.resize(N);
  atom_copy.potential_per_atom.resize(N, 0);
  atom_copy.force_per_atom.resize(3 * N, 0);
  atom_copy.velocity_per_atom.resize(3 * N, 0);
  atom_copy.position_per_atom.resize(3 * N);
  atom_copy.virial_per_atom.resize(9 * N, 0);

  atom_copy.mass.copy_from_device(atom.mass.data(), N);
  atom_copy.type.copy_from_device(atom.type.data(), N);
  atom_copy.position_per_atom.copy_from_device(atom.position_per_atom.data(), 3 * N);

  //calculate the energy after swap
  exchange<<<1, 1>>>(
    i,
    j,
    type_i,
    type_j,
    atom_copy.type.data(),
    atom_copy.mass.data(),
    atom_copy.velocity_per_atom.data(),
    atom_copy.velocity_per_atom.data() + N,
    atom_copy.velocity_per_atom.data() + N * 2);

  Minimizer_FIRE minimizer(N, max_relax_steps, force_tolerance);
  minimizer.compute(
    force,
    box,
    atom_copy.position_per_atom,
    atom_copy.type,
    group,
    atom_copy.potential_per_atom,
    atom_copy.force_per_atom,
    atom_copy.virial_per_atom);
  double pe_after_total = 0;
  std::vector<double> pe_after_cpu(N);
  atom.potential_per_atom.copy_to_host(pe_after_cpu.data(), N);
  for (int n = 0; n < N; ++n) {
    pe_after_total += pe_after_cpu[n];
  }

  double energy_difference = pe_after_total - energy_last_step;
  std::uniform_real_distribution<float> r2(0, 1);
  float random_number = r2(rng);
  double probability = exp(-energy_difference / (K_B * temperature));

  if (random_number < probability) {
    ++num_accepted;

    atom.cpu_type[i] = type_j;
    atom.cpu_type[j] = type_i;

    auto atom_symbol_i = atom.cpu_atom_symbol[i];
    atom.cpu_atom_symbol[i] = atom.cpu_atom_symbol[j];
    atom.cpu_atom_symbol[j] = atom_symbol_i;

    double mass_i = atom.cpu_mass[i];
    atom.cpu_mass[i] = atom.cpu_mass[j];
    atom.cpu_mass[j] = mass_i;

    atom.position_per_atom.copy_from_device(atom_copy.position_per_atom.data(), 3 * N);
    atom.potential_per_atom.copy_from_device(atom_copy.potential_per_atom.data(), N);

    exchange<<<1, 1>>>(
      i,
      j,
      type_i,
      type_j,
      atom.type.data(),
      atom.mass.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + N * 2);

    mc_output << step << "\t" << energy_last_step << "\t" << pe_after_total << "\t" << num_accepted / (double(step) + 1) << std::endl;

    energy_last_step = pe_after_total;
    }
  }
}