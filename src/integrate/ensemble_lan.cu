/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
The Bussi-Parrinello integrator of the Langevin thermostat:
[1] G. Bussi and M. Parrinello, Phys. Rev. E 75, 056707 (2007).
------------------------------------------------------------------------------*/

#include "ensemble_lan.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include <cstdlib>

Ensemble_LAN::Ensemble_LAN(int t, int fg, int N, double T, double Tc)
{
  type = t;
  fixed_group = fg;
  temperature = T;
  temperature_coupling = Tc;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * T);
  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  CUDA_CHECK_KERNEL
}

Ensemble_LAN::Ensemble_LAN(
  int t,
  int fg,
  int source_input,
  int sink_input,
  int source_size,
  int sink_size,
  int source_offset,
  int sink_offset,
  double T,
  double Tc,
  double dT)
{
  type = t;
  fixed_group = fg;
  temperature = T;
  temperature_coupling = Tc;
  delta_temperature = dT;
  source = source_input;
  sink = sink_input;
  N_source = source_size;
  N_sink = sink_size;
  offset_source = source_offset;
  offset_sink = sink_offset;
  c1 = exp(-0.5 / temperature_coupling);
  c2_source = sqrt((1 - c1 * c1) * K_B * (T + dT));
  c2_sink = sqrt((1 - c1 * c1) * K_B * (T - dT));
  curand_states_source.resize(N_source);
  curand_states_sink.resize(N_sink);
  int grid_size_source = (N_source - 1) / 128 + 1;
  int grid_size_sink = (N_sink - 1) / 128 + 1;
  initialize_curand_states<<<grid_size_source, 128>>>(
    curand_states_source.data(), N_source, rand());
  CUDA_CHECK_KERNEL
  initialize_curand_states<<<grid_size_sink, 128>>>(curand_states_sink.data(), N_sink, rand());
  CUDA_CHECK_KERNEL
  energy_transferred[0] = 0.0;
  energy_transferred[1] = 0.0;
}

Ensemble_LAN::~Ensemble_LAN(void)
{
  // nothing
}

// wrapper of the global Langevin thermostatting kernels
void Ensemble_LAN::integrate_nvt_lan_half(
  const GPU_Vector<double>& mass, GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  gpu_langevin<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(), number_of_atoms, c1, c2, mass.data(), velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  gpu_find_momentum<<<4, 1024>>>(
    number_of_atoms, mass.data(), velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  gpu_correct_momentum<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL
}

// wrapper of the local Langevin thermostatting kernels
void Ensemble_LAN::integrate_heat_lan_half(
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  int Ng = group[0].number;

  std::vector<double> ek2(Ng);
  GPU_Vector<double> ke(Ng);

  find_ke<<<Ng, 512>>>(
    group[0].size.data(), group[0].size_sum.data(), group[0].contents.data(), mass.data(),
    velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, ke.data());
  CUDA_CHECK_KERNEL

  ke.copy_to_host(ek2.data());
  energy_transferred[0] += ek2[source] * 0.5;
  energy_transferred[1] += ek2[sink] * 0.5;

  gpu_langevin<<<(N_source - 1) / 128 + 1, 128>>>(
    curand_states_source.data(), N_source, offset_source, group[0].contents.data(), c1, c2_source,
    mass.data(), velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  gpu_langevin<<<(N_sink - 1) / 128 + 1, 128>>>(
    curand_states_sink.data(), N_sink, offset_sink, group[0].contents.data(), c1, c2_sink,
    mass.data(), velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  find_ke<<<Ng, 512>>>(
    group[0].size.data(), group[0].size_sum.data(), group[0].contents.data(), mass.data(),
    velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, ke.data());
  CUDA_CHECK_KERNEL

  ke.copy_to_host(ek2.data());
  energy_transferred[0] -= ek2[source] * 0.5;
  energy_transferred[1] -= ek2[sink] * 0.5;
}

void Ensemble_LAN::compute1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  if (type == 3) {
    integrate_nvt_lan_half(mass, velocity_per_atom);

    velocity_verlet(
      true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
  } else {
    integrate_heat_lan_half(group, mass, velocity_per_atom);

    velocity_verlet(
      true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
  }
}

void Ensemble_LAN::compute2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  if (type == 3) {
    velocity_verlet(
      false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    integrate_nvt_lan_half(mass, velocity_per_atom);

    find_thermo(
      true, box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom,
      thermo);
  } else {
    velocity_verlet(
      false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    integrate_heat_lan_half(group, mass, velocity_per_atom);
  }
}
