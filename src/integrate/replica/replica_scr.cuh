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


#pragma once
#include "model/box.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include "utilities/gpu_stream.cuh"
#include <random>
#include <string>
#include <vector>

namespace replica
{
// Stochastic cell rescaling barostat (Bernetti and Bussi, J. Chem. Phys. 153,
// 114107 (2020)) for one replica. epsilon = ln V follows
//   d epsilon = -(beta_T / tau_p) (P0 - P) dt + sqrt(2 kT beta_T / (V tau_p)) dW,
// integrated every `interval` MD steps. The coupled directions are scaled by
// exp(d epsilon / d), positions by that factor and velocities by its inverse.
// The Ito correction cancels the kT/V drift of the equation for V (Eq. 5).
struct SCR_Barostat {
  explicit SCR_Barostat(int device_id);

  double target_pressure = 0.0; // eV/A^3
  double compressibility = 0.0; // (eV/A^3)^-1
  double tau_p = 0.0;           // MD steps
  int interval = 1;             // MD steps between barostat updates
  int couple[3] = {1, 1, 1};    // directions scaled by the barostat
  long long step_counter = 0;   // MD steps integrated by this replica
  // The pressure is measured at the end of every interval-th step and the box is
  // rescaled before the next step, so that the potential energy, positions, and
  // box returned at the end of a block always belong to the same state.
  bool pending = false;
  std::mt19937 rng;
  std::normal_distribution<double> normal;
  GPU_Pinned_Vector<double> host_thermo; // T, U, p_xx, p_yy, p_zz
  GPU_Event ready;

  // Returns the length scale factor and updates the box.
  double update_box(Box& box, double target_temperature);
  std::string export_state() const;
  bool import_state(const std::string& state);
};


struct Replica_Slot;
void refresh_barostat_pressure(Replica_Slot& slot);
void apply_barostat(Replica_Slot& slot);

// Canonical velocities with zero total linear momentum, retaining kinetic fluctuations.
void generate_canonical_velocities(
  double temperature, const std::vector<double>& mass,
  std::vector<double>& velocity, unsigned int seed);
} // namespace replica
