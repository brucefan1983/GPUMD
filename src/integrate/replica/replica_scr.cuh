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

/*----------------------------------------------------------------------------80
Isotropic stochastic cell rescaling (SCR) barostat for one replica of NPT
replica-exchange molecular dynamics.

Reference: M. Bernetti and G. Bussi, J. Chem. Phys. 153, 114107 (2020).
------------------------------------------------------------------------------*/

#pragma once
#include "gpu_stream.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include <random>
#include <string>

class Atom;
class Box;
class Ensemble_BDP_Replica;
template <typename T>
class GPU_Vector;

namespace replica
{
// epsilon = ln V follows
//   d epsilon = -(beta_T / tau_p) (P0 - P) dt + sqrt(2 kT beta_T / (V tau_p)) dW,
// integrated every `interval` MD steps. The box lengths and positions are scaled by
// exp(d epsilon / 3) and the velocities by its inverse. The Ito correction cancels
// the kT/V drift of the equation for V (Eq. 5 of the reference).
//
// The pressure is measured at the end of every interval-th step and the box is
// rescaled before the next step, so that the potential energy, positions, and box
// returned at the end of a block always belong to the same state.
class SCR_Barostat
{
public:
  // bulk_modulus in GPa; tau_p and interval in MD steps.
  SCR_Barostat(
    int device_id,
    int replica_id,
    double bulk_modulus,
    double tau_p,
    int interval,
    unsigned int seed);

  void set_target_pressure(double pressure); // eV/A^3

  // Rescales the box, positions, and velocities if a measured pressure is pending.
  void apply_pending_update(double temperature, Box& box, Atom& atom, gpuStream_t stream);

  // Counts one MD step and measures the pressure at the end of every interval-th step.
  void end_step(
    const Box& box,
    Atom& atom,
    Ensemble_BDP_Replica& thermostat,
    GPU_Vector<double>& thermo,
    const GPU_Stream& stream);

  // Measures a pending pressure again, after the velocities have been rescaled.
  void refresh_pending_pressure(
    const Box& box,
    Atom& atom,
    Ensemble_BDP_Replica& thermostat,
    GPU_Vector<double>& thermo,
    const GPU_Stream& stream);

  long long step_counter() const { return step_counter_; }
  std::string export_state() const;
  bool import_state(const std::string& state);

private:
  void measure_pressure(
    const Box& box,
    Atom& atom,
    Ensemble_BDP_Replica& thermostat,
    GPU_Vector<double>& thermo,
    const GPU_Stream& stream);

  int device_id_;
  int replica_id_;
  double target_pressure_ = 0.0; // eV/A^3
  double compressibility_;       // (eV/A^3)^-1
  double tau_p_;                 // MD steps
  int interval_;                 // MD steps between updates
  long long step_counter_ = 0;   // MD steps integrated by this replica
  bool pending_ = false;
  std::mt19937 rng_;
  std::normal_distribution<double> normal_;
  GPU_Pinned_Vector<double> host_thermo_; // T, U, p_xx, p_yy, p_zz
  GPU_Event ready_;
};
} // namespace replica
