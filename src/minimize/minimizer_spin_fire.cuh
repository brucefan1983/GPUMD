/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/
#pragma once
#include "minimizer.cuh"

// Spherical or Cartesian spins, optionally coupled to atoms and cell strain.
class Minimizer_Spin_FIRE : public Minimizer
{
public:
  Minimizer_Spin_FIRE(int atoms, int steps, double tolerance, bool fixed_magnitude,
    bool relax_atoms=false, bool box_change=false, bool hydrostatic_strain=false)
    : Minimizer(-1, 0, atoms, steps, tolerance), fixed_magnitude_(fixed_magnitude),
      relax_atoms_(relax_atoms), box_change_(box_change), hydrostatic_strain_(hydrostatic_strain) {}
  void compute(Force&, Box&, Atom&, GPU_Vector<double>&, std::vector<Group>&) override;

private:
  // FIRE controls in fictitious optimization time, not physical MD time.
  static constexpr double f_inc = 1.1;
  static constexpr double f_dec = 0.5;
  static constexpr double alpha_start = 0.1;
  static constexpr double f_alpha = 0.99;
  static constexpr double dt_0 = 0.1;
  static constexpr double dt_max_spin = 1.0;
  static constexpr double dt_max_joint = 0.3;
  static constexpr double dt_min = 1.0e-12;
  static constexpr int N_min = 5; // Grow dt after more than N_min positive-power steps.

  // Per-step limits and the zero-pressure convergence target.
  static constexpr double max_spin_angle = 0.1; // rad, fixed magnitude
  static constexpr double max_spin_displacement = 0.1; // mu_B, variable magnitude
  static constexpr double max_atom_displacement = 0.1; // angstrom, nonaffine
  static constexpr double max_cell_strain = 0.01; // Frobenius norm
  static constexpr double stress_tolerance_gpa = 1.0e-4;

  // Stagnation detection; these do not relax the requested force tolerance.
  static constexpr double tiny_update_epsilon_factor = 8.0;
  static constexpr int max_tiny_updates = 20;
  static constexpr double residual_progress_ratio = 0.99;
  static constexpr int max_steps_without_progress = 1000;

  bool fixed_magnitude_, relax_atoms_, box_change_, hydrostatic_strain_;
};
