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

#pragma once
#include "minimizer.cuh"
#include <string>

// Spherical or Cartesian spins, optionally coupled to atoms and cell strain.
class Minimizer_Spin_FIRE : public Minimizer
{
public:
  Minimizer_Spin_FIRE(const std::vector<std::string>&, int fixed_group, const Atom&, const Box&);
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
  static constexpr double max_spin_angle = 0.1;        // rad, fixed magnitude
  static constexpr double max_spin_displacement = 0.1; // mu_B, variable magnitude
  static constexpr double max_atom_displacement = 0.1; // angstrom, nonaffine
  static constexpr double max_cell_strain = 0.01;      // Frobenius norm
  static constexpr double stress_tolerance_gpa = 1.0e-4;

  // Stagnation detection; these do not relax the requested force tolerance.
  static constexpr double tiny_update_epsilon_factor = 8.0;
  static constexpr int max_tiny_updates = 20;
  static constexpr double residual_progress_ratio = 0.99;
  static constexpr int max_steps_without_progress = 1000;

  bool fixed_magnitude_, relax_atoms_, box_change_, hydrostatic_strain_;
};
