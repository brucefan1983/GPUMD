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
#include "force/force.cuh"
#include "minimizer.cuh"
#include "utilities/common.cuh"

class Minimizer_FIRE2 : public Minimizer
{
private:
  // FIRE2 algorithm parameters
  const double finc_ = 1.1;    // Time step increase factor
  const double fdec_ = 0.5;    // Time step decrease factor
  const double astart_ = 0.25; // Initial velocity mixing parameter
  const double fa_ = 0.99;     // Velocity mixing decay factor
  const int Nmin_ = 20;        // Minimum steps before increasing dt

  // Time step parameters (converted to internal units)
  const double dt_0_ = 0.1;   // Initial time step
  double dt_;                                        // Current time step
  const double dtmax_ = 1.0;  // Maximum time step
  const double dtmin_ = 2e-3; // Minimum time step

  // Displacement limit
  const double maxstep_ = 0.2; // Maximum displacement per step (Angstrom)

  // Cell optimization parameters
  bool optimize_cell_;      // Whether to optimize cell
  bool hydrostatic_strain_; // Whether to apply hydrostatic strain only
  double cell_factor_;      // Scale factor for cell degrees of freedom (= number of atoms)

public:
  // Constructor with cell optimization options
  Minimizer_FIRE2(
    const int number_of_atoms,
    const int number_of_steps,
    const double force_tolerance,
    const bool optimize_cell = false,
    const bool hydrostatic_strain = false)
    : Minimizer(number_of_atoms, number_of_steps, force_tolerance),
      dt_(dt_0_),
      optimize_cell_(optimize_cell),
      hydrostatic_strain_(hydrostatic_strain),
      cell_factor_(static_cast<double>(number_of_atoms))
  {
  }

  void compute(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
};