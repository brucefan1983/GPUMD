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
The FIRE2 (fast inertial relaxation engine) minimizer
Reference: Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#pragma once
#include "force/force.cuh"
#include "minimizer.cuh"
#include "utilities/common.cuh"
#include <cmath>
#include <vector>

class Minimizer_FIRE2 : public Minimizer
{
private:
  // FIRE2 algorithm parameters
  const double finc_ = 1.1;    // Time step increase factor
  const double fdec_ = 0.5;    // Time step decrease factor
  const double astart_ = 0.25; // Initial velocity mixing parameter
  const double fa_ = 0.99;     // Velocity mixing decay factor
  const int Nmin_ = 20;        // Minimum steps before increasing dt

  // Time step parameters (in GPUMD internal units).
  // dtmax = 0.1 corresponds to ~7-20 fs depending on element mass,
  // which is within the safe MD stability range for metals.
  // The original dtmax=1.0 was 5-14x too large, causing integrator
  // instability once forces become small near convergence.
  double dt_0_ = 0.01;   // Initial time step (conservative start)
  double dt_;            // Current time step
  double dtmax_ = 0.1;   // Maximum time step (was 1.0 - too large!)
  double dtmin_ = 2e-4;  // Minimum time step

  // Maximum atomic displacement per step (Angstrom).
  // Hard safety backstop independent of dt.
  const double maxstep_ = 0.2;

  // Maximum cell strain per step (dimensionless).
  // Scaled by cbrt(N) so absolute box displacement is size-independent:
  //   N=2:   ~0.016,  N=54:  ~0.005,  N=250: ~0.003
  // This is the GPUMD analogue of LAMMPS box/relax vmax.
  double max_strain_step_;

  // Hard cap on the Nesterov abc_multiplier to prevent runaway amplification
  // of atomic velocities when alpha has decayed and Nsteps is large.
  const double abc_mult_max_ = 1.5;

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
    : Minimizer(-1, number_of_atoms, number_of_steps, force_tolerance),
      dt_(dt_0_),
      optimize_cell_(optimize_cell),
      hydrostatic_strain_(hydrostatic_strain),
      cell_factor_(static_cast<double>(number_of_atoms)),
      max_strain_step_(0.02 / std::cbrt(static_cast<double>(number_of_atoms)))
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
