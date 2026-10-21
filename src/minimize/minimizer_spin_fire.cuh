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
  bool fixed_magnitude_, relax_atoms_, box_change_, hydrostatic_strain_;
};
