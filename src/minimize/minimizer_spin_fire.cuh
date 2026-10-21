/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/
#pragma once
#include "minimizer.cuh"

// Fixed lattice, with either spherical or Cartesian spin coordinates.
class Minimizer_Spin_FIRE : public Minimizer
{
public:
  Minimizer_Spin_FIRE(int atoms, int steps, double tolerance, bool fixed_magnitude)
    : Minimizer(-1, 0, atoms, steps, tolerance), fixed_magnitude_(fixed_magnitude) {}
  void compute(Force&, Box&, Atom&, GPU_Vector<double>&, std::vector<Group>&) override;

private:
  bool fixed_magnitude_;
};
