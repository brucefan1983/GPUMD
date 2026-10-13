/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "ensemble_fixed_lattice.cuh"

Ensemble_Fixed_Lattice::Ensemble_Fixed_Lattice(EnsembleType type) { this->type = type; }

void Ensemble_Fixed_Lattice::compute1(
  const double,
  const int,
  const int,
  const std::vector<Group>&,
  Box&,
  Atom&,
  GPU_Vector<double>&)
{
}

void Ensemble_Fixed_Lattice::compute2(
  const double,
  const int,
  const int,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo, Force& force)
{
  find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
}
