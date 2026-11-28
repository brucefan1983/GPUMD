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

#include <string>
#include <vector>

struct MolecularAngle {
  int first_, center_, last_, type_;
};

struct MolecularInput {
  double cutoff_;
  int num_types_;
  std::vector<int> types_;
  std::vector<float> c6_, c12_;
  std::vector<int> offsets_, partners_;
  std::vector<float> lj_weights_;
  std::vector<float> bond_k_, bond_r0_;
  std::vector<int> bond_offsets_, bond_partners_, bond_types_;
  std::vector<float> angle_k_, angle_theta0_, angle_pi_minus_theta0_;
  std::vector<MolecularAngle> angles_;
  std::vector<int> angle_offsets_, angle_indices_;
};

MolecularInput read_molecular_input(
  const char* forcefield,
  const char* topology,
  const int number_of_atoms,
  const std::vector<std::string>& atom_symbols);
