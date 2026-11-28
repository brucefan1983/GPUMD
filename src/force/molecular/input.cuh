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
  int first, center, last, type;
};

struct MolecularInput {
  double cutoff;
  int num_types;
  std::vector<int> types;
  std::vector<double> c6, c12;
  std::vector<int> offsets, partners;
  std::vector<double> lj_weights;
  std::vector<double> bond_k, bond_r0;
  std::vector<int> bond_offsets, bond_partners, bond_types;
  std::vector<double> angle_k, angle_theta0;
  std::vector<MolecularAngle> angles;
  std::vector<int> angle_offsets, angle_indices;
};

MolecularInput read_molecular_input(
  const char* forcefield,
  const char* topology,
  const int number_of_atoms,
  const std::vector<std::string>& atom_symbols);
