/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "replicate.cuh"

void Replicate(const char** param, int num_param, Box& box, Atom& atoms, std::vector<Group>& groups)
{
  int r[3]; // the number of replicates
  if (num_param != 4) {
    PRINT_INPUT_ERROR(
      "Replicate should have 3 parameters: number of replications in a, b and c directions.");
  }
  for (int i = 0; i < 3; i++) {
    if (!is_valid_int(param[i + 1], r + i))
      PRINT_INPUT_ERROR("Number of replications should be an integer.");
  }
  // repeat atom and group
  Atom new_atoms;
  int n = atoms.number_of_atoms;
  int N = n * r[0] * r[1] * r[2];

  std::vector<Group> new_groups;
  new_groups.resize(groups.size());
  for (int m = 0; m < groups.size(); m++) {
    new_groups[m].number = groups[m].number;
    new_groups[m].cpu_label.resize(N);
  }

  new_atoms.number_of_atoms = N;
  new_atoms.cpu_type.resize(N);
  new_atoms.cpu_mass.resize(N);
  new_atoms.cpu_atom_symbol.resize(N);
  new_atoms.cpu_position_per_atom.resize(N * 3);
  new_atoms.cpu_velocity_per_atom.resize(N * 3);
  int cur = 0;
  for (int i = 0; i < r[0]; i++) {
    for (int j = 0; j < r[1]; j++) {
      for (int k = 0; k < r[2]; k++) {
        int ijk[3] = {i, j, k};
        for (int nn = 0; nn < atoms.number_of_atoms; nn++) {
          new_atoms.cpu_type[cur] = atoms.cpu_type[nn];
          new_atoms.cpu_mass[cur] = atoms.cpu_mass[nn];
          new_atoms.cpu_atom_symbol[cur] = atoms.cpu_atom_symbol[nn];
          for (int m = 0; m < groups.size(); m++)
            new_groups[m].cpu_label[cur] = groups[m].cpu_label[nn];
          for (int d = 0; d < 3; d++) {
            new_atoms.cpu_position_per_atom[cur + d * N] = atoms.cpu_position_per_atom[nn + d * n];
            if (!box.triclinic)
              new_atoms.cpu_position_per_atom[cur + d * N] += ijk[d] * box.cpu_h[d];
            else
              new_atoms.cpu_position_per_atom[cur + d * N] +=
                i * box.cpu_h[d * 3] + j * box.cpu_h[d * 3 + 1] + k * box.cpu_h[d * 3 + 2];
            new_atoms.cpu_velocity_per_atom[cur + d * N] = atoms.cpu_velocity_per_atom[nn + d * n];
          }
          cur++;
        }
      }
    }
  }

  // repeat box
  if (box.triclinic) {
    for (int i = 0; i < 9; i++) {
      int direction = i % 3;
      box.cpu_h[i] *= r[direction];
    }
    box.get_inverse();
  } else {
    for (int i = 0; i < 3; i++) {
      box.cpu_h[i] *= r[i];
      box.cpu_h[i + 3] *= r[i];
    }
  }
  // copy to old
  for (int m = 0; m < groups.size(); m++) {
    groups[m].number = new_groups[m].number;
    groups[m].cpu_label.assign(new_groups[m].cpu_label.begin(), new_groups[m].cpu_label.end());
    groups[m].find_size(N, m);
    groups[m].find_contents(N);
  }
  atoms.number_of_atoms = N;
  atoms.cpu_type.assign(new_atoms.cpu_type.begin(), new_atoms.cpu_type.end());
  atoms.cpu_mass.assign(new_atoms.cpu_mass.begin(), new_atoms.cpu_mass.end());
  atoms.cpu_atom_symbol.assign(new_atoms.cpu_atom_symbol.begin(), new_atoms.cpu_atom_symbol.end());
  atoms.cpu_position_per_atom.assign(
    new_atoms.cpu_position_per_atom.begin(), new_atoms.cpu_position_per_atom.end());
  atoms.cpu_velocity_per_atom.assign(
    new_atoms.cpu_velocity_per_atom.begin(), new_atoms.cpu_velocity_per_atom.end());
  atoms.cpu_type_size.assign(atoms.cpu_type_size.begin(), atoms.cpu_type_size.end());
  for (int& i : atoms.cpu_type_size)
    i = i * r[0] * r[1] * r[2];

  print_line_1();
  printf("Replicate cell by %d * %d * %d.\n", r[0], r[1], r[2]);
  printf("Number of atoms is %d.\n", atoms.number_of_atoms);
  int number_of_types = atoms.cpu_type_size.size();
  if (number_of_types == 1) {
    printf("There is only one atom type.\n");
  } else {
    printf("There are %d atom types.\n", number_of_types);
  }
  for (int m = 0; m < number_of_types; m++) {
    printf("    %d atoms of type %d.\n", atoms.cpu_type_size[m], m);
  }
  print_line_2();
}