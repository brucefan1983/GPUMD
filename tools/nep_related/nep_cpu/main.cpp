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

/*----------------------------------------------------------------------------80
Usage:
    Compile:
        g++ -O3 main.cpp nep.cpp
    run:
        ./a.out
------------------------------------------------------------------------------*/

#include "nep.h"
#include "utility.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>

const int MN = 1000;
const int num_repeats = 1000;

struct Atom {
  int N;
  std::vector<int> num_cells, type, NN_radial, NL_radial, NN_angular, NL_angular;
  std::vector<double> box, ebox, position, r12, potential, force, virial;
};
void readXYZ(Atom& atom);
void find_speed();
void compare_analytical_and_finite_difference();

int main(int argc, char* argv[])
{

  find_speed();
  compare_analytical_and_finite_difference();

  return 0;
}

void readXYZ(Atom& atom)
{
  std::cout << "Reading xyz.in.\n";

  std::ifstream input_file("xyz.in");

  if (!input_file) {
    std::cout << "Cannot open xyz.in\n";
    exit(1);
  }

  input_file >> atom.N;
  std::cout << "    Number of atoms is " << atom.N << ".\n";

  atom.num_cells.resize(3);
  atom.box.resize(18);
  atom.ebox.resize(18);
  input_file >> atom.box[0];
  input_file >> atom.box[3];
  input_file >> atom.box[6];
  input_file >> atom.box[1];
  input_file >> atom.box[4];
  input_file >> atom.box[7];
  input_file >> atom.box[2];
  input_file >> atom.box[5];
  input_file >> atom.box[8];
  get_inverse(atom.box.data());

  std::cout << "    Box matrix h = [a, b, c] is\n";
  for (int d1 = 0; d1 < 3; ++d1) {
    for (int d2 = 0; d2 < 3; ++d2) {
      std::cout << "\t" << atom.box[d1 * 3 + d2];
    }
    std::cout << "\n";
  }

  std::cout << "    Inverse box matrix g = inv(h) is\n";
  for (int d1 = 0; d1 < 3; ++d1) {
    for (int d2 = 0; d2 < 3; ++d2) {
      std::cout << "\t" << atom.box[9 + d1 * 3 + d2];
    }
    std::cout << "\n";
  }

  std::vector<std::string> atom_symbols = get_atom_symbols();

  atom.type.resize(atom.N);
  atom.NN_radial.resize(atom.N);
  atom.NL_radial.resize(atom.N * MN);
  atom.NN_angular.resize(atom.N);
  atom.NL_angular.resize(atom.N * MN);
  atom.r12.resize(atom.N * MN * 6);
  atom.position.resize(atom.N * 3);
  atom.potential.resize(atom.N);
  atom.force.resize(atom.N * 3);
  atom.virial.resize(atom.N * 9);

  for (int n = 0; n < atom.N; n++) {
    std::string atom_symbol_tmp;
    input_file >> atom_symbol_tmp >> atom.position[n] >> atom.position[n + atom.N] >>
      atom.position[n + atom.N * 2];
    bool is_allowed_element = false;
    for (int t = 0; t < atom_symbols.size(); ++t) {
      if (atom_symbol_tmp == atom_symbols[t]) {
        atom.type[n] = t;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      std::cout << "There is atom in xyz.in that is not allowed in the used NEP potential.\n";
      exit(1);
    }
  }
}

void find_speed()
{
  Atom atom;
  readXYZ(atom);
  NEP3 nep3(atom.N);

  const int size_x12 = atom.NL_radial.size();

  find_neighbor_list_small_box(
    nep3.paramb.rc_radial, nep3.paramb.rc_angular, atom.N, atom.box.data(), atom.position.data(),
    atom.position.data() + atom.N, atom.position.data() + atom.N * 2, atom.num_cells.data(),
    atom.ebox.data(), atom.NN_radial.data(), atom.NL_radial.data(), atom.NN_angular.data(),
    atom.NL_angular.data(), atom.r12.data(), atom.r12.data() + size_x12,
    atom.r12.data() + size_x12 * 2, atom.r12.data() + size_x12 * 3, atom.r12.data() + size_x12 * 4,
    atom.r12.data() + size_x12 * 5);

  clock_t time_begin = clock();

  for (int n = 0; n < num_repeats; ++n) {
    nep3.compute(
      atom.NN_radial, atom.NL_radial, atom.NN_angular, atom.NL_angular, atom.type, atom.r12,
      atom.potential, atom.force, atom.virial);
  }

  clock_t time_finish = clock();
  double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
  std::cout << "Time used for NEP calculations = " << time_used << " s.\n";

  double speed = atom.N * num_repeats / time_used;
  double cost = 1000 / speed;
  std::cout << "Computational speed = " << speed << " atom-step/second.\n";
  std::cout << "Computational cost = " << cost << " mini-second/atom-step.\n";
}

void compare_analytical_and_finite_difference()
{
  Atom atom;
  readXYZ(atom);

  std::vector<double> force_finite_difference(atom.force.size());
  std::vector<double> position_copy(atom.position.size());
  for (int n = 0; n < atom.position.size(); ++n) {
    position_copy[n] = atom.position[n];
  }

  NEP3 nep3(atom.N);

  const double delta = 1.0e-7;

  for (int n = 0; n < atom.N; ++n) {
    for (int d = 0; d < 3; ++d) {
      atom.position[n + d * atom.N] = position_copy[n + d * atom.N] - delta; // negative shift
      const int size_x12 = atom.NL_radial.size();

      find_neighbor_list_small_box(
        nep3.paramb.rc_radial, nep3.paramb.rc_angular, atom.N, atom.box.data(),
        atom.position.data(), atom.position.data() + atom.N, atom.position.data() + atom.N * 2,
        atom.num_cells.data(), atom.ebox.data(), atom.NN_radial.data(), atom.NL_radial.data(),
        atom.NN_angular.data(), atom.NL_angular.data(), atom.r12.data(), atom.r12.data() + size_x12,
        atom.r12.data() + size_x12 * 2, atom.r12.data() + size_x12 * 3,
        atom.r12.data() + size_x12 * 4, atom.r12.data() + size_x12 * 5);

      nep3.compute(
        atom.NN_radial, atom.NL_radial, atom.NN_angular, atom.NL_angular, atom.type, atom.r12,
        atom.potential, atom.force, atom.virial);

      double energy_negative_shift = 0.0;
      for (int n = 0; n < atom.N; ++n) {
        energy_negative_shift += atom.potential[n];
      }

      atom.position[n + d * atom.N] = position_copy[n + d * atom.N] + delta; // positive shift

      find_neighbor_list_small_box(
        nep3.paramb.rc_radial, nep3.paramb.rc_angular, atom.N, atom.box.data(),
        atom.position.data(), atom.position.data() + atom.N, atom.position.data() + atom.N * 2,
        atom.num_cells.data(), atom.ebox.data(), atom.NN_radial.data(), atom.NL_radial.data(),
        atom.NN_angular.data(), atom.NL_angular.data(), atom.r12.data(), atom.r12.data() + size_x12,
        atom.r12.data() + size_x12 * 2, atom.r12.data() + size_x12 * 3,
        atom.r12.data() + size_x12 * 4, atom.r12.data() + size_x12 * 5);

      nep3.compute(
        atom.NN_radial, atom.NL_radial, atom.NN_angular, atom.NL_angular, atom.type, atom.r12,
        atom.potential, atom.force, atom.virial);

      double energy_positive_shift = 0.0;
      for (int n = 0; n < atom.N; ++n) {
        energy_positive_shift += atom.potential[n];
      }

      force_finite_difference[n + d * atom.N] =
        (energy_negative_shift - energy_positive_shift) / (2.0 * delta);
    }
  }

  const int size_x12 = atom.NL_radial.size();

  find_neighbor_list_small_box(
    nep3.paramb.rc_radial, nep3.paramb.rc_angular, atom.N, atom.box.data(), position_copy.data(),
    position_copy.data() + atom.N, position_copy.data() + atom.N * 2, atom.num_cells.data(),
    atom.ebox.data(), atom.NN_radial.data(), atom.NL_radial.data(), atom.NN_angular.data(),
    atom.NL_angular.data(), atom.r12.data(), atom.r12.data() + size_x12,
    atom.r12.data() + size_x12 * 2, atom.r12.data() + size_x12 * 3, atom.r12.data() + size_x12 * 4,
    atom.r12.data() + size_x12 * 5);

  nep3.compute(
    atom.NN_radial, atom.NL_radial, atom.NN_angular, atom.NL_angular, atom.type, atom.r12,
    atom.potential, atom.force, atom.virial);

  std::ofstream output_file("force_analytical.out");

  if (!output_file.is_open()) {
    std::cout << "Cannot open force_analytical.out\n";
    exit(1);
  }
  for (int n = 0; n < atom.N; ++n) {
    output_file << std::setprecision(15) << atom.force[n] << " " << atom.force[n + atom.N] << " "
                << atom.force[n + atom.N * 2] << "\n";
  }
  output_file.close();

  std::ofstream output_finite_difference("force_finite_difference.out");

  if (!output_finite_difference.is_open()) {
    std::cout << "Cannot open force_finite_difference.out\n";
    exit(1);
  }
  for (int n = 0; n < atom.N; ++n) {
    output_finite_difference << std::setprecision(15) << force_finite_difference[n] << " "
                             << force_finite_difference[n + atom.N] << " "
                             << force_finite_difference[n + atom.N * 2] << "\n";
  }
  output_finite_difference.close();
}
