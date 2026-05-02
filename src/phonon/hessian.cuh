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
#include "utilities/gpu_vector.cuh"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

class Box;
class Group;
class Force;
class Atom;

class Hessian
{
private:
  int cxyz[3] = {1, 1, 1};

public:
  double displacement = 0.015;
  double cutoff = 4.0;
  double phonon_cutoff = 8.0;

  void compute(Force& force, Box& box, Atom& atom, std::vector<Group>& group);

  void parse(const char**, int);
  void get_cutoff_from_potential(Force& force);

protected:
  size_t num_basis;
  size_t num_kpoints;

  std::vector<size_t> basis;
  std::vector<size_t> label;
  std::vector<double> mass;
  std::vector<double> kpoints;
  std::vector<double> kpath;
  std::vector<double> kpath_sym;
  std::vector<double> H;
  std::vector<double> DR;
  std::vector<double> DI;
  std::vector<std::string> hsp_names;

  void create_basis(const std::vector<double>& cpu_mass, int N);
  void create_kpoints(const Box& box);
  void initialize(const std::vector<double>& cpu_mass, Box& box, Force& force, int N);
  void finalize(void);

  void find_H(Force& force, Box& box, Atom& atom, std::vector<Group>& group);

  bool is_too_far(
    const Box& box,
    const std::vector<double>& cpu_position_per_atom,
    const size_t n1,
    const size_t n2);

  void find_dispersion(const Box& box, Atom& atom);

  void find_D(const Box& box, Atom& atom);

  void find_eigenvectors();
  void output_D();
  void find_omega(FILE*, size_t, size_t);
  void find_omega_batch(FILE*);
};
