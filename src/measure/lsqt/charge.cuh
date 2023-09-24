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

#pragma once
#include "common.cuh"
#include <random>

class Charge
{
public:
  void add_impurities(
    std::mt19937&,
    int,
    real*,
    int*,
    std::vector<real>&,
    std::vector<real>&,
    std::vector<real>&,
    real*);
  bool has = false;
  int Ni;  // number of impurities
  real W;  // impurity strength
  real xi; // impurity range
private:
  int Nx, Ny, Nz, Nxyz; // number of cells
  real rc;              // cutoff distance for impurity potential
  real rc2;             // cutoff square
  std::vector<int> cell_count;
  std::vector<int> cell_count_sum;
  std::vector<int> cell_contents;
  std::vector<int> impurity_indices;
  std::vector<real> impurity_strength;
  void find_impurity_indices(std::mt19937&, int);
  void find_impurity_strength(std::mt19937&);
  void find_potentials(
    int, real*, int*, std::vector<real>&, std::vector<real>&, std::vector<real>&, real*);
  int find_cell_id(real, real, real, real);
  void find_cell_id(real, real, real, real, int&, int&, int&, int&);
  void find_cell_numbers(int*, real*);
  void
  find_cell_contents(int, int*, real*, std::vector<real>&, std::vector<real>&, std::vector<real>&);
  int find_neighbor_cell(int, int, int, int, int, int, int);
};
