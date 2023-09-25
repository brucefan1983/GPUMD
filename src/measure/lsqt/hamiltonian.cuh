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
#include "utilities/gpu_vector.cuh"
class Vector;

class Hamiltonian
{
public:
  void initialize_gpu(int number_of_atoms, int mn, int number_of_pairs, real emax);
  void apply(Vector&, Vector&);
  void apply_commutator(Vector&, Vector&);
  void apply_current(Vector&, Vector&);
  void kernel_polynomial(Vector&, Vector&, Vector&);
  void chebyshev_01(Vector&, Vector&, Vector&, real, real, int);
  void chebyshev_2(Vector&, Vector&, Vector&, Vector&, real, int);
  void chebyshev_1x(Vector&, Vector&, real);
  void chebyshev_2x(Vector&, Vector&, Vector&, Vector&, Vector&, Vector&, Vector&, real, int);

private:
  GPU_Vector<int> neighbor_number;
  GPU_Vector<int> neighbor_list;
  GPU_Vector<real> potential;
  GPU_Vector<real> hopping_real;
  GPU_Vector<real> hopping_imag;
  GPU_Vector<real> xx;
  int grid_size;
  int n;
  int max_neighbor;
  real energy_max;
};
