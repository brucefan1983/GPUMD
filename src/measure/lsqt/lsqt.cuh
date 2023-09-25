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
#include "hamiltonian.cuh"
#include "vector.cuh"
#include <random>
#include <string>
#include <vector>

class Vector;

class LSQT
{
public:
  void postprocess();

private:
  void initialize();
  void initialize_state(Vector& random_state);
  void find_moments_chebyshev(Vector& state_left, Vector& state_right, Vector& output);
  void apply_damping(real* inner_product_real, real* inner_product_imag);
  void perform_chebyshev_summation(
    real* inner_product_real, real* inner_product_imag, real* correlation_function);
  void evolve(int direction, real time_step_scaled, Vector& state_in);
  void evolvex(int direction, real time_step_scaled, Vector& state_in);
  void find_dos(Vector& random_state);
  void find_vac0(Vector& random_state);
  void find_vac(Vector& random_state);
  void find_msd(Vector& random_state);

  Hamiltonian H;
  std::mt19937 generator;
  int number_of_random_vectors = 1;
  int number_of_atoms = 0;
  int max_neighbor = 0;
  int number_of_pairs = 0;
  int number_of_energy_points = 201;
  int number_of_moments = 1000;
  int number_of_steps_correlation = 10;
  real energy_max = 10.1;
  std::vector<real> energy;
  std::vector<real> time_step;
  real volume;
};
