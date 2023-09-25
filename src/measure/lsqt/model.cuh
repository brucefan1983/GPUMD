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
#include "anderson.cuh"
#include "charge.cuh"
#include "common.cuh"
#include <random>
class Vector;

class Model
{
public:
  Model(std::string input_dir);
  ~Model();
  void initialize_state(Vector& random_state, int orbital);

  bool calculate_vac0 = false;
  bool calculate_vac = false;
  bool calculate_msd = false;
  bool calculate_spin = false;
  bool calculate_ldos = false;

  int number_of_random_vectors = 1;
  int number_of_atoms = 0;
  int max_neighbor = 0;
  int number_of_pairs = 0;
  int number_of_energy_points = 0;
  int number_of_moments = 1000;
  int number_of_steps_correlation = 0;
  int number_of_local_orbitals = 0;
  std::string input_dir;
  real energy_max = 10;

  real* energy;
  real* time_step;
  std::vector<int> local_orbitals;

  int* neighbor_number;
  int* neighbor_list;
  real* xx;
  real* potential;
  real* hopping_real;
  real* hopping_imag;

  real volume;

private:
  void print_started_reading(std::string filename);
  void print_finished_reading(std::string filename);

  // for both lattice and general models
  void initialize_parameters();
  void verify_parameters();
  void initialize_energy();
  void initialize_time();
  void initialize_local_orbitals();

  // only for general model
  void initialize_neighbor();
  void initialize_positions();
  void initialize_potential();
  void initialize_hopping();
  void initialize_model_general();

  bool requires_time = false;

  // disorder
  Anderson anderson;
  Charge charge;

  bool has_vacancy_disorder = false;
  int number_of_vacancies;

  int pbc[3];
  real box_length[3];
  std::vector<real> x, y, z;

  std::mt19937 generator;
};
