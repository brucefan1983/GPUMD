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
#include <random>
#include <vector>
class Fitness;
class Fitness;

class SNES
{
public:
  SNES(char*, Parameters&, Fitness*);

protected:
  std::mt19937 rng;
  int maximum_generation = 10000;
  int number_of_variables = 10;
  int population_size = 20;
  float eta_sigma = 0.1f;
  std::vector<int> index;
  std::vector<float> fitness;
  std::vector<float> fitness_copy;
  std::vector<float> population;
  std::vector<float> population_copy;
  std::vector<float> mu;
  std::vector<float> sigma;
  std::vector<float> utility;
  std::vector<float> s;
  std::vector<float> s_copy;
  void initialize_rng();
  void initialize_mu_and_sigma(char*);
  void calculate_utility();
  void compute(char*, Parameters&, Fitness*);
  void create_population();
  void regularize(Parameters&);
  void sort_population();
  void update_mu_and_sigma(char*);
};
