/*
    Copyright 2019 Zheyong Fan
    This file is part of GPUGA.
    GPUGA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUGA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUGA.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <random>
#include <vector>
class Fitness;

class GA
{
public:
  GA(char*, Fitness*);

protected:
  std::mt19937 rng;
  int maximum_generation = 2000;
  int number_of_variables = 10;
  int population_size = 20;
  float eta_sigma = 0.1f;
  std::vector<int> index;
  std::vector<float> fitness;
  std::vector<float> population;
  std::vector<float> population_copy;
  std::vector<float> mu;
  std::vector<float> sigma;
  std::vector<float> utility;
  std::vector<float> s;
  std::vector<float> s_copy;
  void initialize_rng();
  void initialize_mu_and_sigma();
  void calculate_utility();
  void compute(char*, Fitness*);
  void create_population();
  void regularize();
  void sort();
  void output(int, FILE*);
  void update_mu_and_sigma();
};
