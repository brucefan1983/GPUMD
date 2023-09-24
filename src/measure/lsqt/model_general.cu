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

#include "model.cuh"
#include <fstream>
#include <iostream>

void Model::initialize_model_general()
{
  initialize_neighbor();
  initialize_positions();
  initialize_potential();
  initialize_hopping();
}

void Model::initialize_neighbor()
{
  std::string filename = input_dir + "/neighbor.in";
  std::ifstream input(filename);

  if (!input.is_open()) {
    std::cout << "Error: cannot open " + filename << std::endl;
    exit(1);
  }
  print_started_reading(filename);

  input >> number_of_atoms >> max_neighbor;
  number_of_pairs = number_of_atoms * max_neighbor;

  neighbor_number = new int[number_of_atoms];
  neighbor_list = new int[number_of_pairs];

  for (int n = 0; n < number_of_atoms; ++n) {
    input >> neighbor_number[n];
    for (int m = 0; m < neighbor_number[n]; ++m) {
      int index = n * max_neighbor + m;
      input >> neighbor_list[index];
    }
  }

  input.close();

  std::cout << "- Number of orbitals is " << number_of_atoms << std::endl;
  std::cout << "- Maximum neighbor number is " << max_neighbor << std::endl;
  print_finished_reading(filename);
}

real reduce_distance(real d, real box)
{
  if (d > box / 2.0)
    return d - box;
  if (d < -box / 2.0)
    return d + box;
  else
    return d;
}

void Model::initialize_positions()
{
  std::string filename = input_dir + "/position.in";
  std::ifstream input(filename);

  if (!input.is_open()) {
    std::cout << "Error: cannot open " + filename << std::endl;
    exit(1);
  }
  print_started_reading(filename);

  real box;
  input >> box >> volume;
  real* x = new real[number_of_atoms];

  for (int i = 0; i < number_of_atoms; ++i)
    input >> x[i];
  input.close();

  std::cout << "- Box length along the transport direction is " << box << std::endl;
  std::cout << "- System volume is " << volume << std::endl;

  xx = new real[number_of_pairs];
  for (int n = 0; n < number_of_atoms; ++n) {
    for (int m = 0; m < neighbor_number[n]; ++m) {
      int index = n * max_neighbor + m;
      xx[index] = reduce_distance(x[neighbor_list[index]] - x[n], box);
    }
  }

  delete[] x;
  print_finished_reading(filename);
}

void Model::initialize_potential()
{
  std::string filename = input_dir + "/potential.in";
  print_started_reading(filename);

  std::ifstream input(filename);
  bool nonzero_potential = true;
  if (!input.is_open()) {
    std::cout << "- Could not open " + filename << std::endl;
    std::cout << "- Assuming zero on-site potential" << std::endl;
    nonzero_potential = false;
  } else {
    std::cout << "- On-site potential will be read in" << std::endl;
  }

  potential = new real[number_of_atoms];

  for (int n = 0; n < number_of_atoms; ++n) {
    if (nonzero_potential)
      input >> potential[n];
    else
      potential[n] = 0.0;
  }

  input.close();

  print_finished_reading(filename);
}

void Model::initialize_hopping()
{
  std::string filename = input_dir + "/hopping.in";
  print_started_reading(filename);
  std::ifstream input(filename);

  // type == 1 : complex hoppings
  // type == 2 : real hoppings
  // type == 3 : uniform hoppings (hoppings.in is not read)
  int type = 0;

  if (!input.is_open()) {
    type = 3;
    std::cout << "- Could not open " + filename << std::endl;
    std::cout << "- Assuming uniform hoppings with strength -1" << std::endl;
  } else {
    std::string first_line;
    input >> first_line;
    if (first_line == "complex") {
      type = 1;
      std::cout << "- Hoppings have imaginary part" << std::endl;
    } else if (first_line == "real") {
      type = 2;
      std::cout << "- Hoppings are real" << std::endl;
    } else {
      std::cout << "- Hoppings can only be real or complex" << std::endl;
      exit(1);
    }
  }

  hopping_real = new real[number_of_pairs];
  hopping_imag = new real[number_of_pairs];
  for (int n = 0; n < number_of_atoms; ++n) {
    for (int m = 0; m < neighbor_number[n]; ++m) {
      int index = n * max_neighbor + m;
      if (type < 3)
        input >> hopping_real[index];
      else
        hopping_real[index] = -1.0;
      if (type == 1)
        input >> hopping_imag[index];
      else
        hopping_imag[index] = 0.0;
    }
  }
  input.close();

  print_finished_reading(filename);
}
