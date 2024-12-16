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
#include "dataset.cuh"
#include "adam.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <stdio.h>
#include <vector>

class Parameters;

class Fitness
{
public:
  Fitness(Parameters& para, Adam* adam);
  ~Fitness();
  void update_learning_rate(float& lr, const int step); // Update learning rate
  void compute(
    Parameters& para);
  void report_error(
    Parameters& para,
    const int generation,
    const float loss_total,
    float* step_parameters);
  void predict(Parameters& para, float* step_parameters);

protected:
  bool has_test_set = false;
  int N; // max atom number
  int num_batches = 0;
  int number_of_variables = 10; // number of variables
  int number_of_variables_ann = 0; // number of variables in ANN
  int number_of_variables_descriptor = 0; // number of variables in descriptor
  int maximum_generation = 10000; // maximum number of iterations
  float lr = 1e-3f; // learning rate
  float start_lr = 1e-3f;     // start learning rate
  float stop_lr = 3.51e-08f; // stop learning rate
  int stop_step = 1000000; // stop step
  int decay_step = 5000; // decay 
  float decay_rate; // decay rate
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size
  Adam* optimizer;
  GPU_Vector<float> gpu_gradients; // Gradients of parameters g
  std::vector<float> fitness_loss;
  FILE* fid_loss_out = NULL;
  std::unique_ptr<Potential> potential;
  std::vector<std::vector<Dataset>> train_set;
  std::vector<Dataset> test_set;
  void output(
    bool is_stress,
    int num_components,
    FILE* fid,
    float* prediction,
    float* reference,
    Dataset& dataset);
  void update_energy_force_virial(
    FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset);
  void update_dipole(FILE* fid_dipole, Dataset& dataset);
  void update_polarizability(FILE* fid_polarizability, Dataset& dataset);
  void write_nep_txt(FILE* fid_nep, Parameters& para, float* step_parameters);
};
