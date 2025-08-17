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
  void compute(
    Parameters& para);
  void report_error(
    Parameters& para,
    float time_used,
    const int epoch,
    const float loss_total,
    const float rmse_energy_train,
    const float rmse_force_train,
    const float rmse_virial_train,
    const float lr,
    float* step_parameters);
  void predict(Parameters& para, float* step_parameters);

protected:
  bool has_test_set = false;
  int N; // max atom number
  int num_batches = 0;
  int number_of_variables = 10; // number of variables
  int number_of_variables_ann = 0; // number of variables in ANN
  int number_of_variables_descriptor = 0; // number of variables in descriptor
  int maximum_epochs = 50; // maximum number of epochs
  int maximum_steps = 10000; // maximum number of steps
  float lr; // learning rate
  float start_lr;     // start learning rate
  float stop_lr; // stop learning rate 
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size
  Adam* optimizer;
  FILE* fid_loss_out = NULL;
  std::unique_ptr<Potential> potential;
  std::vector<std::vector<Dataset>> train_set;
  std::vector<Dataset> test_set;
  std::vector<int> batch_indices;
  std::vector<std::vector<int>> batch_type_sums;
  std::vector<float> batch_energies;
  void output(
    bool is_stress,
    int num_components,
    FILE* fid,
    float* prediction,
    float* reference,
    Dataset& dataset);
  void update_learning_rate_cos(float& lr, int step, int num_batches, Parameters& para); // Update learning rate with Cosine Annealing
  void update_learning_rate_cos_restart(float& lr, int step, int num_batches, Parameters& para); // Update learning rate with Cosine Annealing Warmup Restarts
  void update_energy_force_virial(
    FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset);
  void update_dipole(FILE* fid_dipole, Dataset& dataset);
  void update_polarizability(FILE* fid_polarizability, Dataset& dataset);
  void write_gnep_txt(FILE* fid_gnep, Parameters& para, float* step_parameters);
};
