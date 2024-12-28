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
  void update_learning_rate(double& lr, int step, int Nc); // Update learning rate
  void compute(
    Parameters& para);
  void report_error(
    Parameters& para,
    const int generation,
    const double loss_total,
    const double rmse_energy_train,
    const double rmse_force_train,
    const double rmse_virial_train,
    const double lr,
    double* step_parameters);
  void predict(Parameters& para, double* step_parameters);

protected:
  bool has_test_set = false;
  int N; // max atom number
  int num_batches = 0;
  int number_of_variables = 10; // number of variables
  int number_of_variables_ann = 0; // number of variables in ANN
  int number_of_variables_descriptor = 0; // number of variables in descriptor
  int maximum_generation = 10000; // maximum number of iterations
  double lr = 1e-3; // learning rate
  double start_lr = 1e-3;     // start learning rate
  double stop_lr = 3.51e-08; // stop learning rate
  int decay_step = 5000; // decay 
  double decay_rate; // decay rate
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size
  Adam* optimizer;
  GPU_Vector<double> gpu_gradients; // Gradients of parameters g
  FILE* fid_loss_out = NULL;
  std::unique_ptr<Potential> potential;
  std::vector<std::vector<Dataset>> train_set;
  std::vector<Dataset> test_set;
  void output(
    bool is_stress,
    int num_components,
    FILE* fid,
    double* prediction,
    double* reference,
    Dataset& dataset);
  void update_energy_force_virial(
    FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset);
  void update_dipole(FILE* fid_dipole, Dataset& dataset);
  void update_polarizability(FILE* fid_polarizability, Dataset& dataset);
  void write_nep_txt(FILE* fid_nep, Parameters& para, double* step_parameters);
};
