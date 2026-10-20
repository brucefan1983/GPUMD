/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "dataset.cuh"
#include "energy_difference.cuh"
#include "potential.cuh"
#include <cstdio>
#include <utility>
#include <vector>

class Parameters;

struct NEP_Energy_Error
{
  float energy = 0.0f;
  float force = 0.0f;
  float virial = 0.0f;
  float ediff = 0.0f;
  float energy_shift = 0.0f;
};

// Common data preparation used by the NEP models. The owning model performs
// evaluation and chooses its losses and prediction outputs.
class NEP_Training
{
public:
  explicit NEP_Training(Parameters& para);
  ~NEP_Training();
  NEP_Training(const NEP_Training&) = delete;
  NEP_Training& operator=(const NEP_Training&) = delete;

  int N;
  int Nc;
  int device_count;
  int num_batches = 0;
  bool has_test_set = false;
  int max_NN_radial;
  int max_NN_angular;
  FILE* fid_loss_out = NULL;
  EnergyDifference energy_difference;
  std::vector<std::vector<Dataset>> train_set;
  std::vector<Dataset> test_set;

  void open_loss_file(const Parameters& para, const char* columns, bool include_ediff);
  void write_model(Parameters& para, const TrainingReport& report);
  void get_prediction_order(std::vector<std::pair<Dataset*, int>>& structures);

private:
  void prepare_datasets(Parameters& para, int deviceCount);
  void find_dataset_sizes(int& N, int& Nc);
};

// Energy, force, virial and energy-difference losses shared by energy models.
// Charge models keep their additional losses in their own classes.
class NEP_Energy_Loss
{
public:
  void initialize(const Parameters& para);
  void evaluate(Parameters& para, Dataset& data, EnergyDifference& energy_difference,
    int batch, int candidate, int device_id);
  void accumulate_fitness(const Parameters& para, float* fitness_total);
  void accumulate_energy_difference(const Parameters& para, float* fitness_total);
  NEP_Energy_Error get_error(Parameters& para, Dataset& data,
    EnergyDifference& energy_difference, bool is_train, int batch);

private:
  std::vector<float> fitness_energy;
  std::vector<float> fitness_force;
  std::vector<float> fitness_virial;
  std::vector<float> fitness_ediff;
};
