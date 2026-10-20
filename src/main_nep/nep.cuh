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
#include "potential.cuh"
#include "parameters.cuh"
#include "nep_training.cuh"
#include "nep_batch_data.cuh"
#include "energy_difference.cuh"
#include "nep_compile.cuh"
#include "nep_evaluate.cuh"
#include "nep_common.cuh"
#include <memory>
class Parameters;

struct NEP_Temperature_Data {
  GPU_Vector<float> temperature;
};

class NEP : public Potential
{
public:
  explicit NEP(std::unique_ptr<Parameters> model_parameters);
  int prepare_data() override;
  void prepare_training() override;
  void evaluate(int batch, const float* population) override;
  void accumulate_fitness(float* fitness_total) override;
  void print_training_header() override;
  void report_error(const TrainingReport& report) override;
  void predict(float* parameters) override;
  Parameters& get_parameters() override { return *model_parameters; }

  void initialize_parameters(
    std::mt19937& rng,
    std::vector<float>& mu, std::vector<float>& sigma) override;
  void parameter_groups(std::vector<int>& groups) override;
  void read_model(float* parameters) override;

  void write_restart(
    const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma) override;

private:
  void configure(Parameters& para);
  void initialize(Parameters& para, int N, int Nc, int deviceCount);
  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount);
  std::unique_ptr<Parameters> model_parameters;
  std::unique_ptr<NEP_Training> training;
  NEP_Energy_Loss energy_loss;
  NEP_Batch_Data<NEP_Temperature_Data> temperature_data;
  EnergyDifference energy_difference;
  NEP_Neighbor neighbor[16];
  NEP_Descriptor_Parameters paramb;
  NEP_ANN annmb[16];
  NEP_Data nep_data[16];
  NEP_ZBL zbl;
  std::unique_ptr<NEP_Compile> compiled_kernel_;
  void initialize_model(const Parameters& para);
  void initialize_device(const Parameters& para, int N, int device_id);
  void initialize_compiled_kernel(const Parameters& para);
};
