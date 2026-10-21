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
#include "main_nep/dataset.cuh"
#include "main_nep/potential.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "main_nep/common/nep_training.cuh"
#include "main_nep/common/nep_batch_data.cuh"
#include "main_nep/common/energy_difference.cuh"
#include "main_nep/common/nep_compile.cuh"
#include "nep_evaluate.cuh"
#include "main_nep/common/nep_common.cuh"
#include <memory>

struct NEP_Temperature_Data {
  GPU_Vector<float> temperature;
};

// Modes retained inside ordinary NEP: potential and temperature-dependent free energy.
class NEP_Potential_Parameters : public NEP_Energy_Parameters
{
public:
  int model_type = 0;
};

class NEP : public Potential
{
public:
  explicit NEP(const ParameterInput& input);
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
  void configure(NEP_Potential_Parameters& para);
  void initialize(NEP_Potential_Parameters& para, int N, int deviceCount);
  void find_force(
    NEP_Potential_Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount);
  std::unique_ptr<NEP_Potential_Parameters> model_parameters;
  std::unique_ptr<NEP_Training> training;
  NEP_Energy_Loss energy_loss;
  NEP_Batch_Data<NEP_Temperature_Data> temperature_data;
  EnergyDifference energy_difference;
  NEP_Neighbor neighbor[16];
  NEP_Descriptor_Parameters paramb;
  NEP_ANN annmb[16];
  NEP_Data nep_data[16];
  NEP_ZBL zbl;
  std::unique_ptr<NEP_Compiled_Evaluation> compiled_kernel_;
  void initialize_model(const NEP_Potential_Parameters& para);
  void initialize_device(const NEP_Potential_Parameters& para, int N, int device_id);
  void initialize_compiled_kernel(const NEP_Potential_Parameters& para);
};
