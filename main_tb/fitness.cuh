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
#include "tb.cuh"

#include <iosfwd>
#include <memory>
#include <vector>

class Parameters;

class Fitness
{
public:
  struct Breakdown {
    double band_rmse = 0.0;
    double dispersion_rmse = 0.0;
    double data_loss = 0.0;
  };

  Fitness(
    const Parameters& para,
    const DatasetCollection& dataset,
    const DatasetCollection* test = nullptr);

  void initialize_parameters(std::vector<double>& parameters) const;
  void print_training_header(int steps_per_epoch, int plateau_patience) const;
  bool begin_epoch(int epoch);
  void compute(const double* parameters, std::vector<double>& gradient);
  double report_epoch(
    int epoch,
    const double* parameters,
    double learning_rate,
    double mean_abs_parameter,
    double epoch_seconds);
  void next_batch() { batch_.next(); }

  double evaluate(const double* parameters, Breakdown* breakdown = nullptr)
  {
    return evaluate_data(train_, parameters, breakdown, nullptr, false);
  }

  double evaluate_with_gradient(
    const double* parameters, std::vector<double>& gradient, Breakdown* breakdown = nullptr)
  {
    return evaluate_data(train_, parameters, breakdown, &gradient, false);
  }

  double evaluate_full_k(const double* parameters, Breakdown* breakdown = nullptr)
  {
    return evaluate_data(train_, parameters, breakdown, nullptr, true);
  }
  void write_final_outputs(const double* parameters);
  void predict();

  int estimated_steps_per_epoch() const { return batch_.estimated_steps_per_epoch(); }

private:
  const Parameters& para_;
  struct Data {
    const DatasetCollection& dataset_;
    std::vector<std::unique_ptr<TB>> models_;
    std::vector<std::vector<float>> eigenvalues_;
    Data(const Parameters& para, const DatasetCollection& dataset);
  };
  Data train_;
  std::unique_ptr<Data> test_;
  DatasetCollection::Batch batch_;
  double epoch_loss_sum_ = 0.0;
  double epoch_band_sum_ = 0.0;
  double epoch_dispersion_sum_ = 0.0;
  int epoch_steps_ = 0;

  int num_reference_bands_ = 0;
  int active_num_reference_bands_ = 0;
  double active_band_weight_sum_ = 0.0;

  void initialize_global_scaler();
  void set_q_scaler(const std::vector<float>& scaler);
  void set_active_band_count(int count);
  int curriculum_epoch_count() const;

  double evaluate_data(
    Data& data,
    const double* parameters,
    Breakdown* breakdown,
    std::vector<double>* gradient,
    bool full);
  double band_mse_from_eigenvalues(
    const Data& data,
    int structure_index,
    const std::vector<float>& eigenvalues,
    const std::vector<int>& k_indices,
    int band_count,
    double band_weight_sum) const;
  double evaluate_dispersion(
    const Data& data,
    int structure_index,
    const std::vector<float>& eigenvalues,
    const std::vector<int>& k_indices,
    int band_count,
    double& gradient_denominator,
    std::vector<double>* gradient = nullptr,
    double prefactor = 0.0) const;
  void accumulate_eigenvalue_gradient(
    const Data& data,
    int structure_index,
    const std::vector<float>& eigenvalues,
    const std::vector<int>& k_indices,
    double band_rmse,
    double dispersion_rmse,
    double dispersion_denominator,
    int structure_count,
    std::vector<double>& eigenvalue_gradient) const;

  void write_bands(
    const Data& data,
    std::ostream& output,
    int structure_index,
    const std::vector<float>& eigenvalues) const;
  void write_outputs(Data& data, const char* label, const double* parameters);
};
