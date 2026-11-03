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

#include "adamw.cuh"
#include "fitness.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

AdamW::AdamW(const Parameters& para, Fitness& fitness)
  : para_(para), fitness_(fitness), number_of_variables_(para.number_of_variables)
{
  fitness_.initialize_parameters(parameters_);
  first_moment_.assign(number_of_variables_, 0.0);
  second_moment_.assign(number_of_variables_, 0.0);
  gradient_.assign(number_of_variables_, 0.0);
  steps_per_epoch_ = std::max(1, fitness_.estimated_steps_per_epoch());
  resolved_plateau_patience_ =
    para_.lr_scheduler_explicit ? para_.lr_patience : (steps_per_epoch_ == 1 ? 15 : 5);
  current_learning_rate_ = para_.learning_rate;
}

double AdamW::mean_abs_parameter() const
{
  double sum = 0.0;
  for (double value : parameters_) {
    sum += std::fabs(value);
  }
  return sum / number_of_variables_;
}

void AdamW::adamw_update(const int step_index, const double learning_rate)
{
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double epsilon = 1.0e-8;
  const double corr1 = 1.0 - std::pow(beta1, step_index + 1);
  const double corr2 = 1.0 - std::pow(beta2, step_index + 1);

  for (int v = 0; v < number_of_variables_; ++v) {
    const double g = gradient_[v];

    double p_after_decay = parameters_[v];
    if (para_.weight_decay != 0.0) {
      p_after_decay *= (1.0 - learning_rate * para_.weight_decay);
    }
    first_moment_[v] = beta1 * first_moment_[v] + (1.0 - beta1) * g;
    second_moment_[v] = beta2 * second_moment_[v] + (1.0 - beta2) * g * g;
    const double mhat = first_moment_[v] / corr1;
    const double vhat = second_moment_[v] / corr2;
    parameters_[v] = p_after_decay - learning_rate * (mhat / (std::sqrt(vhat) + epsilon));
    if (!std::isfinite(parameters_[v])) {
      PRINT_INPUT_ERROR("AdamW produced a non-finite TB parameter.");
    }
  }
}

void AdamW::update_epoch_scheduler(const double epoch_average_loss)
{
  if (!std::isfinite(epoch_average_loss)) {
    return;
  }

  if (epoch_average_loss < scheduler_plateau_best_ * (1.0 - para_.lr_rel_threshold)) {
    scheduler_plateau_best_ = epoch_average_loss;
    scheduler_bad_epochs_ = 0;
  } else {
    ++scheduler_bad_epochs_;
  }

  if (scheduler_bad_epochs_ > resolved_plateau_patience_) {
    current_learning_rate_ = std::max(para_.lr_min, current_learning_rate_ * para_.lr_factor);
    scheduler_bad_epochs_ = 0;
  }
}

bool AdamW::plateau_at_min_learning_rate() const
{
  const double tolerance = std::max(1.0e-12, 1.0e-5 * std::fabs(para_.lr_min));
  return std::fabs(current_learning_rate_ - para_.lr_min) <= tolerance;
}

void AdamW::train()
{
  if (para_.maximum_epoch > std::numeric_limits<int>::max() / steps_per_epoch_) {
    PRINT_INPUT_ERROR("Requested epoch count produces too many AdamW steps.");
  }
  fitness_.print_training_header(steps_per_epoch_, resolved_plateau_patience_);

  for (int epoch_index = 0; epoch_index < para_.maximum_epoch; ++epoch_index) {
    const int epoch = epoch_index + 1;
    const bool loss_comparable = fitness_.begin_epoch(epoch);
    const double lr = current_learning_rate_;
    double epoch_seconds_sum = 0.0;

    for (int step_in_epoch = 0; step_in_epoch < steps_per_epoch_; ++step_in_epoch) {
      const int step_index = epoch_index * steps_per_epoch_ + step_in_epoch;
      const auto begin = std::chrono::steady_clock::now();
      fitness_.compute(parameters_.data(), gradient_);

      adamw_update(step_index, lr);

      const auto end = std::chrono::steady_clock::now();
      const double elapsed = std::chrono::duration<double>(end - begin).count();
      epoch_seconds_sum += elapsed;
    }

    const double average_total_loss =
      fitness_.report_epoch(epoch, parameters_.data(), lr, mean_abs_parameter(), epoch_seconds_sum);
    if (!loss_comparable) {
      scheduler_plateau_best_ = 1.0e30;
      scheduler_bad_epochs_ = 0;
      continue;
    }
    update_epoch_scheduler(average_total_loss);
    if (plateau_at_min_learning_rate()) {
      printf(
        "  Plateau scheduler reached the minimum learning rate at epoch %d; "
        "training stops at this epoch.\n",
        epoch);
      break;
    }
  }

  fitness_.write_final_outputs(parameters_.data());
}
