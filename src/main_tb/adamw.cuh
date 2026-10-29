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

#include <vector>

class Parameters;
class Fitness;

class AdamW
{
public:
  AdamW(const Parameters& para, Fitness& fitness);
  void train();

private:
  const Parameters& para_;
  Fitness& fitness_;
  int number_of_variables_ = 0;
  int steps_per_epoch_ = 1;
  int resolved_plateau_patience_ = 5;
  std::vector<double> parameters_;
  std::vector<double> first_moment_;
  std::vector<double> second_moment_;
  std::vector<double> gradient_;

  double current_learning_rate_ = 0.0;
  double scheduler_plateau_best_ = 1.0e30;
  int scheduler_bad_epochs_ = 0;

  double mean_abs_parameter() const;
  void adamw_update(int step_index, double learning_rate);

  void update_epoch_scheduler(double epoch_average_loss);
  bool plateau_at_min_learning_rate() const;
};
