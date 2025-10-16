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

#include "model/atom.cuh"
#include "model/group.cuh"
#include <string>
#include <cstdio> 
#include <vector>

class Deposit
{
public:
  void parse(const char** param, int num_param, const Atom& atom, const std::vector<Group>& group);
  void compute(int step, Atom& atom, std::vector<Group>& group);
  void finalize();

private:
  bool active_ = false;
  std::string symbol_;
  double range_[3][2] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  double velocity_input_[3] = {0.0, 0.0, 0.0};
  double velocity_natural_[3] = {0.0, 0.0, 0.0};
  int interval_ = 0;
  int next_step_ = -1;
  int type_index_ = -1;
  double mass_ = 0.0;
  float charge_ = 0.0f;
  bool has_group_target_ = false;
  int group_method_target_ = -1;
  int group_label_target_ = -1;

  void zero_total_linear_momentum(
    std::vector<double>& velocity, const std::vector<double>& mass, int number_of_atoms);
};
