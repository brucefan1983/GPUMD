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
#include "measure/action.cuh"
#include <cstdio>

class ComputeCmass : public Action
{
public:
  ComputeCmass(
    const std::vector<std::string>& tokens, const std::vector<Group>& groups, Atom& atom);

  void pre_run(
    int number_of_steps,
    double time_step,
    Integrate& integrate,
    std::vector<Group>& groups,
    Atom& atom,
    Box& box,
    Force& force) override;

  void end_of_step(
    int number_of_steps,
    int step,
    int fixed_group,
    int move_group,
    double global_time,
    double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& groups,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    int number_of_steps,
    double time_step,
    double temperature) override;

private:
  int sample_interval_ = 1;
  int grouping_method_ = -1;
  int group_id_ = -1;
  GPU_Vector<double> sums_;
  FILE* file_ = nullptr;
};
