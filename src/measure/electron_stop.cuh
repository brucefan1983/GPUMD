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

#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Electron_Stop : public Action
{
public:
  Electron_Stop(
    const char** param,
    int num_param,
    const int num_atoms,
    const int num_types);

  void setup_force(
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;


  void end_of_step(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override;

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  int num_points = 0;
  double energy_min;
  double energy_max;
  double energy_interval;
  double stopping_power_loss = 0.0;
  std::vector<double> stopping_power_cpu;
  GPU_Vector<double> stopping_power_gpu;
  GPU_Vector<double> stopping_force;
  GPU_Vector<double> stopping_loss;

  void apply_stopping_force(const double time_step, Atom& atom);
  void accumulate_stopping_loss(Atom& atom);
};
