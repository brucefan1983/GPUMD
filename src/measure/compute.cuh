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
#include "property.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Compute : public Property
{
public:
  int compute_temperature = 0;
  int compute_potential = 0;
  int compute_force = 0;
  int compute_virial = 0;
  int compute_jp = 0;
  int compute_jk = 0;
  int compute_momentum = 0;

  int sample_interval = 1;
  int output_interval = 1;
  int grouping_method = 0;

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void process(
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
      Force& force);

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);


  Compute(const char**, int, const std::vector<Group>& group);
  void parse(const char**, int, const std::vector<Group>& group);

private:
  FILE* fid;

  std::vector<double> cpu_group_sum;
  std::vector<double> cpu_group_sum_ave;
  GPU_Vector<double> gpu_group_sum;
  GPU_Vector<double> gpu_per_atom_x;
  GPU_Vector<double> gpu_per_atom_y;
  GPU_Vector<double> gpu_per_atom_z;

  int number_of_scalars = 0;

  void output_results(const double energy_transferred[], const std::vector<Group>& group);
};
