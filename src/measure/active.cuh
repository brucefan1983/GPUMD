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
#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Atom;
class Group;
class Force;

class Active : public Action
{
public:
  Active(const std::vector<std::string>& tokens);
  void parse(const std::vector<std::string>& tokens);
  virtual void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void end_of_step(
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

  virtual void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

private:
  int check_interval_ = 1;
  int has_velocity_ = 0;
  int has_force_ = 0;
  int has_uncertainty_ = 0;
  double threshold_ = 0.0;
  FILE* exyz_file_;
  FILE* out_file_;
  std::vector<double> cpu_position_per_atom_;
  std::vector<double> cpu_velocity_per_atom_;
  std::vector<double> cpu_force_per_atom_;
  std::vector<double> cpu_total_virial_;
  std::vector<double> cpu_uncertainty_;
  GPU_Vector<double> gpu_total_virial_;
  GPU_Vector<double> mean_force_;
  GPU_Vector<double> squared_force_deviation_sum_;
  GPU_Vector<double> gpu_uncertainty_;
  GPU_Vector<double> active_potential_per_atom_;
  GPU_Vector<double> active_force_per_atom_;
  GPU_Vector<double> active_virial_per_atom_;
  GPU_Vector<double> active_thermo_;
  void output_line2(const double time, const Box& box, double uncertainty);
  void write_exyz(const double global_time, const Box& box, Atom& atom, double uncertainty);
  void write_uncertainty(const double global_time, double uncertainty);
};
