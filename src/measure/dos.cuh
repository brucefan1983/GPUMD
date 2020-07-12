/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
#include "utilities/gpu_vector.cuh"
#include <vector>
class Group;

class DOS
{
public:
  bool compute_ = false;
  int sample_interval_ = 1;
  int num_correlation_steps_ = 100;
  double omega_max_ = 400.0;
  int grouping_method_ = -1;
  int group_id_ = -1;
  int num_dos_points_ = -1;

  void preprocess(
    const double time_step, const std::vector<Group>& group, const GPU_Vector<double>& mass);
  void process(
    const int step, const std::vector<Group>& groups, const GPU_Vector<double>& velocity_per_atom);
  void postprocess(const char*);
  void parse(char** param, const int num_param, const std::vector<Group>& groups);

private:
  int num_atoms_;
  int num_groups_;
  int num_time_origins_;
  double dt_in_natural_units_;
  double dt_in_ps_;
  void find_dos(const char*);
  GPU_Vector<double> mass_;
  GPU_Vector<double> vx_, vy_, vz_;
  GPU_Vector<double> vacx_, vacy_, vacz_;

  void parse_num_dos_points(char** param, int& k);
};
