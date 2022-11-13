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

class CVAC
{
public:
  bool compute_ = false;
  int sample_interval_ = 1;
  int num_correlation_steps_ = 100;
  int grouping_method_ = -1;
  int group_id_ = -1;

  void preprocess(const int num_atoms, const double time_step, const std::vector<Group>& groups);
  void process(
    const int step, const std::vector<Group>& groups, const GPU_Vector<double>& velocity_per_atom);
  void postprocess(const char*);
  void parse(const char** param, const int num_param, const std::vector<Group>& groups);

private:
  int num_atoms_;
  GPU_Vector<float> vx_, vy_, vz_;
  GPU_Vector<float> vacx_, vacy_, vacz_;
};
