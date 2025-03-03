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
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Group;
class Atom;

class ADF : public Property
{

public:
  bool compute_ = false;
  bool global_ = true; // global ADF or local triple ADF
  int adf_bins_ = 30;
  int num_interval_ = 100;

  ADF(const char** param, const int num_param, Box& box, const int number_of_types);

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

  void parse(const char** param, const int num_param, Box& box, const int number_of_types);

private:
  FILE* fid;
  int num_atoms_;
  int num_triples_ = 0;
  double rc_min_ = 0.0;
  double rc_max_ = 3.0;

  std::vector<int> adf;
  std::vector<double> angle;
  std::vector<int> itype_cpu;
  std::vector<int> jtype_cpu;
  std::vector<int> ktype_cpu;
  std::vector<double> rc_min_j_cpu;
  std::vector<double> rc_max_j_cpu;
  std::vector<double> rc_min_k_cpu;
  std::vector<double> rc_max_k_cpu;

  GPU_Vector<int> adf_gpu;
  GPU_Vector<int> itype_gpu;
  GPU_Vector<int> jtype_gpu;
  GPU_Vector<int> ktype_gpu;
  GPU_Vector<double> rc_min_j_gpu;
  GPU_Vector<double> rc_max_j_gpu;
  GPU_Vector<double> rc_min_k_gpu;
  GPU_Vector<double> rc_max_k_gpu;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> NN;
  GPU_Vector<int> NL;
};