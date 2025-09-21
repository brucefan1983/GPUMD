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
#include "model/box.cuh"
#include "property.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Group;
class Atom;

class OrientOrder : public Property
{

public:
  bool compute_ = false;
  int num_interval_ = 100;

  OrientOrder(const char** param, const int num_param);

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

  void parse(const char** param, const int num_param);

private:
  int num_atoms_;
  FILE* fid;
  std::string mode_ = "cutoff";
  double rc_ = 6.0; // default cutoff for nnn mode.
  int nnn_ = 0;
  bool wl_ = false;
  bool wlhat_ = false;
  int ndegrees_;
  bool average_ = false;
  int lmax_;
  int ncol_;

  std::vector<int> llist;
  std::vector<double> qnarray;
  std::vector<double> cglist;

  GPU_Vector<int> llist_gpu;
  GPU_Vector<double> qlm_r_gpu;
  GPU_Vector<double> qlm_i_gpu;
  GPU_Vector<double> aqlm_r_gpu;
  GPU_Vector<double> aqlm_i_gpu;
  GPU_Vector<double> cglist_gpu;
  GPU_Vector<double> qnarray_gpu;

  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> NN;
  GPU_Vector<int> NL;
  GPU_Vector<double> NLD; // Neighbor distance
};