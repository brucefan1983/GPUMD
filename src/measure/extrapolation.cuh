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

#ifdef USE_HIP
#include <hipblas/hipblas.h>
#else
#include <cublas_v2.h>
#endif
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "property.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>

class Extrapolation : public Property
{
public:
  Extrapolation(const char** params, int num_params);

  void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

  void process(
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

  FILE* f;
  std::vector<std::unique_ptr<GPU_Vector<double>>> asi_list;
  GPU_Vector<double> B;          // N x B_size
  GPU_Vector<double> gamma_full; // N x B_size
  GPU_Vector<double> gamma;      // maximum of each component: N
  GPU_Vector<double*> blas_A, blas_x, blas_y;
  Atom* atom;
  Box* box;
  int B_size_per_atom;
  int check_interval = 1;
  int dump_interval = 1;
  int last_dump = 0;
  double gamma_low = 0;
  double gamma_high = 1e100;
  double max_gamma; // global maximum
  std::string asi_file_name;
  gpublasHandle_t handle;

private:
  void load_asi();
  void dump();
  void calculate_gamma();
};
