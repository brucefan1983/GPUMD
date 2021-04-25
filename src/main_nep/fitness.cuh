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
#include "dataset.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <stdio.h>
#include <vector>

class Parameters;

class Fitness
{
public:
  Fitness(char*, Parameters& para);
  ~Fitness();
  void compute(const int generation, Parameters& para, const float*, float*);
  void report_error(
    char* input_dir,
    Parameters& para,
    const int generation,
    const float loss_total,
    const float loss_L1,
    const float loss_L2,
    const float,
    const float,
    const float,
    const float* elite);

protected:
  // output files:
  FILE* fid_train_out;
  FILE* fid_potential_out;

  // functions related to fitness evaluation
  void predict_energy_or_stress(FILE*, float*, float*);

  // other classes
  std::unique_ptr<Potential> potential;
  Dataset data_set; // the whole data set, which is divided into a training set and a test set
  Dataset train_set;
  Dataset test_set;
};
