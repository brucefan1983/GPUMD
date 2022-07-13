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
    float* elite);

protected:
  int num_batches = 0;
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size
  FILE* fid_loss_out;
  std::unique_ptr<Potential> potential;
  std::vector<Dataset> train_set;
  Dataset test_set;
  void predict_energy_or_stress(FILE* fid, float* data, float* ref, Dataset& dataset);
  void
  update_energy_force_virial(FILE* fid_energy, FILE* fid_force, FILE* fid_virial, Dataset& dataset);
};
