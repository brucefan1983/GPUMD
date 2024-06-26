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

#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cooperative_groups.h>
#include <string>
#include <vector>

class Dump_Shock_NEMD
{
public:
  void parse(const char** param, int num_param);
  void preprocess(Atom& atom, Box& box);
  void process(Atom& atom, Box& box, const int step);
  void postprocess();

private:
  bool dump_ = false;
  int n;
  int dump_interval_ = -1;
  int direction = 0;
  int bins;
  double slice_vol = 1;
  double avg_window = 10;
  FILE *temp_file, *pxx_file, *pyy_file, *pzz_file, *density_file, *com_vx_file;
  GPU_Vector<double> gpu_temp, gpu_pxx, gpu_pyy, gpu_pzz, gpu_density, gpu_com_vx, gpu_com_vy,
    gpu_com_vz, gpu_number;
  std::vector<double> cpu_temp, cpu_pxx, cpu_pyy, cpu_pzz, cpu_density, cpu_com_vx, cpu_com_vy,
    cpu_com_vz;
};