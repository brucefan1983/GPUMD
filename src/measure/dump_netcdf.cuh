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

#ifdef USE_NETCDF

#pragma once

#include "utilities/gpu_vector.cuh"
#include <vector>
class Box;

class DUMP_NETCDF
{
public:
  void parse(const char** param, int num_param);
  void preprocess(const int number_of_atoms);
  void process(
    const int step,
    const double global_time,
    const Box& box,
    const std::vector<int>& cpu_type,
    GPU_Vector<double>& position_per_atom,
    std::vector<double>& cpu_position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    std::vector<double>& cpu_velocity_per_atom);
  void postprocess();

private:
  bool dump_ = false;
  int interval = 1;          // output interval
  int has_velocity_ = 0;     // 0 wthout velocities, 1 with velocities
  char file_position[200];
  int precision = 2;         // 1 = single precision, 2 = double

  int ncid;                  // NetCDF ID
  static bool append;

  // dimensions
  int frame_dim;
  int spatial_dim;
  int atom_dim;
  int cell_spatial_dim;
  int cell_angular_dim;
  int label_dim;

  // label variables
  int spatial_var;
  int cell_spatial_var;
  int cell_angular_var;

  // data variables
  int time_var;
  int cell_lengths_var;
  int cell_angles_var;
  int coordinates_var;
  int velocities_var;
  int type_var;

  size_t lenp; // frame number

  void open_file(int frame_in_run);
  void write(
    const double global_time,
    const Box& box,
    const std::vector<int>& cpu_type,
    GPU_Vector<double>& position_per_atom,
    std::vector<double>& cpu_position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    std::vector<double>& cpu_velocity_per_atom);
};

#endif
