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
#include "property.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Group;

class DUMP_NETCDF : public Property
{
public:
  DUMP_NETCDF(const char** param, int num_param, const std::vector<Group>& groups);
  void parse(const char** param, int num_param, const std::vector<Group>& groups);
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

private:
  bool dump_ = false;
  int grouping_method_ = -1;
  int group_id_ = 0;
  int interval_ = 1;
  int has_velocity_ = 0;
  int precision_ = 1;          // 1 = single precision, 2 = double
  int compression_level_ = -1; // -1 = classic NetCDF, 0-9 = NetCDF4 deflate
  int number_of_atoms_to_dump_ = 0;
  std::string filename_;

  std::vector<int> cpu_type_to_dump_;
  std::vector<double> cpu_group_position_;
  std::vector<double> cpu_group_velocity_;
  std::vector<float> cpu_position_float_;
  std::vector<float> cpu_velocity_float_;
  std::vector<double> cpu_position_double_;
  std::vector<double> cpu_velocity_double_;
  GPU_Vector<double> group_position_;
  GPU_Vector<double> group_velocity_;

  int ncid = -1; // NetCDF ID
  static std::vector<std::string> initialized_files_;

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

  void create_file();
  void load_file_definition();
  void validate_file_definition();
  void write(
    const double global_time,
    const Box& box,
    const std::vector<int>& cpu_type,
    const std::vector<double>& cpu_position_per_atom,
    const std::vector<double>& cpu_velocity_per_atom);
};

#endif
