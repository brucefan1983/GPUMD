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
#include "parse_utilities.cuh"
#include "property.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Atom;
class Box;
class Group;

class DUMP_NETCDF : public Action
{
public:
  DUMP_NETCDF(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom);
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
  bool is_nep_charge_ = false;
  int grouping_method_ = -1;
  int group_id_ = 0;
  int interval_ = 1;
  int precision_ = 1;          // 1 = single precision, 2 = double
  int compression_level_ = -1; // -1 = classic NetCDF, 0-9 = NetCDF4 deflate
  int number_of_atoms_to_dump_ = 0;
  int number_of_atoms_ = 0;
  int number_of_grouping_methods_ = 0;
  DumpQuantities quantities_;
  std::string filename_;
  // the requested quantities in a canonical order, stored in the file as gpumd_quantities and
  // compared against when appending
  std::string quantity_list_;

  // maps an atom of the output to its index in the full system. Used for the quantities that are
  // host-resident anyway, namely the types, the masses and the group labels, and uploaded to the
  // device to drive the gather for everything else.
  std::vector<int> dump_indices_;

  std::vector<int> cpu_type_to_dump_;
  std::vector<int> cpu_group_labels_;

  // Staging for one quantity of one frame, holding only the atoms that are written. When a group
  // is selected its values are gathered into the device buffer first, so that the copy across the
  // bus is the size of the group rather than of the system. One buffer per element type is enough,
  // since the quantities are staged and written one at a time.
  GPU_Vector<int> gpu_dump_indices_;
  GPU_Vector<double> gather_double_;
  GPU_Vector<float> gather_float_;
  std::vector<double> host_double_;
  std::vector<float> host_float_;

  // scratch for one variable of one frame, in the element type the file uses; only the one
  // matching precision_ is allocated
  std::vector<float> pack_float_;
  std::vector<double> pack_double_;

  int ncid = -1; // NetCDF ID
  static std::vector<std::string> active_files_;

  // dimensions
  int frame_dim;
  int spatial_dim;
  int atom_dim;
  int cell_spatial_dim;
  int cell_angular_dim;
  int label_dim;
  int grouping_method_dim;

  // label variables
  int spatial_var;
  int cell_spatial_var;
  int cell_angular_var;

  // data variables
  int time_var;
  int cell_lengths_var;
  int cell_angles_var;
  int coordinates_var;
  int type_var;
  int velocities_var;
  int forces_var;
  int unwrapped_coordinates_var;
  int potential_var;
  int charge_var;
  int bec_var;
  int virial_var;
  int mass_var;
  int group_labels_var;

  size_t lenp; // frame number

  void create_file(const std::vector<Group>& groups);
  void define_per_frame_variable(
    const char* name, const int rank, const int values_per_atom, const char* units, int& var);
  void load_file_definition();
  void validate_file_definition();
  void append_mismatch(const char* what);
  void put_packed(const int var, const size_t* start, const size_t* count);
  template <typename T>
  void
  stage(GPU_Vector<T>& source, const int components, GPU_Vector<T>& scratch, std::vector<T>& host);
  void write(const double global_time, const Box& box, Atom& atom, Force& force);
};

#endif
