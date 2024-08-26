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

#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Atom;
class Force;
class Integrate;

class Dump_Observer
{
public:
  void parse(const char** param, int num_param);
  void preprocess(const int number_of_atoms, const int number_of_files, Force& force);
  void process(
    int step,
    const double global_time,
    const int number_of_atoms_fixed,
    std::vector<Group>& group,
    Box& box,
    Atom& atom,
    Force& force,
    Integrate& integrate,
    GPU_Vector<double>& thermo);
  void postprocess();

private:
  bool dump_ = false;
  int dump_interval_thermo_ = 1;
  int dump_interval_exyz_ = 1;
  int has_velocity_ = 0;
  int has_force_ = 0;
  std::vector<FILE*> exyz_files_;
  std::vector<FILE*> thermo_files_;
  std::vector<double> cpu_force_per_atom_;
  GPU_Vector<double> gpu_total_virial_;
  std::vector<double> cpu_total_virial_;
  std::string mode_ = "observe"; // observe or average
  void output_line2(
    const double time,
    const Box& box,
    const std::vector<std::string>& cpu_atom_symbol,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& gpu_thermo,
    FILE* fid_);
  void write_exyz(
    const int step,
    const double global_time,
    const Box& box,
    const std::vector<std::string>& cpu_atom_symbol,
    const std::vector<int>& cpu_type,
    GPU_Vector<double>& position_per_atom,
    std::vector<double>& cpu_position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    std::vector<double>& cpu_velocity_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& gpu_thermo,
    const int file_index);
  void write_thermo(
    const int step,
    const int number_of_atoms,
    const int number_of_atoms_fixed,
    const Box& box,
    GPU_Vector<double>& gpu_thermo,
    const int file_index);
};
