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
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Atom;
class Group;

class Dump_XYZ : public Property
{
public:
  Dump_XYZ(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom);
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

  struct Quantities {
    bool has_velocity_ = false;
    bool has_force_ = false;
    bool has_potential_ = false;
    bool has_unwrapped_position_ = false;
    bool has_mass_ = false;
    bool has_charge_ = false;
    bool has_bec_ = false;
    bool has_virial_ = false;
    bool has_group_ = false;
  };

private:

  bool is_nep_charge = false;
  int grouping_method_ = -1;
  int group_id_ = -1;
  int dump_interval_ = 1;
  Quantities quantities;
  int separated_ = 0;
  std::string filename_;
  FILE* fid_;

  std::vector<double> cpu_unwrapped_position_;
  std::vector<double> cpu_force_per_atom_;
  std::vector<double> cpu_potential_per_atom_;
  std::vector<double> cpu_virial_per_atom_;
  GPU_Vector<double> gpu_total_virial_;
  std::vector<double> cpu_total_virial_;
  std::vector<float> cpu_bec_;

  void output_line2(
    const double time,
    const Box& box,
    std::vector<Group>& group,
    const std::vector<std::string>& cpu_atom_symbol,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& gpu_thermo);
};
