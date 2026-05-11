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
#include <vector>
class Group;
class Atom;

class IC : public Property
{
public:
  bool compute_ = false;
  int sample_interval_ = 1;
  int num_correlation_steps_ = 100;

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

  virtual void write(const char* filename);

  IC(const char** param, const int num_param, Atom& atom);
  void parse(const char** param, const int num_param);

private:
  int num_atoms_;
  int num_time_origins_;
  double dt_in_natural_units_;
  double dt_in_ps_;
  int target_type_ = -1; 
  GPU_Vector<int> type_atom_list_; 
  GPU_Vector<double> x_, y_, z_;
  GPU_Vector<double> msdx_, msdy_, msdz_;
  double charge_ = 0.0;
  double volume_ = 0.0;
  double temperature_ = 0.0;
};
