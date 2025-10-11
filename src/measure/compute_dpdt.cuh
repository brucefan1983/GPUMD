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

class Compute_dpdt : public Property
{
public:
  Compute_dpdt(const char**, int);

  int sample_interval;

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

  void parse(const char**, int);

private:
  GPU_Vector<float> gpu_dpdt_per_atom;
  GPU_Vector<float> gpu_dpdt_total;
  std::vector<float> cpu_dpdt_total;
  double p_integral[3];
  double p_integral_dt;
  double p_integral_time;
  FILE* fid;
};
