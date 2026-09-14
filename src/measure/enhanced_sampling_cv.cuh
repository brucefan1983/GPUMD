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
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>

class EnhancedSamplingCV
{
public:
  explicit EnhancedSamplingCV(const std::string& name) : name_(name) {}
  virtual ~EnhancedSamplingCV() {}

  const std::string& get_name() const { return name_; }

  virtual void prepare(const int number_of_atoms) = 0;

  virtual void compute(
    const Box& box,
    const GPU_Vector<double>& position_per_atom,
    double* g_value) = 0;

  virtual void add_force_virial(
    const double* g_derivative,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    double* g_bias_virial) = 0;

  virtual void check() {}

private:
  std::string name_;
};
