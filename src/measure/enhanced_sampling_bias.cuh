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
#include <string>

class EnhancedSamplingBias
{
public:
  EnhancedSamplingBias(
    const std::string& name,
    const std::string& argument_name,
    const int argument_index)
    : name_(name), argument_name_(argument_name), argument_index_(argument_index)
  {
  }

  virtual ~EnhancedSamplingBias() {}

  const std::string& get_name() const { return name_; }
  const std::string& get_argument_name() const { return argument_name_; }
  int get_argument_index() const { return argument_index_; }

  virtual void evaluate(
    const double* g_cv_values,
    double* g_cv_derivatives,
    double* g_bias_energy,
    double* g_bias_derivative) = 0;

  virtual void update(const int step, const double time) {}

private:
  std::string name_;
  std::string argument_name_;
  int argument_index_;
};
