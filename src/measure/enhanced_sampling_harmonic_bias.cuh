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
#include "enhanced_sampling_bias.cuh"

class EnhancedSamplingHarmonicBias : public EnhancedSamplingBias
{
public:
  EnhancedSamplingHarmonicBias(
    const std::string& name,
    const std::string& argument_name,
    const int argument_index,
    const double center,
    const double kappa);

  void evaluate(
    const double* g_cv_values,
    double* g_cv_derivatives,
    double* g_bias_energy,
    double* g_bias_derivative) override;

  double get_center() const { return center_; }
  double get_kappa() const { return kappa_; }

private:
  double center_;
  double kappa_;
};
