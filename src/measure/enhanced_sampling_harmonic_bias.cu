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

/*----------------------------------------------------------------------------80
Harmonic bias for enhanced sampling.
------------------------------------------------------------------------------*/

#include "enhanced_sampling_harmonic_bias.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

static __global__ void gpu_enhanced_sampling_evaluate_harmonic_bias(
  const int argument_index,
  const double center,
  const double kappa,
  const double* g_cv_values,
  double* g_cv_derivatives,
  double* g_bias_energy,
  double* g_bias_derivative)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const double displacement = g_cv_values[argument_index] - center;
    const double derivative = kappa * displacement;
    g_bias_energy[0] = 0.5 * derivative * displacement;
    g_bias_derivative[0] = derivative;
    g_cv_derivatives[argument_index] += derivative;
  }
}

EnhancedSamplingHarmonicBias::EnhancedSamplingHarmonicBias(
  const std::string& name,
  const std::string& argument_name,
  const int argument_index,
  const double center,
  const double kappa)
  : EnhancedSamplingBias(name, argument_name, argument_index),
    center_(center),
    kappa_(kappa)
{
  if (argument_index < 0) {
    PRINT_INPUT_ERROR("The argument index for a harmonic bias should be non-negative.\n");
  }
  if (kappa < 0.0) {
    PRINT_INPUT_ERROR("Kappa for a harmonic bias should be non-negative.\n");
  }
}

void EnhancedSamplingHarmonicBias::evaluate(
  const double* g_cv_values,
  double* g_cv_derivatives,
  double* g_bias_energy,
  double* g_bias_derivative)
{
  gpu_enhanced_sampling_evaluate_harmonic_bias<<<1, 1>>>(
    get_argument_index(),
    center_,
    kappa_,
    g_cv_values,
    g_cv_derivatives,
    g_bias_energy,
    g_bias_derivative);
  GPU_CHECK_KERNEL
}
