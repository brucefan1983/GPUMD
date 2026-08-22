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
The FIRE (fast inertial relaxation engine) minimizer
Reference: PhysRevLett 97, 170201 (2006)
           Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>

namespace
{
__global__ void gpu_multiply(
  const int size, const double factor, const double* input, double* output)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    output[n] = input[n] * factor;
}

__global__ void gpu_vector_sum(
  const int size, const double* first, const double* second, double* output)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    output[n] = first[n] + second[n];
}

__global__ void gpu_pairwise_product(
  const int size, const double* first, const double* second, double* output)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    output[n] = first[n] * second[n];
}

__global__ void gpu_sum(const int size, const double* input, double* result)
{
  const int number_of_patches = (size - 1) / 1024 + 1;
  const int tid = threadIdx.x;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * 1024;
    if (n < size)
      data[tid] += input[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      data[tid] += data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = data[0];
}
} // namespace

double Minimizer_FIRE::dot(
  const GPU_Vector<double>& first,
  const GPU_Vector<double>& second,
  const gpuStream_t stream)
{
  const int size = first.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128, 0, stream>>>(
    size, first.data(), second.data(), pair_product_.data());
  gpu_sum<<<1, 1024, 0, stream>>>(size, pair_product_.data(), scalar_result_.data());
  GPU_CHECK_KERNEL
  if (stream == nullptr) {
    scalar_result_.copy_to_host(host_scalar_.data());
  } else {
    scalar_result_.copy_to_host_async(host_scalar_.data(), stream);
    CHECK(gpuStreamSynchronize(stream));
  }
  return host_scalar_[0];
}

void Minimizer_FIRE::scalar_multiply(
  const double factor,
  const GPU_Vector<double>& input,
  GPU_Vector<double>& output,
  const gpuStream_t stream)
{
  const int size = input.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128, 0, stream>>>(
    size, factor, input.data(), output.data());
  GPU_CHECK_KERNEL
}

void Minimizer_FIRE::vector_sum(
  const GPU_Vector<double>& first,
  const GPU_Vector<double>& second,
  GPU_Vector<double>& output,
  const gpuStream_t stream)
{
  const int size = first.size();
  gpu_vector_sum<<<(size - 1) / 128 + 1, 128, 0, stream>>>(
    size, first.data(), second.data(), output.data());
  GPU_CHECK_KERNEL
}

void Minimizer_FIRE::compute(
  Force& force,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& position_per_atom,
  std::vector<Group>& group)
{
  (void)compute(force, box, atom, position_per_atom, group, nullptr, true);
}

Minimization_Result Minimizer_FIRE::compute(
  Force& force,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& position_per_atom,
  std::vector<Group>& group,
  const gpuStream_t stream,
  const bool verbose)
{
  Minimization_Result result;
  double dt = dt_0;
  double alpha = alpha_start;
  int number_of_positive_steps = 0;
  const int output_interval = number_of_steps_ >= 10 ? number_of_steps_ / 10 : 1;

  if (stream == nullptr)
    velocity_.fill(0.0);
  else
    velocity_.fill_async(0.0, stream);

  if (verbose)
    printf("\nEnergy minimization started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    if (stream == nullptr) {
      force.compute(
        box,
        position_per_atom,
        atom.type,
        group,
        atom.potential_per_atom,
        atom.force_per_atom,
        atom.virial_per_atom);
      calculate_force_square_max(atom.force_per_atom);
      calculate_total_potential(atom.potential_per_atom);
    } else {
      force.compute(
        box,
        position_per_atom,
        atom.type,
        group,
        atom.potential_per_atom,
        atom.force_per_atom,
        atom.virial_per_atom,
        stream);
      calculate_force_square_max(atom.force_per_atom, stream);
      calculate_total_potential(atom.potential_per_atom, stream);
    }

    result.force_max = std::sqrt(cpu_force_square_max_[0]);
    result.total_potential = cpu_total_potential_[0];
    result.steps = step;
    if (!std::isfinite(result.force_max) || !std::isfinite(result.total_potential))
      PRINT_INPUT_ERROR("FIRE minimization encountered a non-finite value.");

    if (
      verbose &&
      (step == 0 || (step + 1) % output_interval == 0 ||
       result.force_max < force_tolerance_)) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        step == 0 ? 0 : (step + 1),
        result.total_potential,
        result.force_max);
    }
    if (result.force_max < force_tolerance_) {
      result.converged = true;
      break;
    }

    const double power = dot(velocity_, atom.force_per_atom, stream);
    if (power > 0.0) {
      if (number_of_positive_steps > N_min) {
        const double next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha *= f_alpha;
      }
      ++number_of_positive_steps;
    } else {
      const double next_dt = dt * f_dec;
      if (next_dt > dt_min)
        dt = next_dt;
      alpha = alpha_start;
      scalar_multiply(-0.5 * dt, velocity_, temp1_, stream);
      vector_sum(position_per_atom, temp1_, position_per_atom, stream);
      if (stream == nullptr)
        velocity_.fill(0.0);
      else
        velocity_.fill_async(0.0, stream);
      number_of_positive_steps = 0;
    }

    const double force_squared = dot(atom.force_per_atom, atom.force_per_atom, stream);
    const double velocity_squared = dot(velocity_, velocity_, stream);
    if (force_squared <= 0.0) {
      result.converged = true;
      result.force_max = 0.0;
      break;
    }
    const double force_modulus = std::sqrt(force_squared);
    const double velocity_modulus = std::sqrt(std::max(0.0, velocity_squared));

    scalar_multiply(dt / m, atom.force_per_atom, temp2_, stream);
    vector_sum(velocity_, temp2_, velocity_, stream);
    scalar_multiply(1.0 - alpha, velocity_, temp1_, stream);
    scalar_multiply(
      alpha * velocity_modulus / force_modulus,
      atom.force_per_atom,
      temp2_,
      stream);
    vector_sum(temp1_, temp2_, velocity_, stream);
    scalar_multiply(dt, velocity_, temp1_, stream);
    vector_sum(position_per_atom, temp1_, position_per_atom, stream);
    result.steps = step + 1;
  }

  CHECK(gpuStreamSynchronize(stream));
  if (verbose)
    printf("Energy minimization finished.\n");
  return result;
}
