/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

namespace
{
__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

__global__ void gpu_vector_sum(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] + b[n];
}

__global__ void gpu_pairwise_product(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] * b[n];
}

void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

__global__ void gpu_sum(const int size, double* a, double* result)
{
  int number_of_patches = (size - 1) / 1024 + 1;
  int tid = threadIdx.x;
  int n, patch;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * 1024;
    if (n < size)
      data[tid] += a[n];
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

double sum(GPU_Vector<double>& a)
{
  double ret;
  GPU_Vector<double> result(1);
  gpu_sum<<<1, 1024>>>(a.size(), a.data(), result.data());
  result.copy_to_host(&ret);
  return ret;
}

double dot(GPU_Vector<double>& a, GPU_Vector<double>& b)
{
  GPU_Vector<double> temp(a.size());
  pairwise_product(a, b, temp);
  return sum(temp);
}

void scalar_multiply(const double& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = b.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(size, a, b.data(), c.data());
}

void vector_sum(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_vector_sum<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}
} // namespace

void Minimizer_FIRE::compute(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  double next_dt;
  const int size = number_of_atoms_ * 3;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;
  // create a velocity vector in GPU
  GPU_Vector<double> v(size, 0);
  GPU_Vector<double> temp1(size);
  GPU_Vector<double> temp2(size);

  printf("\nEnergy minimization started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    force.compute(
      box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);
    calculate_force_square_max(force_per_atom);
    const double force_max = sqrt(cpu_force_square_max_[0]);
    calculate_total_potential(potential_per_atom);

    if (step % base == 0 || force_max < force_tolerance_) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        step,
        cpu_total_potential_[0],
        force_max);
      if (force_max < force_tolerance_)
        break;
    }

    P = dot(v, force_per_atom);

    if (P > 0) {
      if (N_neg > N_min) {
        next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha *= f_alpha;
      }
      N_neg++;
    } else {
      next_dt = dt * f_dec;
      if (next_dt > dt_min)
        dt = next_dt;
      alpha = alpha_start;
      // move position back
      scalar_multiply(-0.5 * dt, v, temp1);
      vector_sum(position_per_atom, temp1, position_per_atom);
      v.fill(0);
      N_neg = 0;
    }

    // md step
    // implicit Euler integration
    double F_modulus = sqrt(dot(force_per_atom, force_per_atom));
    double v_modulus = sqrt(dot(v, v));
    // dv = F/m*dt
    scalar_multiply(dt / m, force_per_atom, temp2);
    vector_sum(v, temp2, v);
    scalar_multiply(1 - alpha, v, temp1);
    scalar_multiply(alpha * v_modulus / F_modulus, force_per_atom, temp2);
    vector_sum(temp1, temp2, v);
    // dx = v*dt
    scalar_multiply(dt, v, temp1);
    vector_sum(position_per_atom, temp1, position_per_atom);
  }

  printf("Energy minimization finished.\n");
}