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

#include "ensemble_ti_spring.cuh"

namespace
{
static __global__ void gpu_add_spring_force(
  int number_of_atoms,
  double lambda,
  double* espring,
  double* k,
  double* x,
  double* y,
  double* z,
  double* x0,
  double* y0,
  double* z0,
  double* fx,
  double* fy,
  double* fz)
{
  double dx, dy, dz;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    dx = x[i] - x0[i];
    dy = y[i] - y0[i];
    dz = z[i] - z0[i];
    fx[i] = (1 - lambda) * fx[i] + lambda * (-k[i] * dx);
    fy[i] = (1 - lambda) * fy[i] + lambda * (-k[i] * dy);
    fz[i] = (1 - lambda) * fz[i] + lambda * (-k[i] * dz);
    espring[i] = k[i] * (dx * dx + dy * dy + dz * dz);
  }
}

static __global__ void gpu_get_espring_sum(const int N, double* espring)
{
  //<<<1, 1024>>>
  int tid = threadIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N)
      s_data[tid] += espring[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    espring[0] = s_data[0];
}

} // namespace

Ensemble_TI_Spring::Ensemble_TI_Spring(const char** params, int num_params)
{
  use_barostat = false;
  use_thermostat = true;
}

void Ensemble_TI_Spring::init()
{
  int N = atom->number_of_atoms;
  gpu_k.resize(N);
  gpu_espring.resize(N);
  position_0.resize(3 * N);
  CHECK(cudaMemcpy(
    position_0.data(),
    atom->position_per_atom.data(),
    sizeof(double) * position_0.size(),
    cudaMemcpyDeviceToDevice));
}

Ensemble_TI_Spring::~Ensemble_TI_Spring(void) {}

void Ensemble_TI_Spring::add_spring_force()
{
  int N = atom->number_of_atoms;
  gpu_add_spring_force(
    N,
    lambda,
    gpu_espring.data(),
    gpu_k.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    position_0.data(),
    position_0.data() + N,
    position_0.data() + 2 * N,
    atom->force_per_atom.data(),
    atom->force_per_atom.data() + N,
    atom->force_per_atom.data() + 2 * N);
}

double Ensemble_TI_Spring::get_espring_sum()
{
  double temp;
  gpu_get_espring_sum(atom->number_of_atoms, gpu_espring.data());
  gpu_espring.copy_to_host(&temp, sizeof(double));
  return temp;
}
void Ensemble_TI_Spring::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0)
    init();
  Ensemble_MTTK::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Spring::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  // modify force by spring
  add_spring_force();
  double espring = get_espring_sum();
  Ensemble_MTTK::compute2(time_step, group, box, atoms, thermo);
}
