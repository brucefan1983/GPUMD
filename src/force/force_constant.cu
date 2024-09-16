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
Use finite difference to calculate the seconod order force constants：
    Phi_ij^ab = [F_i^a(-) - F_i^a(+)] / [u_j^b(+) - u_j^b(-)]
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "force_constant.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include <vector>

static __global__ void gpu_shift_atom(const double dx, double* x) { x[0] += dx; }

static void shift_atom(
  const double dx, const size_t n2, const size_t beta, GPU_Vector<double>& position_per_atom)
{
  const int number_of_atoms = position_per_atom.size() / 3;

  if (beta == 0) {
    gpu_shift_atom<<<1, 1>>>(dx, position_per_atom.data() + n2);
    CUDA_CHECK_KERNEL
  } else if (beta == 1) {
    gpu_shift_atom<<<1, 1>>>(dx, position_per_atom.data() + number_of_atoms + n2);
    CUDA_CHECK_KERNEL
  } else {
    gpu_shift_atom<<<1, 1>>>(dx, position_per_atom.data() + number_of_atoms * 2 + n2);
    CUDA_CHECK_KERNEL
  }
}

static void get_f(
  const double dx,
  const size_t n1,
  const size_t n2,
  const size_t beta,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  Force& force,
  double* f)
{
  const int number_of_atoms = type.size();

  shift_atom(dx, n2, beta, position_per_atom);

  force.compute(
    box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);

  size_t M = sizeof(double);
  CHECK(cudaMemcpy(f + 0, force_per_atom.data() + n1, M, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(f + 1, force_per_atom.data() + n1 + number_of_atoms, M, cudaMemcpyDeviceToHost));
  CHECK(
    cudaMemcpy(f + 2, force_per_atom.data() + n1 + number_of_atoms * 2, M, cudaMemcpyDeviceToHost));

  shift_atom(-dx, n2, beta, position_per_atom);
}

void find_H12(
  const double displacement,
  const size_t n1,
  const size_t n2,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  Force& force,
  double* H12)
{
  double dx2 = displacement * 2;
  double f_positive[3];
  double f_negative[3];
  for (size_t beta = 0; beta < 3; ++beta) {
    get_f(
      -displacement,
      n1,
      n2,
      beta,
      box,
      position_per_atom,
      type,
      group,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      force,
      f_negative);

    get_f(
      displacement,
      n1,
      n2,
      beta,
      box,
      position_per_atom,
      type,
      group,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      force,
      f_positive);

    for (size_t alpha = 0; alpha < 3; ++alpha) {
      size_t index = alpha * 3 + beta;
      H12[index] = (f_negative[alpha] - f_positive[alpha]) / dx2;
    }
  }
}
