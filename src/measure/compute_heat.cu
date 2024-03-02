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

/*-----------------------------------------------------------------------------------------------100
Calculate the potential part of the per-atom heat current

Reference:

Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju.
Force and heat current formulas for many-body potentials in molecular dynamics simulations with
applications to thermal conductivity calculations. Phys. Rev. B 92, 094301, (2015).
https://doi.org/10.1103/PhysRevB.92.094301
--------------------------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "utilities/error.cuh"

namespace
{
static __global__ void gpu_compute_heat(
  const int N,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  double* jx_in,
  double* jx_out,
  double* jy_in,
  double* jy_out,
  double* jz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    jx_in[n] = sxx[n] * vx[n] + sxy[n] * vy[n];
    jx_out[n] = sxz[n] * vz[n];
    jy_in[n] = syx[n] * vx[n] + syy[n] * vy[n];
    jy_out[n] = syz[n] * vz[n];
    jz[n] = szx[n] * vx[n] + szy[n] * vy[n] + szz[n] * vz[n];
  }
}
} // namespace

void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom)
{
  const int N = velocity_per_atom.size() / 3;

  // the virial tensor:
  // xx xy xz    0 3 4
  // yx yy yz    6 1 5
  // zx zy zz    7 8 2
  gpu_compute_heat<<<(N - 1) / 128 + 1, 128>>>(
    N,
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2,
    heat_per_atom.data() + N * 3,
    heat_per_atom.data() + N * 4);
  CUDA_CHECK_KERNEL
}

namespace
{
static __global__ void gpu_compute_heat(
  const int N,
  const double* mass,
  const double* potential,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  double* jx,
  double* jy,
  double* jz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    double v_x = vx[n];
    double v_y = vy[n];
    double v_z = vz[n];
    double energy = mass[n] * (v_x * v_x + v_y * v_y + v_z * v_z) * 0.5 + potential[n];
    jx[n] = (energy + sxx[n]) * v_x + sxy[n] * v_y + sxz[n] * v_z;
    jy[n] = syx[n] * v_x + (energy + syy[n]) * v_y + syz[n] * v_z;
    jz[n] = szx[n] * v_x + szy[n] * v_y + (energy + szz[n]) * v_z;
  }
}
} // namespace

void compute_heat(
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom)
{
  const int N = velocity_per_atom.size() / 3;

  // the virial tensor:
  // xx xy xz    0 3 4
  // yx yy yz    6 1 5
  // zx zy zz    7 8 2
  gpu_compute_heat<<<(N - 1) / 128 + 1, 128>>>(
    N,
    mass.data(),
    potential.data(),
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2);
  CUDA_CHECK_KERNEL
}
