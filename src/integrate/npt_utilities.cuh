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
Some CUDA kernels for position scaling in NPT
------------------------------------------------------------------------------*/

static __global__ void gpu_pressure_orthogonal(
  const int number_of_particles,
  const double scale_factor_x,
  const double scale_factor_y,
  const double scale_factor_z,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    g_x[i] *= scale_factor_x;
    g_y[i] *= scale_factor_y;
    g_z[i] *= scale_factor_z;
  }
}

static __global__ void gpu_pressure_isotropic(
  int number_of_particles, double scale_factor, double* g_x, double* g_y, double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    g_x[i] *= scale_factor;
    g_y[i] *= scale_factor;
    g_z[i] *= scale_factor;
  }
}

static __global__ void gpu_pressure_triclinic(
  int number_of_particles,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    g_x[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_y[i] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_z[i] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
  }
}
