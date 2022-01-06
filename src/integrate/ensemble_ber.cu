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
The Berendsen thermostat:
[1] H. J. C. Berendsen et al. J. Chem. Phys. 81, 3684 (1984).
------------------------------------------------------------------------------*/

#include "ensemble_ber.cuh"

Ensemble_BER::Ensemble_BER(int t, int fg, double T, double Tc)
{
  type = t;
  fixed_group = fg;
  temperature = T;
  temperature_coupling = Tc;
}

Ensemble_BER::Ensemble_BER(
  int t,
  int fg,
  double T,
  double Tc,
  double target_p[6],
  int num_target_p,
  double pc,
  int dx,
  int dy,
  int dz,
  double rate)
{
  type = t;
  fixed_group = fg;
  temperature = T;
  temperature_coupling = Tc;
  for (int i = 0; i < 6; i++) {
    target_pressure[i] = target_p[i];
  }
  num_target_pressure_components = num_target_p;
  pressure_coupling = pc;
  deform_x = dx;
  deform_y = dy;
  deform_z = dz;
  deform_rate = rate;
}

Ensemble_BER::~Ensemble_BER(void)
{
  // nothing now
}

static __global__ void gpu_berendsen_temperature(
  int N,
  double temperature,
  double coupling,
  double* g_prop,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double factor = sqrt(1.0 + coupling * (temperature / g_prop[0] - 1.0));
    g_vx[i] *= factor;
    g_vy[i] *= factor;
    g_vz[i] *= factor;
  }
}

static __global__ void gpu_berendsen_pressure_orthogonal(
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

static __global__ void gpu_berendsen_pressure_isotropic(
  int number_of_particles, double scale_factor, double* g_x, double* g_y, double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    g_x[i] *= scale_factor;
    g_y[i] *= scale_factor;
    g_z[i] *= scale_factor;
  }
}

static __global__ void gpu_berendsen_pressure_triclinic(
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

static void cpu_berendsen_pressure_orthogonal(
  int deform_x,
  int deform_y,
  int deform_z,
  double deform_rate,
  Box& box,
  double* p0,
  double p_coupling,
  double* thermo,
  double* scale_factor)
{
  double p[3];
  CHECK(cudaMemcpy(p, thermo + 2, sizeof(double) * 3, cudaMemcpyDeviceToHost));

  if (deform_x) {
    scale_factor[0] = box.cpu_h[0];
    scale_factor[0] = (scale_factor[0] + deform_rate) / scale_factor[0];
    box.cpu_h[0] *= scale_factor[0];
    box.cpu_h[3] = box.cpu_h[0] * 0.5;
  } else if (box.pbc_x == 1) {
    scale_factor[0] = 1.0 - p_coupling * (p0[0] - p[0]);
    box.cpu_h[0] *= scale_factor[0];
    box.cpu_h[3] = box.cpu_h[0] * 0.5;
  } else {
    scale_factor[0] = 1.0;
  }

  if (deform_y) {
    scale_factor[1] = box.cpu_h[1];
    scale_factor[1] = (scale_factor[1] + deform_rate) / scale_factor[1];
    box.cpu_h[1] *= scale_factor[1];
    box.cpu_h[4] = box.cpu_h[1] * 0.5;
  } else if (box.pbc_y == 1) {
    scale_factor[1] = 1.0 - p_coupling * (p0[1] - p[1]);
    box.cpu_h[1] *= scale_factor[1];
    box.cpu_h[4] = box.cpu_h[1] * 0.5;
  } else {
    scale_factor[1] = 1.0;
  }

  if (deform_z) {
    scale_factor[2] = box.cpu_h[2];
    scale_factor[2] = (scale_factor[2] + deform_rate) / scale_factor[2];
    box.cpu_h[2] *= scale_factor[2];
    box.cpu_h[5] = box.cpu_h[2] * 0.5;
  } else if (box.pbc_z == 1) {
    scale_factor[2] = 1.0 - p_coupling * (p0[2] - p[2]);
    box.cpu_h[2] *= scale_factor[2];
    box.cpu_h[5] = box.cpu_h[2] * 0.5;
  } else {
    scale_factor[2] = 1.0;
  }
}

static void cpu_berendsen_pressure_isotropic(
  Box& box, double* p0, double p_coupling, double* thermo, double& scale_factor)
{
  double p[3];
  CHECK(cudaMemcpy(p, thermo + 2, sizeof(double) * 3, cudaMemcpyDeviceToHost));
  scale_factor = 1.0 - p_coupling * (p0[0] - (p[0] + p[1] + p[2]) * 0.3333333333333333);
  box.cpu_h[0] *= scale_factor;
  box.cpu_h[1] *= scale_factor;
  box.cpu_h[2] *= scale_factor;
  box.cpu_h[3] = box.cpu_h[0] * 0.5;
  box.cpu_h[4] = box.cpu_h[1] * 0.5;
  box.cpu_h[5] = box.cpu_h[2] * 0.5;
}

static void cpu_berendsen_pressure_triclinic(
  Box& box, double* p0, double p_coupling, double* thermo, double* mu)
{
  double p[6];
  CHECK(cudaMemcpy(p, thermo + 2, sizeof(double) * 6, cudaMemcpyDeviceToHost));
  mu[0] = 1.0 - p_coupling * (p0[0] - p[0]);
  mu[4] = 1.0 - p_coupling * (p0[1] - p[1]);
  mu[8] = 1.0 - p_coupling * (p0[2] - p[2]);
  mu[3] = mu[1] = -p_coupling * (p0[3] - p[3]);
  mu[6] = mu[2] = -p_coupling * (p0[4] - p[4]);
  mu[7] = mu[5] = -p_coupling * (p0[5] - p[5]);
  double h_old[9];
  for (int i = 0; i < 9; ++i) {
    h_old[i] = box.cpu_h[i];
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < 3; ++k) {
        tmp += mu[r * 3 + k] * h_old[k * 3 + c];
      }
      box.cpu_h[r * 3 + c] = tmp;
    }
  }
  box.get_inverse();
}

void Ensemble_BER::compute1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet(
    true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
}

void Ensemble_BER::compute2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  const int number_of_atoms = mass.size();

  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  find_thermo(
    box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom, thermo);
  gpu_berendsen_temperature<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, temperature, temperature_coupling, thermo.data(), velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  if (type == 11) {
    if (num_target_pressure_components == 1) {
      double scale_factor;
      cpu_berendsen_pressure_isotropic(
        box, target_pressure, pressure_coupling, thermo.data(), scale_factor);
      gpu_berendsen_pressure_isotropic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms, scale_factor, position_per_atom.data(),
        position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2);
    } else if (num_target_pressure_components == 3) {
      double scale_factor[3];
      cpu_berendsen_pressure_orthogonal(
        deform_x, deform_y, deform_z, deform_rate, box, target_pressure, pressure_coupling,
        thermo.data(), scale_factor);
      gpu_berendsen_pressure_orthogonal<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms, scale_factor[0], scale_factor[1], scale_factor[2],
        position_per_atom.data(), position_per_atom.data() + number_of_atoms,
        position_per_atom.data() + number_of_atoms * 2);
      CUDA_CHECK_KERNEL
    } else {
      double mu[9];
      cpu_berendsen_pressure_triclinic(box, target_pressure, pressure_coupling, thermo.data(), mu);
      gpu_berendsen_pressure_triclinic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms, mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6], mu[7], mu[8],
        position_per_atom.data(), position_per_atom.data() + number_of_atoms,
        position_per_atom.data() + number_of_atoms * 2);
    }
  }
}
