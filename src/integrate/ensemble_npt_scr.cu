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
The stochastic cell rescaling barostat (combined with the BDP thermostat):
[1] Mattia Bernetti and Giovanni Bussi,
Pressure control using stochastic cell rescaling,
J. Chem. Phys. 153, 114107 (2020).
------------------------------------------------------------------------------*/

#include "ensemble_npt_scr.cuh"
#include "npt_utilities.cuh"
#include "svr_utilities.cuh"
#include "utilities/common.cuh"
#include <chrono>

void Ensemble_NPT_SCR::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

Ensemble_NPT_SCR::Ensemble_NPT_SCR(
  int type_input,
  int fixed_group_input,
  double temperature_input,
  double temperature_coupling_input,
  double target_pressure_input[6],
  int num_target_pressure_components_input,
  double pressure_coupling_input[6],
  int deform_x_input,
  int deform_y_input,
  int deform_z_input,
  double deform_rate_input[3])
{
  type = type_input;
  fixed_group = fixed_group_input;
  temperature = temperature_input;
  temperature_coupling = temperature_coupling_input;
  for (int i = 0; i < 6; i++) {
    target_pressure[i] = target_pressure_input[i];
    pressure_coupling[i] = pressure_coupling_input[i];
  }
  num_target_pressure_components = num_target_pressure_components_input;

  deform_x = deform_x_input;
  deform_y = deform_y_input;
  deform_z = deform_z_input;
  deform_rate[0] = deform_rate_input[0];
  deform_rate[1] = deform_rate_input[1];
  deform_rate[2] = deform_rate_input[2];

  initialize_rng();
}

Ensemble_NPT_SCR::~Ensemble_NPT_SCR(void)
{
  // nothing now
}

static void cpu_pressure_orthogonal(
  std::mt19937 rng,
  int deform_x,
  int deform_y,
  int deform_z,
  double deform_rate[3],
  Box& box,
  double target_temperature,
  double* p0,
  double* p_coupling,
  double* thermo,
  double* scale_factor)
{
  double p[3];
  CHECK(cudaMemcpy(p, thermo + 2, sizeof(double) * 3, cudaMemcpyDeviceToHost));

  if (deform_x) {
    scale_factor[0] = box.cpu_h[0];
    scale_factor[0] = (scale_factor[0] + deform_rate[0]) / scale_factor[0];
    box.cpu_h[0] *= scale_factor[0];
    box.cpu_h[3] = box.cpu_h[0] * 0.5;
  } else if (box.pbc_x == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[0] * (p0[0] - p[0]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[0] * K_B * target_temperature / box.get_volume()) * gasdev(rng);
    scale_factor[0] = scale_factor_Berendsen + scale_factor_stochastic;
    box.cpu_h[0] *= scale_factor[0];
    box.cpu_h[3] = box.cpu_h[0] * 0.5;
  } else {
    scale_factor[0] = 1.0;
  }

  if (deform_y) {
    scale_factor[1] = box.cpu_h[1];
    scale_factor[1] = (scale_factor[1] + deform_rate[1]) / scale_factor[1];
    box.cpu_h[1] *= scale_factor[1];
    box.cpu_h[4] = box.cpu_h[1] * 0.5;
  } else if (box.pbc_y == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[1] * (p0[1] - p[1]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[1] * K_B * target_temperature / box.get_volume()) * gasdev(rng);
    scale_factor[1] = scale_factor_Berendsen + scale_factor_stochastic;
    box.cpu_h[1] *= scale_factor[1];
    box.cpu_h[4] = box.cpu_h[1] * 0.5;
  } else {
    scale_factor[1] = 1.0;
  }

  if (deform_z) {
    scale_factor[2] = box.cpu_h[2];
    scale_factor[2] = (scale_factor[2] + deform_rate[2]) / scale_factor[2];
    box.cpu_h[2] *= scale_factor[2];
    box.cpu_h[5] = box.cpu_h[2] * 0.5;
  } else if (box.pbc_z == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[2] * (p0[2] - p[2]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[2] * K_B * target_temperature / box.get_volume()) * gasdev(rng);
    scale_factor[2] = scale_factor_Berendsen + scale_factor_stochastic;
    box.cpu_h[2] *= scale_factor[2];
    box.cpu_h[5] = box.cpu_h[2] * 0.5;
  } else {
    scale_factor[2] = 1.0;
  }
}

static void cpu_pressure_isotropic(
  std::mt19937 rng,
  Box& box,
  double target_temperature,
  double* target_pressure,
  double* p_coupling,
  double* thermo,
  double& scale_factor)
{
  double p[3];
  CHECK(cudaMemcpy(p, thermo + 2, sizeof(double) * 3, cudaMemcpyDeviceToHost));
  const double pressure_instant = (p[0] + p[1] + p[2]) * 0.3333333333333333;
  const double scale_factor_Berendsen =
    1.0 - p_coupling[0] * (target_pressure[0] - pressure_instant);
  // The factor 0.666666666666667 is 2/3, where 3 means the number of directions that are coupled
  const double scale_factor_stochastic =
    sqrt(0.666666666666667 * p_coupling[0] * K_B * target_temperature / box.get_volume()) *
    gasdev(rng);
  scale_factor = scale_factor_Berendsen + scale_factor_stochastic;
  box.cpu_h[0] *= scale_factor;
  box.cpu_h[1] *= scale_factor;
  box.cpu_h[2] *= scale_factor;
  box.cpu_h[3] = box.cpu_h[0] * 0.5;
  box.cpu_h[4] = box.cpu_h[1] * 0.5;
  box.cpu_h[5] = box.cpu_h[2] * 0.5;
}

static void cpu_pressure_triclinic(
  std::mt19937 rng,
  Box& box,
  double target_temperature,
  double* p0,
  double* p_coupling,
  double* thermo,
  double* mu)
{
  // p_coupling and p0 are in Voigt notation: xx, yy, zz, yz, xz, xy
  double p[6]; // but thermo is this order: xx, yy, zz, xy, xz, yz
  CHECK(cudaMemcpy(p, thermo + 2, sizeof(double) * 6, cudaMemcpyDeviceToHost));
  mu[0] = 1.0 - p_coupling[0] * (p0[0] - p[0]);    // xx
  mu[4] = 1.0 - p_coupling[1] * (p0[1] - p[1]);    // yy
  mu[8] = 1.0 - p_coupling[2] * (p0[2] - p[2]);    // zz
  mu[3] = mu[1] = -p_coupling[5] * (p0[5] - p[3]); // xy
  mu[6] = mu[2] = -p_coupling[4] * (p0[4] - p[4]); // xz
  mu[7] = mu[5] = -p_coupling[3] * (p0[3] - p[5]); // yz
  double p_coupling_3by3[3][3] = {
    {p_coupling[0], p_coupling[3], p_coupling[4]},
    {p_coupling[3], p_coupling[1], p_coupling[5]},
    {p_coupling[4], p_coupling[5], p_coupling[2]}};
  const double volume = box.get_volume();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      mu[r * 3 + c] +=
        sqrt(2.0 * p_coupling_3by3[r][c] * K_B * target_temperature / volume) * gasdev(rng);
    }
  }
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

void Ensemble_NPT_SCR::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet(
    true, time_step, group, atom.mass, atom.force_per_atom, atom.position_per_atom,
    atom.velocity_per_atom);
}

void Ensemble_NPT_SCR::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  const int number_of_atoms = atom.mass.size();

  velocity_verlet(
    false, time_step, group, atom.mass, atom.force_per_atom, atom.position_per_atom,
    atom.velocity_per_atom);

  int N_fixed = (fixed_group == -1) ? 0 : group[0].cpu_size[fixed_group];
  find_thermo(
    true, box.get_volume(), group, atom.mass, atom.potential_per_atom, atom.velocity_per_atom,
    atom.virial_per_atom, thermo);

  double ek[1];
  thermo.copy_to_host(ek, 1);
  int ndeg = 3 * (number_of_atoms - N_fixed);
  ek[0] *= ndeg * K_B * 0.5;
  double sigma = ndeg * K_B * temperature * 0.5;
  double factor = resamplekin(ek[0], sigma, ndeg, temperature_coupling, rng);
  factor = sqrt(factor / ek[0]);
  scale_velocity_global(factor, atom.velocity_per_atom);

  if (num_target_pressure_components == 1) {
    double scale_factor;
    cpu_pressure_isotropic(
      rng, box, temperature, target_pressure, pressure_coupling, thermo.data(), scale_factor);
    gpu_pressure_isotropic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, scale_factor, atom.position_per_atom.data(),
      atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
  } else if (num_target_pressure_components == 3) {
    double scale_factor[3];
    cpu_pressure_orthogonal(
      rng, deform_x, deform_y, deform_z, deform_rate, box, temperature, target_pressure,
      pressure_coupling, thermo.data(), scale_factor);
    gpu_pressure_orthogonal<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, scale_factor[0], scale_factor[1], scale_factor[2],
      atom.position_per_atom.data(), atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
    CUDA_CHECK_KERNEL
  } else {
    double mu[9];
    cpu_pressure_triclinic(
      rng, box, temperature, target_pressure, pressure_coupling, thermo.data(), mu);
    gpu_pressure_triclinic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, mu[0], mu[1], mu[2], mu[3], mu[4], mu[5], mu[6], mu[7], mu[8],
      atom.position_per_atom.data(), atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
  }
}
