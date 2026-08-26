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
The stochastic cell rescaling barostat (combined with the BDP thermostat):
[1] Mattia Bernetti and Giovanni Bussi,
Pressure control using stochastic cell rescaling,
J. Chem. Phys. 153, 114107 (2020).
------------------------------------------------------------------------------*/

#include "ensemble_npt_scr.cuh"
#include "npt_utilities.cuh"
#include "svr_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <cstring>

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
  double temperature_input,
  double temperature_coupling_input,
  double target_pressure_input[6],
  int num_target_pressure_components_input,
  double pressure_coupling_input[6],
  int deform_x_input,
  int deform_y_input,
  int deform_z_input,
  int deform_xy_input,
  int deform_xz_input,
  int deform_yz_input)
{
  type = type_input;
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
  deform_xy = deform_xy_input;
  deform_xz = deform_xz_input;
  deform_yz = deform_yz_input;

  initialize_rng();
}

Ensemble_NPT_SCR::~Ensemble_NPT_SCR(void)
{
  // nothing now
}

static void cpu_pressure_orthogonal(
  std::mt19937& rng,
  int deform_x,
  int deform_y,
  int deform_z,
  Box& box,
  double target_temperature,
  double* p0,
  double* p_coupling,
  double* thermo,
  double* scale_factor)
{
  double p[3];
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 3, gpuMemcpyDeviceToHost));
  const double volume = box.get_volume();

  // Disable the barostat components controlled by deform.
  if (deform_x) {
    scale_factor[0] = 1.0;
  } else if (box.pbc_x == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[0] * (p0[0] - p[0]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[0] * K_B * target_temperature / volume) * gasdev(rng);
    scale_factor[0] = scale_factor_Berendsen + scale_factor_stochastic;
    box.cpu_h[0] *= scale_factor[0];
  } else {
    scale_factor[0] = 1.0;
  }

  if (deform_y) {
    scale_factor[1] = 1.0;
  } else if (box.pbc_y == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[1] * (p0[1] - p[1]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[1] * K_B * target_temperature / volume) * gasdev(rng);
    scale_factor[1] = scale_factor_Berendsen + scale_factor_stochastic;
    box.cpu_h[4] *= scale_factor[1];
  } else {
    scale_factor[1] = 1.0;
  }

  if (deform_z) {
    scale_factor[2] = 1.0;
  } else if (box.pbc_z == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[2] * (p0[2] - p[2]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[2] * K_B * target_temperature / volume) * gasdev(rng);
    scale_factor[2] = scale_factor_Berendsen + scale_factor_stochastic;
    box.cpu_h[8] *= scale_factor[2];
  } else {
    scale_factor[2] = 1.0;
  }

  box.get_inverse();
}

static void cpu_pressure_isotropic(
  std::mt19937& rng,
  Box& box,
  double target_temperature,
  double* target_pressure,
  double* p_coupling,
  double* thermo,
  double& scale_factor)
{
  double p[3];
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 3, gpuMemcpyDeviceToHost));
  const double pressure_instant = (p[0] + p[1] + p[2]) * 0.3333333333333333;
  const double scale_factor_Berendsen =
    1.0 - p_coupling[0] * (target_pressure[0] - pressure_instant);
  // The factor 0.666666666666667 is 2/3, where 3 means the number of directions that are coupled
  const double scale_factor_stochastic =
    sqrt(0.666666666666667 * p_coupling[0] * K_B * target_temperature / box.get_volume()) *
    gasdev(rng);
  scale_factor = scale_factor_Berendsen + scale_factor_stochastic;
  box.cpu_h[0] *= scale_factor;
  box.cpu_h[4] *= scale_factor;
  box.cpu_h[8] *= scale_factor;
  box.get_inverse();
}

static void cpu_pressure_triclinic(
  std::mt19937& rng,
  int deform_x,
  int deform_y,
  int deform_z,
  int deform_xy,
  int deform_xz,
  int deform_yz,
  Box& box,
  double target_temperature,
  double* p0,
  double* p_coupling,
  double* thermo,
  double* mu)
{
  // p_coupling and p0 are in Voigt notation: xx, yy, zz, yz, xz, xy
  double p[6]; // but thermo is this order: xx, yy, zz, xy, xz, yz
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 6, gpuMemcpyDeviceToHost));
  mu[0] = 1.0 - p_coupling[0] * (p0[0] - p[0]);    // xx
  mu[4] = 1.0 - p_coupling[1] * (p0[1] - p[1]);    // yy
  mu[8] = 1.0 - p_coupling[2] * (p0[2] - p[2]);    // zz
  mu[3] = mu[1] = -p_coupling[5] * (p0[5] - p[3]); // xy
  mu[6] = mu[2] = -p_coupling[4] * (p0[4] - p[4]); // xz
  mu[7] = mu[5] = -p_coupling[3] * (p0[3] - p[5]); // yz
  const double volume = box.get_volume();
  mu[0] += sqrt(2.0 * p_coupling[0] * K_B * target_temperature / volume) * gasdev(rng);
  mu[4] += sqrt(2.0 * p_coupling[1] * K_B * target_temperature / volume) * gasdev(rng);
  mu[8] += sqrt(2.0 * p_coupling[2] * K_B * target_temperature / volume) * gasdev(rng);
  const double noise_yz =
    sqrt(p_coupling[3] * K_B * target_temperature / volume) * gasdev(rng);
  const double noise_xz =
    sqrt(p_coupling[4] * K_B * target_temperature / volume) * gasdev(rng);
  const double noise_xy =
    sqrt(p_coupling[5] * K_B * target_temperature / volume) * gasdev(rng);
  mu[5] += noise_yz;
  mu[7] += noise_yz;
  mu[2] += noise_xz;
  mu[6] += noise_xz;
  mu[1] += noise_xy;
  mu[3] += noise_xy;

  if (deform_x) {
    mu[0] = 1.0;
  }
  if (deform_y) {
    mu[4] = 1.0;
  }
  if (deform_z) {
    mu[8] = 1.0;
  }
  if (deform_xy) {
    mu[1] = mu[3] = 0.0;
  }
  if (deform_xz) {
    mu[2] = mu[6] = 0.0;
  }
  if (deform_yz) {
    mu[5] = mu[7] = 0.0;
  }

  double h_old[9];
  double h_old_inverse[9];
  for (int i = 0; i < 9; ++i) {
    h_old[i] = box.cpu_h[i];
    h_old_inverse[i] = box.cpu_h[i + 9];
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

  // Other barostat components can also change a box component controlled
  // by deform. Restore such components and recompute the actual affine
  // transformation when needed.
  bool need_remap = false;
  if (deform_x && box.cpu_h[0] != h_old[0]) {
    box.cpu_h[0] = h_old[0];
    need_remap = true;
  }
  if (deform_y && box.cpu_h[4] != h_old[4]) {
    box.cpu_h[4] = h_old[4];
    need_remap = true;
  }
  if (deform_z && box.cpu_h[8] != h_old[8]) {
    box.cpu_h[8] = h_old[8];
    need_remap = true;
  }
  if (deform_xy && box.cpu_h[1] != h_old[1]) {
    box.cpu_h[1] = h_old[1];
    need_remap = true;
  }
  if (deform_xz && box.cpu_h[2] != h_old[2]) {
    box.cpu_h[2] = h_old[2];
    need_remap = true;
  }
  if (deform_yz && box.cpu_h[5] != h_old[5]) {
    box.cpu_h[5] = h_old[5];
    need_remap = true;
  }

  if (need_remap) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        double tmp = 0.0;
        for (int k = 0; k < 3; ++k) {
          tmp += box.cpu_h[r * 3 + k] * h_old_inverse[k * 3 + c];
        }
        mu[r * 3 + c] = tmp;
      }
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
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
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
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);

  int N_fixed = (fixed_group == -1) ? 0 : group[fixed_grouping_method].cpu_size[fixed_group];
  find_thermo(
    true,
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);

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
      number_of_atoms,
      scale_factor,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
  } else if (num_target_pressure_components == 3) {
    double scale_factor[3];
    cpu_pressure_orthogonal(
      rng,
      deform_x,
      deform_y,
      deform_z,
      box,
      temperature,
      target_pressure,
      pressure_coupling,
      thermo.data(),
      scale_factor);
    gpu_pressure_orthogonal<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      scale_factor[0],
      scale_factor[1],
      scale_factor[2],
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
    GPU_CHECK_KERNEL
  } else {
    double mu[9];
    cpu_pressure_triclinic(
      rng,
      deform_x,
      deform_y,
      deform_z,
      deform_xy,
      deform_xz,
      deform_yz,
      box,
      temperature,
      target_pressure,
      pressure_coupling,
      thermo.data(),
      mu);
    gpu_pressure_triclinic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      mu[0],
      mu[1],
      mu[2],
      mu[3],
      mu[4],
      mu[5],
      mu[6],
      mu[7],
      mu[8],
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
  }
}
