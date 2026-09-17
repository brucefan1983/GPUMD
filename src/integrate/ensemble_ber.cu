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
The Berendsen thermostat and barostat:
[1] H. J. C. Berendsen et al. J. Chem. Phys. 81, 3684 (1984).
------------------------------------------------------------------------------*/

#include "ensemble_ber.cuh"
#include "npt_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

Ensemble_BER::Ensemble_BER(const char** param, int num_param, const Box& box)
{
  parse(param, num_param, box);
}

void Ensemble_BER::parse(const char** param, int num_param, const Box& box)
{
  num_target_pressure_components = 0;

  if (strcmp(param[1], "nvt_ber") == 0) {
    type = EnsembleType::NVT_BER;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_ber should have 3 parameters.");
    }
  } else if (strcmp(param[1], "npt_ber") == 0) {
    type = EnsembleType::NPT_BER;
    if (num_param != 18 && num_param != 12 && num_param != 8) {
      PRINT_INPUT_ERROR("ensemble npt_ber should have 6, 10, or 16 parameters.");
    }
  } else {
    PRINT_INPUT_ERROR("Invalid Berendsen ensemble type.");
  }

  if (!is_valid_real(param[2], &temperature1_)) {
    PRINT_INPUT_ERROR("Initial temperature should be a number.");
  }
  if (temperature1_ <= 0.0) {
    PRINT_INPUT_ERROR("Initial temperature should > 0.");
  }
  if (!is_valid_real(param[3], &temperature2_)) {
    PRINT_INPUT_ERROR("Final temperature should be a number.");
  }
  if (temperature2_ <= 0.0) {
    PRINT_INPUT_ERROR("Final temperature should > 0.");
  }

  double tau_temperature;
  if (!is_valid_real(param[4], &tau_temperature)) {
    PRINT_INPUT_ERROR("Temperature coupling should be a number.");
  }
  if (tau_temperature < 1.0) {
    PRINT_INPUT_ERROR(
      "Temperature coupling should >= 1. \n(We have changed the convention for this "
      "input starting from GPUMD-V3.0; See the manual for details.)");
  }

  temperature = temperature1_;
  temperature_coupling = 1.0 / tau_temperature;

  if (type == EnsembleType::NVT_BER) {
    printf("Use NVT ensemble for this run.\n");
    printf("    choose the Berendsen method.\n");
    printf("    initial temperature is %g K.\n", temperature1_);
    printf("    final temperature is %g K.\n", temperature2_);
    printf("    tau_T is %g time_step.\n", tau_temperature);
    return;
  }

  double elastic_modulus[6];
  if (num_param == 12) {
    num_target_pressure_components = 3;
    for (int i = 0; i < num_target_pressure_components; ++i) {
      if (!is_valid_real(param[5 + i], &target_pressure[i])) {
        PRINT_INPUT_ERROR("Pressure should be a number.");
      }
    }
    for (int i = 0; i < num_target_pressure_components; ++i) {
      if (!is_valid_real(param[8 + i], &elastic_modulus[i])) {
        PRINT_INPUT_ERROR("elastic modulus should be a number.");
      }
      if (elastic_modulus[i] <= 0.0) {
        PRINT_INPUT_ERROR("elastic modulus should > 0.");
      }
    }
    if (
      box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 || box.cpu_h[5] != 0 ||
      box.cpu_h[6] != 0 || box.cpu_h[7] != 0) {
      PRINT_INPUT_ERROR("Cannot use triclinic box with only 3 target pressure components.");
    }
  } else if (num_param == 8) {
    num_target_pressure_components = 1;
    if (!is_valid_real(param[5], &target_pressure[0])) {
      PRINT_INPUT_ERROR("Pressure should be a number.");
    }
    if (!is_valid_real(param[6], &elastic_modulus[0])) {
      PRINT_INPUT_ERROR("elastic modulus should be a number.");
    }
    if (elastic_modulus[0] <= 0.0) {
      PRINT_INPUT_ERROR("elastic modulus should > 0.");
    }
    if (
      box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 || box.cpu_h[5] != 0 ||
      box.cpu_h[6] != 0 || box.cpu_h[7] != 0) {
      PRINT_INPUT_ERROR("Cannot use triclinic box with only 1 target pressure component.");
    }
    if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
      PRINT_INPUT_ERROR(
        "Cannot use isotropic pressure with non-periodic boundary in any direction.");
    }
  } else {
    num_target_pressure_components = 6;
    for (int i = 0; i < num_target_pressure_components; ++i) {
      if (!is_valid_real(param[5 + i], &target_pressure[i])) {
        PRINT_INPUT_ERROR("Pressure should be a number.");
      }
    }
    for (int i = 0; i < num_target_pressure_components; ++i) {
      if (!is_valid_real(param[11 + i], &elastic_modulus[i])) {
        PRINT_INPUT_ERROR("elastic modulus should be a number.");
      }
      if (elastic_modulus[i] <= 0.0) {
        PRINT_INPUT_ERROR("elastic modulus should > 0.");
      }
    }
    if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
      PRINT_INPUT_ERROR(
        "Cannot use 6 pressure components with non-periodic boundary in any direction.");
    }
  }

  double tau_pressure;
  const int index_pressure_coupling = num_target_pressure_components * 2 + 5;
  if (!is_valid_real(param[index_pressure_coupling], &tau_pressure)) {
    PRINT_INPUT_ERROR("Pressure coupling should be a number.");
  }
  if (tau_pressure < 1.0) {
    PRINT_INPUT_ERROR(
      "Pressure coupling should >= 1. \n(We have changed the convention for this "
      "input starting from GPUMD-V3.0; See the manual for details.)");
  }
  for (int i = 0; i < num_target_pressure_components; ++i) {
    pressure_coupling[i] = 1.0 / (tau_pressure * 3.0 * elastic_modulus[i]);
    if (elastic_modulus[i] > 2.0e3) {
      pressure_coupling[i] = 0.0;
    }
  }

  if (tau_temperature <= 100000) {
    printf("Use NPT ensemble for this run.\n");
    printf("    choose the Berendsen method.\n");
    printf("    initial temperature is %g K.\n", temperature1_);
    printf("    final temperature is %g K.\n", temperature2_);
    printf("    tau_T is %g time_step\n", tau_temperature);
  } else {
    printf("Use NPH ensemble for this run.\n");
    printf("    choose the Berendsen method.\n");
    printf("    initial temperature is %g K but will not be used.\n", temperature1_);
    printf("    final temperature is %g K but will not be used.\n", temperature2_);
    printf("    tau_T is %g time_step but will not be used.\n", tau_temperature);
  }
  if (num_target_pressure_components == 1) {
    printf("    isotropic pressure is %g GPa.\n", target_pressure[0]);
    printf("    bulk modulus is %g GPa.\n", elastic_modulus[0]);
  } else if (num_target_pressure_components == 3) {
    printf("    pressure_xx is %g GPa.\n", target_pressure[0]);
    printf("    pressure_yy is %g GPa.\n", target_pressure[1]);
    printf("    pressure_zz is %g GPa.\n", target_pressure[2]);
    printf("    modulus_xx is %g GPa.\n", elastic_modulus[0]);
    printf("    modulus_yy is %g GPa.\n", elastic_modulus[1]);
    printf("    modulus_zz is %g GPa.\n", elastic_modulus[2]);
  } else {
    printf("    pressure_xx is %g GPa.\n", target_pressure[0]);
    printf("    pressure_yy is %g GPa.\n", target_pressure[1]);
    printf("    pressure_zz is %g GPa.\n", target_pressure[2]);
    printf("    pressure_yz is %g GPa.\n", target_pressure[3]);
    printf("    pressure_xz is %g GPa.\n", target_pressure[4]);
    printf("    pressure_xy is %g GPa.\n", target_pressure[5]);
    printf("    modulus_xx is %g GPa.\n", elastic_modulus[0]);
    printf("    modulus_yy is %g GPa.\n", elastic_modulus[1]);
    printf("    modulus_zz is %g GPa.\n", elastic_modulus[2]);
    printf("    modulus_yz is %g GPa.\n", elastic_modulus[3]);
    printf("    modulus_xz is %g GPa.\n", elastic_modulus[4]);
    printf("    modulus_xy is %g GPa.\n", elastic_modulus[5]);
  }
  printf("    tau_p is %g time_step.\n", tau_pressure);

  for (int i = 0; i < num_target_pressure_components; ++i) {
    target_pressure[i] /= PRESSURE_UNIT_CONVERSION;
    pressure_coupling[i] *= PRESSURE_UNIT_CONVERSION;
  }
}

double Ensemble_BER::get_temperature1() const
{
  return temperature1_;
}

double Ensemble_BER::get_temperature2() const
{
  return temperature2_;
}

int Ensemble_BER::get_num_target_pressure_components() const
{
  return num_target_pressure_components;
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

static void cpu_pressure_orthogonal(
  int deform_x,
  int deform_y,
  int deform_z,
  Box& box,
  double* p0,
  double* p_coupling,
  double* thermo,
  double* scale_factor)
{
  double p[3];
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 3, gpuMemcpyDeviceToHost));

  // Disable the barostat components controlled by deform.
  if (deform_x) {
    scale_factor[0] = 1.0;
  } else if (box.pbc_x == 1) {
    scale_factor[0] = 1.0 - p_coupling[0] * (p0[0] - p[0]);
    box.cpu_h[0] *= scale_factor[0];
  } else {
    scale_factor[0] = 1.0;
  }

  if (deform_y) {
    scale_factor[1] = 1.0;
  } else if (box.pbc_y == 1) {
    scale_factor[1] = 1.0 - p_coupling[1] * (p0[1] - p[1]);
    box.cpu_h[4] *= scale_factor[1];
  } else {
    scale_factor[1] = 1.0;
  }

  if (deform_z) {
    scale_factor[2] = 1.0;
  } else if (box.pbc_z == 1) {
    scale_factor[2] = 1.0 - p_coupling[2] * (p0[2] - p[2]);
    box.cpu_h[8] *= scale_factor[2];
  } else {
    scale_factor[2] = 1.0;
  }

  box.get_inverse();
}

static void cpu_pressure_isotropic(
  Box& box, double* p0, double* p_coupling, double* thermo, double& scale_factor)
{
  double p[3];
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 3, gpuMemcpyDeviceToHost));
  scale_factor = 1.0 - p_coupling[0] * (p0[0] - (p[0] + p[1] + p[2]) * 0.3333333333333333);
  box.cpu_h[0] *= scale_factor;
  box.cpu_h[4] *= scale_factor;
  box.cpu_h[8] *= scale_factor;
  box.get_inverse();
}

static void cpu_pressure_triclinic(
  int deform_x,
  int deform_y,
  int deform_z,
  int deform_xy,
  int deform_xz,
  int deform_yz,
  Box& box,
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

void Ensemble_BER::compute1(
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

void Ensemble_BER::compute2(
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

  find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);

  if (temperature_coupling > 1.0e-5) {
    gpu_berendsen_temperature<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      temperature,
      temperature_coupling,
      thermo.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + number_of_atoms,
      atom.velocity_per_atom.data() + 2 * number_of_atoms);
    GPU_CHECK_KERNEL
  }

  if (type == EnsembleType::NPT_BER) {
    if (num_target_pressure_components == 1) {
      double scale_factor;
      cpu_pressure_isotropic(box, target_pressure, pressure_coupling, thermo.data(), scale_factor);
      gpu_pressure_isotropic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms,
        scale_factor,
        atom.position_per_atom.data(),
        atom.position_per_atom.data() + number_of_atoms,
        atom.position_per_atom.data() + number_of_atoms * 2);
    } else if (num_target_pressure_components == 3) {
      double scale_factor[3];
      cpu_pressure_orthogonal(
        deform_x,
        deform_y,
        deform_z,
        box,
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
        deform_x,
        deform_y,
        deform_z,
        deform_xy,
        deform_xz,
        deform_yz,
        box,
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
}
