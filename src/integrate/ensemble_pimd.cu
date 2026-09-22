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
References for implementation:
[1] The overall integration scheme:
    Ceriotti et al., J. Chem. Phys. 133, 124104 (2010).
[2] The concept of thermostatted RPMD:
    Mariana Rossi et al., J. Chem. Phys. 140, 234116 (2014).
[3] More stable free-polymer integration based on Cayley modification:
    Roman Korol et al., J. Chem Phys. 151, 124103 (2019).
------------------------------------------------------------------------------*/

#include "eco_pimd.cuh"
#include "ensemble_pimd.cuh"
#include "langevin_utilities.cuh"
#include "svr_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

void Ensemble_PIMD::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

Ensemble_PIMD::Ensemble_PIMD(const std::vector<std::string>& tokens, const Box& box)
{
  const int num_param = tokens.size();
  int pimd_num_param = num_param;

  if (tokens[1] == "rpmd") {
    type = EnsembleType::RPMD;
    thermostat_internal = false;
    thermostat_centroid = false;
    if (num_param != 3) {
      PRINT_INPUT_ERROR("ensemble rpmd should have 1 parameter.");
    }
  } else if (tokens[1] == "trpmd") {
    type = EnsembleType::TRPMD;
    thermostat_internal = true;
    thermostat_centroid = false;
    if (num_param != 3) {
      PRINT_INPUT_ERROR("ensemble trpmd should have 1 parameter.");
    }
  } else {
    type = EnsembleType::PIMD;
    thermostat_internal = true;
    thermostat_centroid = true;
    use_scr_barostat = tokens[1] == "pimd_scr";

    // Optional Eco frequencies are selected by appending
    // "eco omega_max_cm1" to an existing PIMD command.
    if (num_param >= 8 && tokens[num_param - 2] == "eco") {
      use_eco_pimd = true;
      pimd_num_param = num_param - 2;
      if (!is_valid_real(tokens[num_param - 1], &eco_omega_max_cm1)) {
        PRINT_INPUT_ERROR("Eco-PIMD omega_max should be a number in cm^-1.");
      }
    }
    if (use_scr_barostat) {
      if (pimd_num_param != 9 && pimd_num_param != 13 && pimd_num_param != 19) {
        PRINT_INPUT_ERROR(
          "ensemble pimd_scr should have 7, 11, or 17 parameters, optionally followed by "
          "eco omega_max_cm1.");
      }
    } else {
      if (
        pimd_num_param != 6 && pimd_num_param != 9 && pimd_num_param != 13 &&
        pimd_num_param != 19) {
        PRINT_INPUT_ERROR(
          "ensemble pimd should have 4, 7, 11, or 17 parameters, optionally followed by "
          "eco omega_max_cm1.");
      }
    }
    if (use_eco_pimd && eco_omega_max_cm1 <= 0.0) {
      PRINT_INPUT_ERROR("Eco-PIMD omega_max should > 0.");
    }
  }

  if (!is_valid_int(tokens[2], &number_of_beads)) {
    PRINT_INPUT_ERROR("number of beads should be an integer.");
  }
  if (number_of_beads < 2) {
    PRINT_INPUT_ERROR("number of beads should >= 2.");
  }
  if (number_of_beads > MAX_NUM_BEADS) {
    PRINT_INPUT_ERROR("number of beads should <= 128.");
  }
  if (number_of_beads % 2 != 0) {
    PRINT_INPUT_ERROR("number of beads should be an even number.");
  }

  num_target_pressure_components = 0;
  if (type == EnsembleType::PIMD) {
    if (!is_valid_real(tokens[3], &temperature1_)) {
      PRINT_INPUT_ERROR("Initial temperature should be a number.");
    }
    if (temperature1_ <= 0.0) {
      PRINT_INPUT_ERROR("Initial temperature should > 0.");
    }
    temperature = temperature1_;

    if (!is_valid_real(tokens[4], &temperature2_)) {
      PRINT_INPUT_ERROR("Final temperature should be a number.");
    }
    if (temperature2_ <= 0.0) {
      PRINT_INPUT_ERROR("Final temperature should > 0.");
    }

    if (!is_valid_real(tokens[5], &temperature_coupling)) {
      PRINT_INPUT_ERROR("Temperature coupling should be a number.");
    }
    if (temperature_coupling < 1.0) {
      PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
    }

    if (pimd_num_param >= 9) {
      if (pimd_num_param == 13) {
        for (int i = 0; i < 3; i++) {
          if (!is_valid_real(tokens[6 + i], &target_pressure[i])) {
            PRINT_INPUT_ERROR("Pressure should be a number.");
          }
        }
        for (int i = 0; i < 3; i++) {
          if (!is_valid_real(tokens[9 + i], &elastic_modulus_[i])) {
            PRINT_INPUT_ERROR("elastic modulus should be a number.");
          }
          if (elastic_modulus_[i] <= 0) {
            PRINT_INPUT_ERROR("elastic modulus should > 0.");
          }
        }
        num_target_pressure_components = 3;
        if (
          box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 || box.cpu_h[5] != 0 ||
          box.cpu_h[6] != 0 || box.cpu_h[7] != 0) {
          PRINT_INPUT_ERROR("Cannot use triclinic box with only 3 target pressure components.");
        }
      } else if (pimd_num_param == 9) {
        if (!is_valid_real(tokens[6], &target_pressure[0])) {
          PRINT_INPUT_ERROR("Pressure should be a number.");
        }
        if (!is_valid_real(tokens[7], &elastic_modulus_[0])) {
          PRINT_INPUT_ERROR("elastic modulus should be a number.");
        }
        if (elastic_modulus_[0] <= 0) {
          PRINT_INPUT_ERROR("elastic modulus should > 0.");
        }
        num_target_pressure_components = 1;
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
        for (int i = 0; i < 6; i++) {
          if (!is_valid_real(tokens[6 + i], &target_pressure[i])) {
            PRINT_INPUT_ERROR("Pressure should be a number.");
          }
        }
        for (int i = 0; i < 6; i++) {
          if (!is_valid_real(tokens[12 + i], &elastic_modulus_[i])) {
            PRINT_INPUT_ERROR("elastic modulus should be a number.");
          }
          if (elastic_modulus_[i] <= 0) {
            PRINT_INPUT_ERROR("elastic modulus should > 0.");
          }
        }
        num_target_pressure_components = 6;
        if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
          PRINT_INPUT_ERROR(
            "Cannot use 6 pressure components with non-periodic boundary in any direction.");
        }
      }

      int index_pressure_coupling = num_target_pressure_components * 2 + 6;
      if (!is_valid_real(tokens[index_pressure_coupling], &tau_p_)) {
        PRINT_INPUT_ERROR("Pressure coupling should be a number.");
      }
      if (tau_p_ < 1) {
        PRINT_INPUT_ERROR("Pressure coupling should >= 1.");
      }
      for (int i = 0; i < num_target_pressure_components; i++) {
        pressure_coupling[i] = 1.0 / (tau_p_ * 3.0 * elastic_modulus_[i]);
        if (elastic_modulus_[i] > 2.0e3) {
          pressure_coupling[i] = 0.0;
        }
      }
    }
  }

  if (type == EnsembleType::RPMD) {
    printf("Use ring-polymer MD (RPMD) for this run.\n");
    printf("    number of beads is %d.\n", number_of_beads);
  } else if (type == EnsembleType::TRPMD) {
    printf("Use thermostatted ring-polyer MD (TRPMD) for this run.\n");
    printf("    number of beads is %d.\n", number_of_beads);
  } else {
    if (pimd_num_param >= 9) {
      if (use_scr_barostat) {
        printf("Use NPT-PIMD with stochastic cell rescaling for this run.\n");
      } else {
        printf("Use NPT-PIMD for this run.\n");
      }
    } else {
      printf("Use NVT-PIMD for this run.\n");
    }
    printf("    number of beads is %d.\n", number_of_beads);
    printf("    initial temperature is %g K.\n", temperature1_);
    printf("    final temperature is %g K.\n", temperature2_);
    printf("    tau_T is %g time_step.\n", temperature_coupling);
    if (pimd_num_param >= 9) {
      if (num_target_pressure_components == 1) {
        printf("    isotropic pressure is %g GPa.\n", target_pressure[0]);
        printf("    bulk modulus is %g GPa.\n", elastic_modulus_[0]);
      } else if (num_target_pressure_components == 3) {
        printf("    pressure_xx is %g GPa.\n", target_pressure[0]);
        printf("    pressure_yy is %g GPa.\n", target_pressure[1]);
        printf("    pressure_zz is %g GPa.\n", target_pressure[2]);
        printf("    modulus_xx is %g GPa.\n", elastic_modulus_[0]);
        printf("    modulus_yy is %g GPa.\n", elastic_modulus_[1]);
        printf("    modulus_zz is %g GPa.\n", elastic_modulus_[2]);
      } else if (num_target_pressure_components == 6) {
        printf("    pressure_xx is %g GPa.\n", target_pressure[0]);
        printf("    pressure_yy is %g GPa.\n", target_pressure[1]);
        printf("    pressure_zz is %g GPa.\n", target_pressure[2]);
        printf("    pressure_yz is %g GPa.\n", target_pressure[3]);
        printf("    pressure_xz is %g GPa.\n", target_pressure[4]);
        printf("    pressure_xy is %g GPa.\n", target_pressure[5]);
        printf("    modulus_xx is %g GPa.\n", elastic_modulus_[0]);
        printf("    modulus_yy is %g GPa.\n", elastic_modulus_[1]);
        printf("    modulus_zz is %g GPa.\n", elastic_modulus_[2]);
        printf("    modulus_yz is %g GPa.\n", elastic_modulus_[3]);
        printf("    modulus_xz is %g GPa.\n", elastic_modulus_[4]);
        printf("    modulus_xy is %g GPa.\n", elastic_modulus_[5]);
      }
      printf("    tau_p is %g time_step.\n", tau_p_);

      for (int i = 0; i < num_target_pressure_components; i++) {
        target_pressure[i] /= PRESSURE_UNIT_CONVERSION;
        pressure_coupling[i] *= PRESSURE_UNIT_CONVERSION;
      }
    }

    if (use_eco_pimd) {
      printf("    use Eco-PIMD internal-mode frequencies.\n");
      printf("    Eco-PIMD omega_max is %g cm^-1.\n", eco_omega_max_cm1);
    }
  }
}

void Ensemble_PIMD::initialize_run(
  const double, Atom& atom, Box&, const std::vector<Group>&)
{
  number_of_atoms = atom.number_of_atoms;
  initialize(atom);
  if (num_target_pressure_components > 0) {
    initialize_rng();
  }
}

void Ensemble_PIMD::initialize(Atom& atom)
{
  kinetic_energy_virial_part.resize(number_of_atoms);
  sum_1024.resize(8 * 1024); // potential, kinetic, and 6 virial components, each with 1024 data

  position_beads.resize(number_of_beads);
  velocity_beads.resize(number_of_beads);
  potential_beads.resize(number_of_beads);
  force_beads.resize(number_of_beads);
  virial_beads.resize(number_of_beads);

  std::vector<double*> position_beads_cpu(number_of_beads);
  std::vector<double*> velocity_beads_cpu(number_of_beads);
  std::vector<double*> potential_beads_cpu(number_of_beads);
  std::vector<double*> force_beads_cpu(number_of_beads);
  std::vector<double*> virial_beads_cpu(number_of_beads);

  if (atom.number_of_beads == 0) {
    if (!thermostat_centroid) {
      PRINT_INPUT_ERROR("Cannot use RPMD or TRPMD before PIMD\n.");
    }
    atom.position_beads.resize(number_of_beads);
    atom.velocity_beads.resize(number_of_beads);
    atom.potential_beads.resize(number_of_beads);
    atom.force_beads.resize(number_of_beads);
    atom.virial_beads.resize(number_of_beads);
  } else {
    if (atom.number_of_beads != number_of_beads) {
      PRINT_INPUT_ERROR("Cannot change the number of beads for PIMD runs\n.");
    }
  }

  for (int k = 0; k < number_of_beads; ++k) {
    if (atom.number_of_beads == 0) {
      atom.position_beads[k].resize(number_of_atoms * 3);
      atom.velocity_beads[k].resize(number_of_atoms * 3);
      atom.potential_beads[k].resize(number_of_atoms);
      atom.force_beads[k].resize(number_of_atoms * 3);
      atom.virial_beads[k].resize(number_of_atoms * 9);

      atom.position_beads[k].copy_from_device(atom.position_per_atom.data());
      atom.velocity_beads[k].copy_from_device(atom.velocity_per_atom.data());
      atom.force_beads[k].copy_from_device(atom.force_per_atom.data());
    }

    position_beads_cpu[k] = atom.position_beads[k].data();
    velocity_beads_cpu[k] = atom.velocity_beads[k].data();
    potential_beads_cpu[k] = atom.potential_beads[k].data();
    force_beads_cpu[k] = atom.force_beads[k].data();
    virial_beads_cpu[k] = atom.virial_beads[k].data();
  }

  atom.number_of_beads = number_of_beads;

  position_beads.copy_from_host(position_beads_cpu.data());
  velocity_beads.copy_from_host(velocity_beads_cpu.data());
  potential_beads.copy_from_host(potential_beads_cpu.data());
  force_beads.copy_from_host(force_beads_cpu.data());
  virial_beads.copy_from_host(virial_beads_cpu.data());

  transformation_matrix.resize(number_of_beads * number_of_beads);
  std::vector<double> transformation_matrix_cpu(number_of_beads * number_of_beads);
  double sqrt_factor_1 = sqrt(1.0 / number_of_beads);
  double sqrt_factor_2 = sqrt(2.0 / number_of_beads);
  for (int j = 1; j <= number_of_beads; ++j) {
    double sign_factor = (j % 2 == 0) ? 1.0 : -1.0;
    for (int k = 0; k < number_of_beads; ++k) {
      int jk = (j - 1) * number_of_beads + k;
      double pi_factor = 2.0 * PI * j * k / number_of_beads;
      if (k == 0) {
        transformation_matrix_cpu[jk] = sqrt_factor_1;
      } else if (k < number_of_beads / 2) {
        transformation_matrix_cpu[jk] = sqrt_factor_2 * cos(pi_factor);
      } else if (k == number_of_beads / 2) {
        transformation_matrix_cpu[jk] = sqrt_factor_1 * sign_factor;
      } else {
        transformation_matrix_cpu[jk] = sqrt_factor_2 * sin(pi_factor);
      }
    }
  }
  transformation_matrix.copy_from_host(transformation_matrix_cpu.data());

  if (use_eco_pimd) {
    eco_mode_factors.resize(number_of_beads);
  }

  position_normal.resize(number_of_atoms * number_of_beads * 3);
  velocity_normal.resize(number_of_atoms * number_of_beads * 3);

  curand_states.resize(number_of_atoms);
  int grid_size = (number_of_atoms - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), number_of_atoms, rand());
  GPU_CHECK_KERNEL
}

void Ensemble_PIMD::update_eco_modes()
{
  if (!use_eco_pimd) {
    return;
  }
  if (!(temperature > 0.0) || !std::isfinite(temperature)) {
    PRINT_INPUT_ERROR("Eco-PIMD requires a positive finite temperature.");
  }

  const double temperature_tolerance =
    1.0e-12 * std::max(1.0, std::fabs(temperature));
  if (std::fabs(temperature - eco_last_temperature) <= temperature_tolerance) {
    return;
  }

  // h*c/k_B in K cm; omega_max is supplied as a wavenumber in cm^-1.
  const double cm_to_kelvin = 1.4387768775039338;
  const double x_max = cm_to_kelvin * eco_omega_max_cm1 / temperature;
  Eco_PIMD_Result result =
    find_eco_pimd_frequencies(number_of_beads, x_max, eco_independent_frequencies);

  eco_mode_factors.copy_from_host(result.mode_factors.data());
  eco_independent_frequencies = std::move(result.independent_frequencies);
  eco_last_temperature = temperature;

  if (!eco_frequencies_reported) {
    printf(
      "    Eco-PIMD frequencies: T=%g K, x_max=%g, RMSE(Trotter)=%g, "
      "RMSE(Eco)=%g, Newton iterations=%d.\n",
      temperature,
      x_max,
      result.rmse_trotter,
      result.rmse_eco,
      result.number_of_iterations);
    eco_frequencies_reported = true;
  }
}

static __global__ void gpu_half_kick(
  const int number_of_atoms,
  const int number_of_beads,
  const double time_step,
  const double* g_mass,
  double** force,
  double** velocity)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (n < number_of_atoms && k < number_of_beads) {
    const double factor = (time_step * 0.5) / g_mass[n];
    for (int d = 0; d < 3; ++d) {
      const int index_dn = d * number_of_atoms + n;
      velocity[k][index_dn] += factor * force[k][index_dn];
    }
  }
}

static __global__ void gpu_nve_forward(
  const int number_of_atoms,
  const int number_of_beads,
  const double omega_n,
  const bool use_eco_pimd,
  const double* eco_mode_factors,
  const double time_step,
  const double* transformation_matrix,
  double** position,
  double** velocity,
  double* position_normal,
  double* velocity_normal)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (n < number_of_atoms && k < number_of_beads) {
    double temp_velocity[3] = {0.0};
    double temp_position[3] = {0.0};
    for (int j = 0; j < number_of_beads; ++j) {
      int index_jk = j * number_of_beads + k;
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        temp_velocity[d] += velocity[j][index_dn] * transformation_matrix[index_jk];
        temp_position[d] += position[j][index_dn] * transformation_matrix[index_jk];
      }
    }

    if (k == 0) {
      for (int d = 0; d < 3; ++d) {
        temp_position[d] += temp_velocity[d] * time_step; // special case of k=0
      }
    } else {
      const double half_time_step = time_step * 0.5;
      const double omega_k = use_eco_pimd ? omega_n * eco_mode_factors[k]
                                          : 2.0 * omega_n * sin(k * PI / number_of_beads);
      // The exact solution is actually not very stable:
      // double cos_factor = cos(omega_k * time_step);
      // double sin_factor = sin(omega_k * time_step);
      // The approximate solution based on Cayley is more stable:
      const double cayley =
        1.0 / (1 + (omega_k * half_time_step) * (omega_k * half_time_step));
      const double cos_factor =
        cayley * (1 - (omega_k * half_time_step) * (omega_k * half_time_step));
      const double sin_factor = cayley * omega_k * time_step;
      const double sin_factor_times_omega = sin_factor * omega_k;
      const double sin_factor_over_omega = sin_factor / omega_k;
      for (int d = 0; d < 3; ++d) {
        const double old_velocity = temp_velocity[d];
        const double old_position = temp_position[d];
        temp_velocity[d] = cos_factor * old_velocity - sin_factor_times_omega * old_position;
        temp_position[d] = sin_factor_over_omega * old_velocity + cos_factor * old_position;
      }
    }

    for (int d = 0; d < 3; ++d) {
      size_t index_kdn = (static_cast<size_t>(k) * 3 + d) * number_of_atoms + n;
      velocity_normal[index_kdn] = temp_velocity[d];
      position_normal[index_kdn] = temp_position[d];
    }
  }
}

static __global__ void gpu_nve_inverse(
  const int number_of_atoms,
  const int number_of_beads,
  const double* transformation_matrix,
  const double* position_normal,
  const double* velocity_normal,
  double** position,
  double** velocity)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  if (n < number_of_atoms && j < number_of_beads) {
    double temp_velocity[3] = {0.0};
    double temp_position[3] = {0.0};
    for (int k = 0; k < number_of_beads; ++k) {
      int index_jk = j * number_of_beads + k;
      for (int d = 0; d < 3; ++d) {
        size_t index_kdn = (static_cast<size_t>(k) * 3 + d) * number_of_atoms + n;
        temp_velocity[d] += velocity_normal[index_kdn] * transformation_matrix[index_jk];
        temp_position[d] += position_normal[index_kdn] * transformation_matrix[index_jk];
      }
    }
    for (int d = 0; d < 3; ++d) {
      int index_dn = d * number_of_atoms + n;
      velocity[j][index_dn] = temp_velocity[d];
      position[j][index_dn] = temp_position[d];
    }
  }
}

static __global__ void gpu_langevin_forward(
  const int number_of_atoms,
  const int number_of_beads,
  const double* transformation_matrix,
  double** velocity,
  double* velocity_normal)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int k = blockIdx.y;
  if (n < number_of_atoms && k < number_of_beads) {
    double temp_velocity[3] = {0.0};
    for (int j = 0; j < number_of_beads; ++j) {
      int index_jk = j * number_of_beads + k;
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        temp_velocity[d] += velocity[j][index_dn] * transformation_matrix[index_jk];
      }
    }
    for (int d = 0; d < 3; ++d) {
      size_t index_kdn = (static_cast<size_t>(k) * 3 + d) * number_of_atoms + n;
      velocity_normal[index_kdn] = temp_velocity[d];
    }
  }
}

static __global__ void gpu_langevin_generate_random(
  const bool thermostat_centroid,
  const int number_of_atoms,
  const int number_of_beads,
  gpurandState* g_state,
  double* random_numbers)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    gpurandState state = g_state[n];
    for (int k = 0; k < number_of_beads; ++k) {
      if (k == 0 && !thermostat_centroid) {
        continue;
      }
      for (int d = 0; d < 3; ++d) {
        size_t index_kdn = (static_cast<size_t>(k) * 3 + d) * number_of_atoms + n;
        random_numbers[index_kdn] = CURAND_NORMAL(&state);
      }
    }
    g_state[n] = state;
  }
}

static __global__ void gpu_langevin_thermostat(
  const bool thermostat_centroid,
  const int number_of_atoms,
  const int number_of_beads,
  const double temperature,
  const double temperature_coupling,
  const double omega_n,
  const bool use_eco_pimd,
  const double* eco_mode_factors,
  const double time_step,
  const double* g_mass,
  const double* random_numbers,
  double* velocity_normal)
{
  const int k = blockIdx.y;
  if (k >= number_of_beads || (k == 0 && !thermostat_centroid)) {
    return;
  }

  __shared__ double s_c1;
  __shared__ double s_thermal_numerator;
  if (threadIdx.x == 0) {
    if (k == 0) {
      s_c1 = exp(-0.5 / temperature_coupling);
    } else if (use_eco_pimd) {
      s_c1 = exp(-0.5 * time_step * omega_n * eco_mode_factors[k]);
    } else {
      s_c1 = exp(-time_step * omega_n * sin(k * PI / number_of_beads));
    }
    s_thermal_numerator =
      (1 - s_c1 * s_c1) * K_B * temperature * number_of_beads;
  }
  __syncthreads();

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    const double c2 = sqrt(s_thermal_numerator / g_mass[n]);
    for (int d = 0; d < 3; ++d) {
      size_t index_kdn = (static_cast<size_t>(k) * 3 + d) * number_of_atoms + n;
      velocity_normal[index_kdn] =
        s_c1 * velocity_normal[index_kdn] + c2 * random_numbers[index_kdn];
    }
  }
}

static __global__ void gpu_langevin_inverse(
  const int number_of_atoms,
  const int number_of_beads,
  const double* transformation_matrix,
  const double* velocity_normal,
  double** velocity)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y;
  if (n < number_of_atoms && j < number_of_beads) {
    double temp_velocity[3] = {0.0};
    for (int k = 0; k < number_of_beads; ++k) {
      int index_jk = j * number_of_beads + k;
      for (int d = 0; d < 3; ++d) {
        size_t index_kdn = (static_cast<size_t>(k) * 3 + d) * number_of_atoms + n;
        temp_velocity[d] += velocity_normal[index_kdn] * transformation_matrix[index_jk];
      }
    }
    for (int d = 0; d < 3; ++d) {
      velocity[j][d * number_of_atoms + n] = temp_velocity[d];
    }
  }
}

__device__ double device_momentum_beads[MAX_NUM_BEADS][4];

static __global__ void
gpu_find_momentum_beads(const int number_of_atoms, const double* g_mass, double** g_velocity)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_rounds = (number_of_atoms - 1) / 1024 + 1;
  __shared__ double s_momentum[4][1024];
  double momentum[4] = {0.0};

  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * 1024;
    if (n < number_of_atoms) {
      for (int d = 0; d < 3; ++d) {
        momentum[d] += g_mass[n] * g_velocity[bid][n + d * number_of_atoms];
      }
      momentum[3] += g_mass[n];
    }
  }

  for (int d = 0; d < 4; ++d) {
    s_momentum[d][tid] = momentum[d];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 4; ++d) {
        s_momentum[d][tid] += s_momentum[d][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int d = 0; d < 4; ++d) {
      device_momentum_beads[bid][d] = s_momentum[d][0];
    }
  }
}

static __global__ void gpu_correct_momentum_beads(
  const int number_of_atoms, const int number_of_beads, double** g_velocity)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y;
  if (i < number_of_atoms && k < number_of_beads) {
    double inverse_of_total_mass = 1.0 / device_momentum_beads[0][3];
    for (int d = 0; d < 3; ++d) {
      g_velocity[k][i + d * number_of_atoms] -=
        device_momentum_beads[k][d] * inverse_of_total_mass;
    }
  }
}

static __global__ void gpu_quantize_reference_position(
  const int number_of_atoms, double** position)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    for (int d = 0; d < 3; ++d) {
      int index_dn = d * number_of_atoms + n;
      float reference = position[0][index_dn];
      position[0][index_dn] = reference;
    }
  }
}

static __global__ void gpu_apply_pbc(
  const Box box, const int number_of_atoms, const int number_of_beads, double** position)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y + 1;
  if (n < number_of_atoms && k < number_of_beads) {
    float reference[3];
    float current[3];
    for (int d = 0; d < 3; ++d) {
      int index_dn = d * number_of_atoms + n;
      reference[d] = position[0][index_dn];
      current[d] = position[k][index_dn];
    }
    float dx_float = current[0] - reference[0];
    float dy_float = current[1] - reference[1];
    float dz_float = current[2] - reference[2];
    double dx = dx_float;
    double dy = dy_float;
    double dz = dz_float;
    apply_mic(box, dx, dy, dz);
    position[k][n] = static_cast<float>(reference[0] + dx);
    position[k][number_of_atoms + n] = static_cast<float>(reference[1] + dy);
    position[k][2 * number_of_atoms + n] = static_cast<float>(reference[2] + dz);
  }
}

static void apply_pbc(
  const Box& box, const int number_of_atoms, const int number_of_beads, double** position)
{
  int grid_size = (number_of_atoms - 1) / 64 + 1;
  gpu_quantize_reference_position<<<grid_size, 64>>>(number_of_atoms, position);
  GPU_CHECK_KERNEL
  if (number_of_beads > 1) {
    const dim3 grid(grid_size, number_of_beads - 1);
    gpu_apply_pbc<<<grid, 64>>>(box, number_of_atoms, number_of_beads, position);
    GPU_CHECK_KERNEL
  }
}

constexpr int PIMD_ATOM_TILE = 8;

static __global__ void gpu_average(
  const int number_of_atoms,
  const int number_of_beads,
  double** position,
  double** velocity,
  double** potential,
  double** force,
  double** virial,
  double* position_averaged,
  double* velocity_averaged,
  double* potential_averaged,
  double* force_averaged,
  double* virial_averaged)
{
  __shared__ double s_value[3][MAX_NUM_BEADS][PIMD_ATOM_TILE];
  int local_atom = threadIdx.x;
  int k = threadIdx.y;
  int n = blockIdx.x * PIMD_ATOM_TILE + local_atom;
  bool valid = n < number_of_atoms;
  double number_of_beads_inverse = 1.0 / number_of_beads;

  for (int d = 0; d < 3; ++d) {
    int index_dn = d * number_of_atoms + n;
    s_value[0][k][local_atom] = valid ? position[k][index_dn] : 0.0;
    s_value[1][k][local_atom] = valid ? velocity[k][index_dn] : 0.0;
    s_value[2][k][local_atom] = valid ? force[k][index_dn] : 0.0;
    __syncthreads();
    if (k == 0 && valid) {
      double pos_ave = 0.0;
      double vel_ave = 0.0;
      double for_ave = 0.0;
      for (int bead = 0; bead < number_of_beads; ++bead) {
        pos_ave += s_value[0][bead][local_atom];
        vel_ave += s_value[1][bead][local_atom];
        for_ave += s_value[2][bead][local_atom];
      }
      position_averaged[index_dn] = pos_ave * number_of_beads_inverse;
      velocity_averaged[index_dn] = vel_ave * number_of_beads_inverse;
      force_averaged[index_dn] = for_ave * number_of_beads_inverse;
    }
    __syncthreads();
  }

  s_value[0][k][local_atom] = valid ? potential[k][n] : 0.0;
  __syncthreads();
  if (k == 0 && valid) {
    double pot_ave = 0.0;
    for (int bead = 0; bead < number_of_beads; ++bead) {
      pot_ave += s_value[0][bead][local_atom];
    }
    potential_averaged[n] = pot_ave * number_of_beads_inverse;
  }
  __syncthreads();

  for (int group = 0; group < 3; ++group) {
    for (int lane = 0; lane < 3; ++lane) {
      int d = group * 3 + lane;
      int index_dn = d * number_of_atoms + n;
      s_value[lane][k][local_atom] = valid ? virial[k][index_dn] : 0.0;
    }
    __syncthreads();
    if (k == 0 && valid) {
      for (int lane = 0; lane < 3; ++lane) {
        int d = group * 3 + lane;
        double vir_ave = 0.0;
        for (int bead = 0; bead < number_of_beads; ++bead) {
          vir_ave += s_value[lane][bead][local_atom];
        }
        virial_averaged[d * number_of_atoms + n] = vir_ave * number_of_beads_inverse;
      }
    }
    __syncthreads();
  }
}

constexpr int PIMD_VIRIAL_ATOM_TILE = 8;

static __global__ void gpu_find_kinetic_energy_virial_part(
  const int number_of_atoms,
  const int number_of_beads,
  double** position,
  double** force,
  double* position_averaged,
  double* kinetic_energy_virial_part,
  double* virial_averaged)
{
  __shared__ double s_value[3][MAX_NUM_BEADS][PIMD_VIRIAL_ATOM_TILE];
  int local_atom = threadIdx.x;
  int k = threadIdx.y;
  int n = blockIdx.x * PIMD_VIRIAL_ATOM_TILE + local_atom;
  bool valid = n < number_of_atoms;
  double number_of_beads_inverse = 1.0 / number_of_beads;
  double diagonal_sum = 0.0;

  double contribution[9] = {0.0};
  if (valid) {
    int index_x = n;
    int index_y = number_of_atoms + n;
    int index_z = 2 * number_of_atoms + n;
    double dx = position[k][index_x] - position_averaged[index_x];
    double dy = position[k][index_y] - position_averaged[index_y];
    double dz = position[k][index_z] - position_averaged[index_z];
    double fx = force[k][index_x];
    double fy = force[k][index_y];
    double fz = force[k][index_z];
    contribution[0] = -dx * fx;
    contribution[1] = -dy * fy;
    contribution[2] = -dz * fz;
    contribution[3] = -dx * fy;
    contribution[4] = -dx * fz;
    contribution[5] = -dy * fz;
    contribution[6] = -dy * fx;
    contribution[7] = -dz * fx;
    contribution[8] = -dz * fy;
  }

  for (int group = 0; group < 3; ++group) {
    for (int lane = 0; lane < 3; ++lane) {
      s_value[lane][k][local_atom] = contribution[group * 3 + lane];
    }
    __syncthreads();
    if (k == 0 && valid) {
      for (int lane = 0; lane < 3; ++lane) {
        int d = group * 3 + lane;
        double sum = 0.0;
        for (int bead = 0; bead < number_of_beads; ++bead) {
          sum += s_value[lane][bead][local_atom];
        }
        virial_averaged[d * number_of_atoms + n] += sum * number_of_beads_inverse;
        if (group == 0) {
          diagonal_sum += sum;
        }
      }
    }
    __syncthreads();
  }

  if (k == 0 && valid) {
    kinetic_energy_virial_part[n] =
      0.5f * diagonal_sum * number_of_beads_inverse;
  }
}

static __global__ void gpu_find_sum_1024(
  const int number_of_atoms,
  const double* g_kinetic_energy_virial_part,
  const double* g_potential,
  const double* g_virial,
  double* g_sum_1024)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ double s_sum[8][128];

  double sum[8] = {0.0};
  const int stride = blockDim.x * gridDim.x;
  for (int n = bid * blockDim.x + tid; n < number_of_atoms; n += stride) {
    sum[0] += g_kinetic_energy_virial_part[n];
    sum[1] += g_potential[n];
    for (int d = 0; d < 6; ++d) {
      sum[d + 2] += g_virial[d * number_of_atoms + n];
    }
  }
  for (int d = 0; d < 8; ++d) {
    s_sum[d][tid] = sum[d];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 8; ++d) {
        s_sum[d][tid] += s_sum[d][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int d = 0; d < 8; ++d) {
      g_sum_1024[d * 1024 + bid] = s_sum[d][0];
    }
  }
}

// g_thermo[0-7] = K, U, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
static __global__ void
gpu_find_thermo(const double volume, const double NkBT, const double* g_sum_1024, double* g_thermo)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ double s_data[1024];
  s_data[tid] = g_sum_1024[bid * 1024 + tid];
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (bid == 0) {
      g_thermo[bid] = 1.5 * NkBT + s_data[0];
    } else if (bid == 1) {
      g_thermo[bid] = s_data[0];
    } else if (bid <= 4) {
      g_thermo[bid] = (NkBT + s_data[0]) / volume;
    } else {
      g_thermo[bid] = s_data[0] / volume;
    }
  }
}

static void cpu_pressure_orthogonal(
  std::mt19937& rng,
  bool use_scr_barostat,
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

  if (box.pbc_x == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[0] * (p0[0] - p[0]);
    scale_factor[0] = scale_factor_Berendsen;
    if (use_scr_barostat) {
      const double scale_factor_stochastic =
        sqrt(2.0 * p_coupling[0] * K_B * target_temperature / volume) * gasdev(rng);
      scale_factor[0] += scale_factor_stochastic;
    }
    box.cpu_h[0] *= scale_factor[0];
  } else {
    scale_factor[0] = 1.0;
  }

  if (box.pbc_y == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[1] * (p0[1] - p[1]);
    scale_factor[1] = scale_factor_Berendsen;
    if (use_scr_barostat) {
      const double scale_factor_stochastic =
        sqrt(2.0 * p_coupling[1] * K_B * target_temperature / volume) * gasdev(rng);
      scale_factor[1] += scale_factor_stochastic;
    }
    box.cpu_h[4] *= scale_factor[1];
  } else {
    scale_factor[1] = 1.0;
  }

  if (box.pbc_z == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[2] * (p0[2] - p[2]);
    scale_factor[2] = scale_factor_Berendsen;
    if (use_scr_barostat) {
      const double scale_factor_stochastic =
        sqrt(2.0 * p_coupling[2] * K_B * target_temperature / volume) * gasdev(rng);
      scale_factor[2] += scale_factor_stochastic;
    }
    box.cpu_h[8] *= scale_factor[2];
  } else {
    scale_factor[2] = 1.0;
  }
  box.get_inverse();
}

static void cpu_pressure_isotropic(
  std::mt19937& rng,
  bool use_scr_barostat,
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
  scale_factor = scale_factor_Berendsen;
  if (use_scr_barostat) {
    const double scale_factor_stochastic =
      sqrt(0.666666666666667 * p_coupling[0] * K_B * target_temperature / box.get_volume()) *
      gasdev(rng);
    scale_factor += scale_factor_stochastic;
  }
  box.cpu_h[0] *= scale_factor;
  box.cpu_h[4] *= scale_factor;
  box.cpu_h[8] *= scale_factor;
  box.get_inverse();
}

static void cpu_pressure_triclinic(
  std::mt19937& rng,
  bool use_scr_barostat,
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
  if (use_scr_barostat) {
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

static __global__ void gpu_pressure_orthogonal(
  const int number_of_particles,
  int number_of_beads,
  const double scale_factor_x,
  const double scale_factor_y,
  const double scale_factor_z,
  double** g_beads_position,
  double* g_average_position)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    double scale_factor[3] = {scale_factor_x, scale_factor_y, scale_factor_z};
    for (int d = 0; d < 3; ++d) {
      const int index = i + d * number_of_particles;
      const double displacement = (scale_factor[d] - 1.0) * g_average_position[index];
      for (int k = 0; k < number_of_beads; ++k) {
        g_beads_position[k][index] += displacement;
      }
      g_average_position[index] *= scale_factor[d];
    }
  }
}

static __global__ void gpu_pressure_isotropic(
  int number_of_particles,
  int number_of_beads,
  double scale_factor,
  double** g_beads_position,
  double* g_average_position)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    for (int d = 0; d < 3; ++d) {
      const int index = i + d * number_of_particles;
      const double displacement = (scale_factor - 1.0) * g_average_position[index];
      for (int k = 0; k < number_of_beads; ++k) {
        g_beads_position[k][index] += displacement;
      }
      g_average_position[index] *= scale_factor;
    }
  }
}

static __global__ void gpu_pressure_triclinic(
  int number_of_particles,
  int number_of_beads,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double** g_beads_position,
  double* g_average_position)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double x_old = g_average_position[i];
    const double y_old = g_average_position[i + number_of_particles];
    const double z_old = g_average_position[i + number_of_particles * 2];
    const double x_new = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    const double y_new = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    const double z_new = mu6 * x_old + mu7 * y_old + mu8 * z_old;
    const double displacement_x = x_new - x_old;
    const double displacement_y = y_new - y_old;
    const double displacement_z = z_new - z_old;
    for (int k = 0; k < number_of_beads; ++k) {
      g_beads_position[k][i] += displacement_x;
      g_beads_position[k][i + number_of_particles] += displacement_y;
      g_beads_position[k][i + number_of_particles * 2] += displacement_z;
    }
    g_average_position[i] = x_new;
    g_average_position[i + number_of_particles] = y_new;
    g_average_position[i + number_of_particles * 2] = z_new;
  }
}

void Ensemble_PIMD::langevin(const double time_step, Atom& atom)
{
  if (thermostat_internal) {
    const dim3 grid((number_of_atoms - 1) / 64 + 1, number_of_beads);
    gpu_langevin_forward<<<grid, 64>>>(
      number_of_atoms,
      number_of_beads,
      transformation_matrix.data(),
      velocity_beads.data(),
      velocity_normal.data());
    GPU_CHECK_KERNEL

    // Reuse position_normal as random-number workspace.
    gpu_langevin_generate_random<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
      thermostat_centroid,
      number_of_atoms,
      number_of_beads,
      curand_states.data(),
      position_normal.data());
    GPU_CHECK_KERNEL

    gpu_langevin_thermostat<<<grid, 64>>>(
      thermostat_centroid,
      number_of_atoms,
      number_of_beads,
      temperature,
      temperature_coupling,
      omega_n,
      use_eco_pimd,
      use_eco_pimd ? eco_mode_factors.data() : nullptr,
      time_step,
      atom.mass.data(),
      position_normal.data(),
      velocity_normal.data());
    GPU_CHECK_KERNEL

    gpu_langevin_inverse<<<grid, 64>>>(
      number_of_atoms,
      number_of_beads,
      transformation_matrix.data(),
      velocity_normal.data(),
      velocity_beads.data());
    GPU_CHECK_KERNEL

    gpu_find_momentum_beads<<<number_of_beads, 1024>>>(
      number_of_atoms, atom.mass.data(), velocity_beads.data());
    GPU_CHECK_KERNEL

    const dim3 momentum_grid((number_of_atoms - 1) / 64 + 1, number_of_beads);
    gpu_correct_momentum_beads<<<momentum_grid, 64>>>(
      number_of_atoms, number_of_beads, velocity_beads.data());
    GPU_CHECK_KERNEL
  }
}

void Ensemble_PIMD::compute1(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  omega_n = number_of_beads * K_B * temperature / HBAR;
  update_eco_modes();

  langevin(time_step, atom);

  apply_pbc(box, number_of_atoms, number_of_beads, position_beads.data());

  const dim3 grid((number_of_atoms - 1) / 64 + 1, number_of_beads);
  gpu_half_kick<<<grid, 64>>>(
    number_of_atoms,
    number_of_beads,
    time_step,
    atom.mass.data(),
    force_beads.data(),
    velocity_beads.data());
  GPU_CHECK_KERNEL

  gpu_nve_forward<<<grid, 64>>>(
    number_of_atoms,
    number_of_beads,
    omega_n,
    use_eco_pimd,
    use_eco_pimd ? eco_mode_factors.data() : nullptr,
    time_step,
    transformation_matrix.data(),
    position_beads.data(),
    velocity_beads.data(),
    position_normal.data(),
    velocity_normal.data());
  GPU_CHECK_KERNEL

  gpu_nve_inverse<<<grid, 64>>>(
    number_of_atoms,
    number_of_beads,
    transformation_matrix.data(),
    position_normal.data(),
    velocity_normal.data(),
    position_beads.data(),
    velocity_beads.data());
  GPU_CHECK_KERNEL
}

void Ensemble_PIMD::compute2(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo,
  Force& force)
{
  omega_n = number_of_beads * K_B * temperature / HBAR;

  const dim3 grid((number_of_atoms - 1) / 64 + 1, number_of_beads);
  gpu_half_kick<<<grid, 64>>>(
    number_of_atoms,
    number_of_beads,
    time_step,
    atom.mass.data(),
    force_beads.data(),
    velocity_beads.data());
  GPU_CHECK_KERNEL

  langevin(time_step, atom);

  apply_pbc(box, number_of_atoms, number_of_beads, position_beads.data());

  const dim3 average_block(PIMD_ATOM_TILE, number_of_beads);
  const dim3 average_grid((number_of_atoms - 1) / PIMD_ATOM_TILE + 1);
  gpu_average<<<average_grid, average_block>>>(
    number_of_atoms,
    number_of_beads,
    position_beads.data(),
    velocity_beads.data(),
    potential_beads.data(),
    force_beads.data(),
    virial_beads.data(),
    atom.position_per_atom.data(),
    atom.velocity_per_atom.data(),
    atom.potential_per_atom.data(),
    atom.force_per_atom.data(),
    atom.virial_per_atom.data());
  GPU_CHECK_KERNEL

  const dim3 virial_block(PIMD_VIRIAL_ATOM_TILE, number_of_beads);
  const dim3 virial_grid((number_of_atoms - 1) / PIMD_VIRIAL_ATOM_TILE + 1);
  gpu_find_kinetic_energy_virial_part<<<virial_grid, virial_block>>>(
    number_of_atoms,
    number_of_beads,
    position_beads.data(),
    force_beads.data(),
    atom.position_per_atom.data(),
    kinetic_energy_virial_part.data(),
    atom.virial_per_atom.data());
  GPU_CHECK_KERNEL

  gpu_find_sum_1024<<<1024, 128>>>(
    number_of_atoms,
    kinetic_energy_virial_part.data(),
    atom.potential_per_atom.data(),
    atom.virial_per_atom.data(),
    sum_1024.data());
  GPU_CHECK_KERNEL

  gpu_find_thermo<<<8, 1024>>>(
    box.get_volume(), number_of_atoms * K_B * temperature, sum_1024.data(), thermo.data());
  GPU_CHECK_KERNEL

  if (num_target_pressure_components == 1) {
    double scale_factor;
    cpu_pressure_isotropic(
      rng,
      use_scr_barostat,
      box,
      temperature,
      target_pressure,
      pressure_coupling,
      thermo.data(),
      scale_factor);
    gpu_pressure_isotropic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      number_of_beads,
      scale_factor,
      position_beads.data(),
      atom.position_per_atom.data());
  } else if (num_target_pressure_components == 3) {
    double scale_factor[3];
    cpu_pressure_orthogonal(
      rng,
      use_scr_barostat,
      box,
      temperature,
      target_pressure,
      pressure_coupling,
      thermo.data(),
      scale_factor);
    gpu_pressure_orthogonal<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      number_of_beads,
      scale_factor[0],
      scale_factor[1],
      scale_factor[2],
      position_beads.data(),
      atom.position_per_atom.data());
    GPU_CHECK_KERNEL
  } else if (num_target_pressure_components == 6) {
    double mu[9];
    cpu_pressure_triclinic(
      rng,
      use_scr_barostat,
      box,
      temperature,
      target_pressure,
      pressure_coupling,
      thermo.data(),
      mu);
    gpu_pressure_triclinic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      number_of_beads,
      mu[0],
      mu[1],
      mu[2],
      mu[3],
      mu[4],
      mu[5],
      mu[6],
      mu[7],
      mu[8],
      position_beads.data(),
      atom.position_per_atom.data());
  }
}
