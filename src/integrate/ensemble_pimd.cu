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

#include "ensemble_pimd.cuh"
#include "langevin_utilities.cuh"
#include "svr_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <cstdlib>
#include <cstring>

void Ensemble_PIMD::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

Ensemble_PIMD::Ensemble_PIMD(
  int number_of_atoms_input, int number_of_beads_input, bool thermostat_internal_input, Atom& atom)
{
  number_of_atoms = number_of_atoms_input;
  number_of_beads = number_of_beads_input;
  num_target_pressure_components = 0;
  thermostat_internal = thermostat_internal_input;
  thermostat_centroid = false;
  initialize(atom);
}

Ensemble_PIMD::Ensemble_PIMD(
  int number_of_atoms_input,
  int number_of_beads_input,
  double temperature_coupling_input,
  Atom& atom)
{
  number_of_atoms = number_of_atoms_input;
  number_of_beads = number_of_beads_input;
  num_target_pressure_components = 0;
  temperature_coupling = temperature_coupling_input;
  thermostat_internal = true;
  thermostat_centroid = true;
  initialize(atom);
}

Ensemble_PIMD::Ensemble_PIMD(
  int number_of_atoms_input,
  int number_of_beads_input,
  double temperature_coupling_input,
  int num_target_pressure_components_input,
  double target_pressure_input[6],
  double pressure_coupling_input[6],
  Atom& atom)
{
  number_of_atoms = number_of_atoms_input;
  number_of_beads = number_of_beads_input;
  temperature_coupling = temperature_coupling_input;
  num_target_pressure_components = num_target_pressure_components_input;
  for (int i = 0; i < 6; i++) {
    target_pressure[i] = target_pressure_input[i];
    pressure_coupling[i] = pressure_coupling_input[i];
  }
  thermostat_internal = true;
  thermostat_centroid = true;
  initialize(atom);
  initialize_rng();
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

  curand_states.resize(number_of_atoms);
  int grid_size = (number_of_atoms - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), number_of_atoms, rand());
  GPU_CHECK_KERNEL
}

Ensemble_PIMD::~Ensemble_PIMD(void)
{
  // nothing
}

static __global__ void gpu_nve_1(
  const int number_of_atoms,
  const int number_of_beads,
  const double omega_n,
  const double time_step,
  const double* transformation_matrix,
  const double* g_mass,
  double** force,
  double** position,
  double** velocity)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    const double half_time_step = time_step * 0.5;
    double factor = half_time_step / g_mass[n];
    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        velocity[k][index_dn] += factor * force[k][index_dn];
      }
    }

    double velocity_normal[MAX_NUM_BEADS * 3];
    double position_normal[MAX_NUM_BEADS * 3];
    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        double temp_velocity = 0.0;
        double temp_position = 0.0;
        for (int j = 0; j < number_of_beads; ++j) {
          int index_dn = d * number_of_atoms + n;
          int index_jk = j * number_of_beads + k;
          temp_velocity += velocity[j][index_dn] * transformation_matrix[index_jk];
          temp_position += position[j][index_dn] * transformation_matrix[index_jk];
        }
        int index_kd = k * 3 + d;
        velocity_normal[index_kd] = temp_velocity;
        position_normal[index_kd] = temp_position;
      }
    }

    for (int d = 0; d < 3; ++d) {
      position_normal[d] += velocity_normal[d] * time_step; // special case of k=0
    }

    for (int k = 1; k < number_of_beads; ++k) {
      double omega_k = 2.0 * omega_n * sin(k * PI / number_of_beads);
      // The exact solution is actaully not very stable:
      // double cos_factor = cos(omega_k * time_step);
      // double sin_factor = sin(omega_k * time_step);
      // The approximate solution based on Cayley is more stable:
      double cayley = 1.0 / (1 + (omega_k * half_time_step) * (omega_k * half_time_step));
      double cos_factor = cayley * (1 - (omega_k * half_time_step) * (omega_k * half_time_step));
      double sin_factor = cayley * omega_k * time_step;
      double sin_factor_times_omega = sin_factor * omega_k;
      double sin_factor_over_omega = sin_factor / omega_k;
      for (int d = 0; d < 3; ++d) {
        int index_kd = k * 3 + d;
        double vel = velocity_normal[index_kd];
        double pos = position_normal[index_kd];
        velocity_normal[index_kd] = cos_factor * vel - sin_factor_times_omega * pos;
        position_normal[index_kd] = sin_factor_over_omega * vel + cos_factor * pos;
      }
    }

    for (int j = 0; j < number_of_beads; ++j) {
      for (int d = 0; d < 3; ++d) {
        double temp_velocity = 0.0;
        double temp_position = 0.0;
        for (int k = 0; k < number_of_beads; ++k) {
          int index_jk = j * number_of_beads + k;
          int index_kd = k * 3 + d;
          temp_velocity += velocity_normal[index_kd] * transformation_matrix[index_jk];
          temp_position += position_normal[index_kd] * transformation_matrix[index_jk];
        }
        int index_dn = d * number_of_atoms + n;
        velocity[j][index_dn] = temp_velocity;
        position[j][index_dn] = temp_position;
      }
    }
  }
}

static __global__ void gpu_nve_2(
  const int number_of_atoms,
  const int number_of_beads,
  const double time_step,
  const double* g_mass,
  double** force,
  double** velocity)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    const double half_time_step = time_step * 0.5;
    double factor = half_time_step / g_mass[n];
    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        velocity[k][index_dn] += factor * force[k][index_dn];
      }
    }
  }
}

static __global__ void gpu_langevin(
  const bool thermostat_centroid,
  const int number_of_atoms,
  const int number_of_beads,
  gpurandState* g_state,
  const double temperature,
  const double temperature_coupling,
  const double omega_n,
  const double time_step,
  const double* transformation_matrix,
  const double* g_mass,
  double** velocity)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {

    double velocity_normal[MAX_NUM_BEADS * 3];

    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        double temp_velocity = 0.0;
        for (int j = 0; j < number_of_beads; ++j) {
          int index_dn = d * number_of_atoms + n;
          int index_jk = j * number_of_beads + k;
          temp_velocity += velocity[j][index_dn] * transformation_matrix[index_jk];
        }
        int index_kd = k * 3 + d;
        velocity_normal[index_kd] = temp_velocity;
      }
    }

    gpurandState state = g_state[n];
    for (int k = 0; k < number_of_beads; ++k) {
      if (k == 0 && !thermostat_centroid) {
        continue;
      }
      double c1 = (k == 0) ? exp(-0.5 / temperature_coupling)
                           : exp(-time_step * omega_n * sin(k * PI / number_of_beads));
      double c2 = sqrt((1 - c1 * c1) * K_B * temperature * number_of_beads / g_mass[n]);
      for (int d = 0; d < 3; ++d) {
        int index_kd = k * 3 + d;
        velocity_normal[index_kd] = c1 * velocity_normal[index_kd] + c2 * CURAND_NORMAL(&state);
      }
    }
    g_state[n] = state;

    for (int j = 0; j < number_of_beads; ++j) {
      for (int d = 0; d < 3; ++d) {
        double temp_velocity = 0.0;
        for (int k = 0; k < number_of_beads; ++k) {
          int index_jk = j * number_of_beads + k;
          int index_kd = k * 3 + d;
          temp_velocity += velocity_normal[index_kd] * transformation_matrix[index_jk];
        }
        int index_dn = d * number_of_atoms + n;
        velocity[j][index_dn] = temp_velocity;
      }
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
  if (i < number_of_atoms) {
    double inverse_of_total_mass = 1.0 / device_momentum_beads[0][3];
    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        g_velocity[k][i + d * number_of_atoms] -=
          device_momentum_beads[k][d] * inverse_of_total_mass;
      }
    }
  }
}

static __global__ void gpu_apply_pbc(
  const Box box, const int number_of_atoms, const int number_of_beads, double** position)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    float pos_temp[3][MAX_NUM_BEADS] = {0.0};
    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        pos_temp[d][k] = position[k][d * number_of_atoms + n];
      }
      if (k > 0) {
        double pos_diff[3] = {0.0};
        for (int d = 0; d < 3; ++d) {
          pos_diff[d] = pos_temp[d][k] - pos_temp[d][0];
        }
        apply_mic(box, pos_diff[0], pos_diff[1], pos_diff[2]);
        for (int d = 0; d < 3; ++d) {
          pos_temp[d][k] = pos_temp[d][0] + pos_diff[d];
        }
      }
      for (int d = 0; d < 3; ++d) {
        position[k][d * number_of_atoms + n] = pos_temp[d][k];
      }
    }
  }
}

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
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    double pos_ave[3] = {0.0}, vel_ave[3] = {0.0}, pot_ave = 0.0, for_ave[3] = {0.0},
           vir_ave[9] = {0.0};
    for (int k = 0; k < number_of_beads; ++k) {
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        pos_ave[d] += position[k][index_dn];
        vel_ave[d] += velocity[k][index_dn];
        for_ave[d] += force[k][index_dn];
      }
      pot_ave += potential[k][n];
      for (int d = 0; d < 9; ++d) {
        vir_ave[d] += virial[k][d * number_of_atoms + n];
      }
    }
    double number_of_beads_inverse = 1.0 / number_of_beads;
    for (int d = 0; d < 3; ++d) {
      int index_dn = d * number_of_atoms + n;
      position_averaged[index_dn] = pos_ave[d] * number_of_beads_inverse;
      velocity_averaged[index_dn] = vel_ave[d] * number_of_beads_inverse;
      force_averaged[index_dn] = for_ave[d] * number_of_beads_inverse;
    }
    potential_averaged[n] = pot_ave * number_of_beads_inverse;
    for (int d = 0; d < 9; ++d) {
      virial_averaged[d * number_of_atoms + n] = vir_ave[d] * number_of_beads_inverse;
    }
  }
}

static __global__ void gpu_find_kinetic_energy_virial_part(
  const Box box,
  const int number_of_atoms,
  const int number_of_beads,
  double** position,
  double** force,
  double* position_averaged,
  double* kinetic_energy_virial_part,
  double* virial_averaged)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    double temp_sum[9] = {0.0};
    for (int k = 0; k < number_of_beads; ++k) {
      int index_x = 0 * number_of_atoms + n;
      int index_y = 1 * number_of_atoms + n;
      int index_z = 2 * number_of_atoms + n;
      // the virial tensor:
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      temp_sum[0] -= (position[k][index_x] - position_averaged[index_x]) * force[k][index_x];
      temp_sum[1] -= (position[k][index_y] - position_averaged[index_y]) * force[k][index_y];
      temp_sum[2] -= (position[k][index_z] - position_averaged[index_z]) * force[k][index_z];
      temp_sum[3] -= (position[k][index_x] - position_averaged[index_x]) * force[k][index_y];
      temp_sum[4] -= (position[k][index_x] - position_averaged[index_x]) * force[k][index_z];
      temp_sum[5] -= (position[k][index_y] - position_averaged[index_y]) * force[k][index_z];
      temp_sum[6] -= (position[k][index_y] - position_averaged[index_y]) * force[k][index_x];
      temp_sum[7] -= (position[k][index_z] - position_averaged[index_z]) * force[k][index_x];
      temp_sum[8] -= (position[k][index_z] - position_averaged[index_z]) * force[k][index_y];
    }
    double number_of_beads_inverse = 1.0 / number_of_beads;
    for (int d = 0; d < 9; ++d) {
      virial_averaged[d * number_of_atoms + n] += temp_sum[d] * number_of_beads_inverse;
    }
    kinetic_energy_virial_part[n] =
      0.5f * (temp_sum[0] + temp_sum[1] + temp_sum[2]) * number_of_beads_inverse;
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
  std::mt19937 rng,
  Box& box,
  double target_temperature,
  double* p0,
  double* p_coupling,
  double* thermo,
  double* scale_factor)
{
  double p[3];
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 3, gpuMemcpyDeviceToHost));

  if (box.pbc_x == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[0] * (p0[0] - p[0]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[0] * K_B * target_temperature / box.get_volume()) * gasdev(rng);
    scale_factor[0] = scale_factor_Berendsen + 0.0 * scale_factor_stochastic;
    box.cpu_h[0] *= scale_factor[0];
    box.cpu_h[3] = box.cpu_h[0] * 0.5;
  } else {
    scale_factor[0] = 1.0;
  }

  if (box.pbc_y == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[1] * (p0[1] - p[1]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[1] * K_B * target_temperature / box.get_volume()) * gasdev(rng);
    scale_factor[1] = scale_factor_Berendsen + 0.0 * scale_factor_stochastic;
    box.cpu_h[1] *= scale_factor[1];
    box.cpu_h[4] = box.cpu_h[1] * 0.5;
  } else {
    scale_factor[1] = 1.0;
  }

  if (box.pbc_z == 1) {
    const double scale_factor_Berendsen = 1.0 - p_coupling[2] * (p0[2] - p[2]);
    const double scale_factor_stochastic =
      sqrt(2.0 * p_coupling[2] * K_B * target_temperature / box.get_volume()) * gasdev(rng);
    scale_factor[2] = scale_factor_Berendsen + 0.0 * scale_factor_stochastic;
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
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 3, gpuMemcpyDeviceToHost));
  const double pressure_instant = (p[0] + p[1] + p[2]) * 0.3333333333333333;
  const double scale_factor_Berendsen =
    1.0 - p_coupling[0] * (target_pressure[0] - pressure_instant);
  // The factor 0.666666666666667 is 2/3, where 3 means the number of directions that are coupled
  const double scale_factor_stochastic =
    sqrt(0.666666666666667 * p_coupling[0] * K_B * target_temperature / box.get_volume()) *
    gasdev(rng);
  scale_factor = scale_factor_Berendsen + 0.0 * scale_factor_stochastic;
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
  CHECK(gpuMemcpy(p, thermo + 2, sizeof(double) * 6, gpuMemcpyDeviceToHost));
  mu[0] = 1.0 - p_coupling[0] * (p0[0] - p[0]);    // xx
  mu[4] = 1.0 - p_coupling[1] * (p0[1] - p[1]);    // yy
  mu[8] = 1.0 - p_coupling[2] * (p0[2] - p[2]);    // zz
  mu[3] = mu[1] = -p_coupling[5] * (p0[5] - p[3]); // xy
  mu[6] = mu[2] = -p_coupling[4] * (p0[4] - p[4]); // xz
  mu[7] = mu[5] = -p_coupling[3] * (p0[3] - p[5]); // yz
  /*
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
  */
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
      g_average_position[index] *= scale_factor[d];
      for (int k = 0; k < number_of_beads; ++k) {
        g_beads_position[k][index] *= scale_factor[d];
      }
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
      g_average_position[index] *= scale_factor;
      for (int k = 0; k < number_of_beads; ++k) {
        g_beads_position[k][index] *= scale_factor;
      }
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
    double x_old = g_average_position[i];
    double y_old = g_average_position[i + number_of_particles];
    double z_old = g_average_position[i + number_of_particles * 2];
    g_average_position[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_average_position[i + number_of_particles] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_average_position[i + number_of_particles * 2] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
    for (int k = 0; k < number_of_beads; ++k) {
      double x_old = g_beads_position[k][i];
      double y_old = g_beads_position[k][i + number_of_particles];
      double z_old = g_beads_position[k][i + number_of_particles * 2];
      g_beads_position[k][i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
      g_beads_position[k][i + number_of_particles] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
      g_beads_position[k][i + number_of_particles * 2] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
    }
  }
}

void Ensemble_PIMD::langevin(const double time_step, Atom& atom)
{
  if (thermostat_internal) {
    gpu_langevin<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
      thermostat_centroid,
      number_of_atoms,
      number_of_beads,
      curand_states.data(),
      temperature,
      temperature_coupling,
      omega_n,
      time_step,
      transformation_matrix.data(),
      atom.mass.data(),
      velocity_beads.data());
    GPU_CHECK_KERNEL

    gpu_find_momentum_beads<<<number_of_beads, 1024>>>(
      number_of_atoms, atom.mass.data(), velocity_beads.data());
    GPU_CHECK_KERNEL

    gpu_correct_momentum_beads<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
      number_of_atoms, number_of_beads, velocity_beads.data());
    GPU_CHECK_KERNEL
  }
}

void Ensemble_PIMD::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  omega_n = number_of_beads * K_B * temperature / HBAR;

  langevin(time_step, atom);

  gpu_apply_pbc<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box, number_of_atoms, number_of_beads, position_beads.data());
  GPU_CHECK_KERNEL

  gpu_nve_1<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms,
    number_of_beads,
    omega_n,
    time_step,
    transformation_matrix.data(),
    atom.mass.data(),
    force_beads.data(),
    position_beads.data(),
    velocity_beads.data());
  GPU_CHECK_KERNEL
}

void Ensemble_PIMD::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  omega_n = number_of_beads * K_B * temperature / HBAR;

  gpu_nve_2<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms,
    number_of_beads,
    time_step,
    atom.mass.data(),
    force_beads.data(),
    velocity_beads.data());
  GPU_CHECK_KERNEL

  langevin(time_step, atom);

  gpu_apply_pbc<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box, number_of_atoms, number_of_beads, position_beads.data());
  GPU_CHECK_KERNEL

  gpu_average<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
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

  gpu_find_kinetic_energy_virial_part<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box,
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
      rng, box, temperature, target_pressure, pressure_coupling, thermo.data(), scale_factor);
    gpu_pressure_isotropic<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      number_of_beads,
      scale_factor,
      position_beads.data(),
      atom.position_per_atom.data());
  } else if (num_target_pressure_components == 3) {
    double scale_factor[3];
    cpu_pressure_orthogonal(
      rng, box, temperature, target_pressure, pressure_coupling, thermo.data(), scale_factor);
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
      rng, box, temperature, target_pressure, pressure_coupling, thermo.data(), mu);
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
