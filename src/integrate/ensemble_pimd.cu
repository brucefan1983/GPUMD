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
#include "utilities/common.cuh"
#include <cstdlib>

Ensemble_PIMD::Ensemble_PIMD(
  int number_of_atoms_input,
  int number_of_beads_input,
  int number_of_steps_pimd_input,
  double temperature_input,
  double temperature_coupling_input,
  Atom& atom)
{
  number_of_atoms = number_of_atoms_input;
  number_of_beads = number_of_beads_input;
  number_of_steps_pimd = number_of_steps_pimd_input;
  temperature = temperature_input;
  temperature_coupling = temperature_coupling_input;
  omega_n = number_of_beads * K_B * temperature / HBAR;

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

  atom.position_beads.resize(number_of_beads);
  atom.velocity_beads.resize(number_of_beads);
  atom.potential_beads.resize(number_of_beads);
  atom.force_beads.resize(number_of_beads);
  atom.virial_beads.resize(number_of_beads);

  for (int k = 0; k < number_of_beads; ++k) {
    atom.position_beads[k].resize(number_of_atoms * 3);
    atom.velocity_beads[k].resize(number_of_atoms * 3);
    atom.potential_beads[k].resize(number_of_atoms);
    atom.force_beads[k].resize(number_of_atoms * 3);
    atom.virial_beads[k].resize(number_of_atoms * 9);

    atom.position_beads[k].copy_from_device(atom.position_per_atom.data());
    atom.velocity_beads[k].copy_from_device(atom.velocity_per_atom.data());
    atom.force_beads[k].copy_from_device(atom.force_per_atom.data());

    position_beads_cpu[k] = atom.position_beads[k].data();
    velocity_beads_cpu[k] = atom.velocity_beads[k].data();
    potential_beads_cpu[k] = atom.potential_beads[k].data();
    force_beads_cpu[k] = atom.force_beads[k].data();
    virial_beads_cpu[k] = atom.virial_beads[k].data();
  }

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
  CUDA_CHECK_KERNEL
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
  const bool use_rpmd,
  const int number_of_atoms,
  const int number_of_beads,
  curandState* g_state,
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

    curandState state = g_state[n];
    for (int k = 0; k < number_of_beads; ++k) {
      if (k == 0 && use_rpmd) {
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
  double* kinetic_energy_virial_part)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    double factor = 0.5 / number_of_beads;
    double temp_sum = 0.0;
    for (int k = 0; k < number_of_beads; ++k) {
      double pos_diff[3] = {0.0};
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        pos_diff[d] = position[k][index_dn] - position_averaged[index_dn];
      }
      apply_mic(box, pos_diff[0], pos_diff[1], pos_diff[2]);
      for (int d = 0; d < 3; ++d) {
        int index_dn = d * number_of_atoms + n;
        temp_sum -= pos_diff[d] * force[k][index_dn];
      }
    }
    kinetic_energy_virial_part[n] = temp_sum * factor;
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
    } else if (bid > 1) {
      g_thermo[bid] = (NkBT + s_data[0]) / volume;
    }
  }
}

void Ensemble_PIMD::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  static int num_calls = 0;
  bool use_rpmd = num_calls >= number_of_steps_pimd;

  gpu_langevin<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    use_rpmd, number_of_atoms, number_of_beads, curand_states.data(), temperature,
    temperature_coupling, omega_n, time_step, transformation_matrix.data(), atom.mass.data(),
    velocity_beads.data());
  CUDA_CHECK_KERNEL

  ++num_calls;

  gpu_apply_pbc<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box, number_of_atoms, number_of_beads, position_beads.data());
  CUDA_CHECK_KERNEL

  gpu_nve_1<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms, number_of_beads, omega_n, time_step, transformation_matrix.data(),
    atom.mass.data(), force_beads.data(), position_beads.data(), velocity_beads.data());
  CUDA_CHECK_KERNEL
}

void Ensemble_PIMD::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  gpu_nve_2<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms, number_of_beads, time_step, atom.mass.data(), force_beads.data(),
    velocity_beads.data());
  CUDA_CHECK_KERNEL

  static int num_calls = 0;
  bool use_rpmd = num_calls >= number_of_steps_pimd;

  gpu_langevin<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    use_rpmd, number_of_atoms, number_of_beads, curand_states.data(), temperature,
    temperature_coupling, omega_n, time_step, transformation_matrix.data(), atom.mass.data(),
    velocity_beads.data());
  CUDA_CHECK_KERNEL

  ++num_calls;

  // TODO: correct momentum

  gpu_apply_pbc<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box, number_of_atoms, number_of_beads, position_beads.data());
  CUDA_CHECK_KERNEL

  gpu_average<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms, number_of_beads, position_beads.data(), velocity_beads.data(),
    potential_beads.data(), force_beads.data(), virial_beads.data(), atom.position_per_atom.data(),
    atom.velocity_per_atom.data(), atom.potential_per_atom.data(), atom.force_per_atom.data(),
    atom.virial_per_atom.data());
  CUDA_CHECK_KERNEL

  gpu_find_kinetic_energy_virial_part<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box, number_of_atoms, number_of_beads, position_beads.data(), force_beads.data(),
    atom.position_per_atom.data(), kinetic_energy_virial_part.data());
  CUDA_CHECK_KERNEL

  gpu_find_sum_1024<<<1024, 128>>>(
    number_of_atoms, kinetic_energy_virial_part.data(), atom.position_per_atom.data(),
    atom.virial_per_atom.data(), sum_1024.data());
  CUDA_CHECK_KERNEL

  gpu_find_thermo<<<8, 1024>>>(
    box.get_volume(), number_of_atoms * K_B * temperature, sum_1024.data(), thermo.data());
  CUDA_CHECK_KERNEL
}
