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
[1] Ceriotti et al., J. Chem. Phys. 133, 124104 (2010).
[2] Mariana Rossi et al., J. Chem. Phys. 140, 234116 (2014).
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
      double cos_factor = cos(omega_k * time_step);
      double sin_factor = sin(omega_k * time_step);
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
        int index_dn = d * number_of_atoms + n;
        velocity[k][index_dn] = c1 * velocity[k][index_dn] + c2 * CURAND_NORMAL(&state);
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

  // get averaged quantities
}
