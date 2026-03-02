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
Add spring forces for a group of atoms.

Implemented by: Hekai Bu (Wuhan University), hekai_bu@whu.edu.cn
------------------------------------------------------------------------------*/

#include "add_spring.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <limits>

static void __global__ gpu_sum_group_mass_pos_reduce(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* __restrict__ g_mass,
  double* __restrict__ d_sum_mr,
  double* __restrict__ d_sum_m)
{
  extern __shared__ double s[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lid = threadIdx.x;

  // Accumulate locally
  double local_mx = 0.0;
  double local_my = 0.0;
  double local_mz = 0.0;
  double local_m = 0.0;
  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    const double mass = g_mass[atom_id];
    local_mx = mass * g_x[atom_id];
    local_my = mass * g_y[atom_id];
    local_mz = mass * g_z[atom_id];
    local_m = mass;
  }

  // Store to shared memory
  s[lid * 4 + 0] = local_mx;
  s[lid * 4 + 1] = local_my;
  s[lid * 4 + 2] = local_mz;
  s[lid * 4 + 3] = local_m;
  __syncthreads();

  // reduction
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      s[lid * 4 + 0] += s[(lid + stride) * 4 + 0];
      s[lid * 4 + 1] += s[(lid + stride) * 4 + 1];
      s[lid * 4 + 2] += s[(lid + stride) * 4 + 2];
      s[lid * 4 + 3] += s[(lid + stride) * 4 + 3];
    }
    __syncthreads();
  }

  // Write results to global memory
  if (lid == 0) {
    atomicAdd(&d_sum_mr[0], s[0]);
    atomicAdd(&d_sum_mr[1], s[1]);
    atomicAdd(&d_sum_mr[2], s[2]);
    atomicAdd(&d_sum_m[0], s[3]);
  }
}

static void __global__ gpu_init_ghost_atom_pos(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double offset_x,
  const double offset_y,
  const double offset_z,
  double* __restrict__ ghost_pos)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    ghost_pos[tid * 3 + 0] = g_x[atom_id] + offset_x;
    ghost_pos[tid * 3 + 1] = g_y[atom_id] + offset_y;
    ghost_pos[tid * 3 + 2] = g_z[atom_id] + offset_z;
  }
}

static void __global__ gpu_update_ghost_atom_pos(
  const int group_size,
  double* __restrict__ ghost_pos,
  const double vx,
  const double vy,
  const double vz)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < group_size) {
    ghost_pos[tid * 3 + 0] += vx;
    ghost_pos[tid * 3 + 1] += vy;
    ghost_pos[tid * 3 + 2] += vz;
  }
}

static void __global__ gpu_add_force_to_group_mass_weighted(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_mass,
  const double sum_mass_inv,
  const double fx,
  const double fy,
  const double fz,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    const double mass_frac = g_mass[atom_id] * sum_mass_inv;
    g_fx[atom_id] += fx * mass_frac;
    g_fy[atom_id] += fy * mass_frac;
    g_fz[atom_id] += fz * mass_frac;
  }
}

// Ghost atom: couple mode (radial spring with R0)
static void __global__ gpu_add_spring_ghost_atom_couple(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* __restrict__ ghost_pos,
  const double k,
  const double R0,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  double* __restrict__ d_sum_force,
  double* __restrict__ d_sum_energy)
{
  extern __shared__ double s[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lid = threadIdx.x;

  double local_fx = 0.0;
  double local_fy = 0.0;
  double local_fz = 0.0;
  double local_e = 0.0;

  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    const double dx = ghost_pos[tid * 3 + 0] - g_x[atom_id];
    const double dy = ghost_pos[tid * 3 + 1] - g_y[atom_id];
    const double dz = ghost_pos[tid * 3 + 2] - g_z[atom_id];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double r = sqrt(r2);
    const double dr = r - R0;
    if (r2 > 1.0e-20) {
      const double f = k * dr / r;
      local_fx = f * dx;
      local_fy = f * dy;
      local_fz = f * dz;
    }
    local_e = 0.5 * k * dr * dr;
    g_fx[atom_id] += local_fx;
    g_fy[atom_id] += local_fy;
    g_fz[atom_id] += local_fz;
  }

  // Store to shared memory
  s[lid * 4 + 0] = local_fx;
  s[lid * 4 + 1] = local_fy;
  s[lid * 4 + 2] = local_fz;
  s[lid * 4 + 3] = local_e;
  __syncthreads();

  // Reduction
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      s[lid * 4 + 0] += s[(lid + stride) * 4 + 0];
      s[lid * 4 + 1] += s[(lid + stride) * 4 + 1];
      s[lid * 4 + 2] += s[(lid + stride) * 4 + 2];
      s[lid * 4 + 3] += s[(lid + stride) * 4 + 3];
    }
    __syncthreads();
  }

  // Write results to global memory
  if (lid == 0) {
    atomicAdd(&d_sum_force[0], s[0]);
    atomicAdd(&d_sum_force[1], s[1]);
    atomicAdd(&d_sum_force[2], s[2]);
    atomicAdd(&d_sum_energy[0], s[3]);
  }
}

// Ghost atom: decouple mode (Cartesian springs)
static void __global__ gpu_add_spring_ghost_atom_decouple(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* __restrict__ ghost_pos,
  const double kx,
  const double ky,
  const double kz,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  double* __restrict__ d_sum_force,
  double* __restrict__ d_sum_energy)
{
  extern __shared__ double s[];
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lid = threadIdx.x;

  double local_fx = 0.0;
  double local_fy = 0.0;
  double local_fz = 0.0;
  double local_e = 0.0;

  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    const double dx = ghost_pos[tid * 3 + 0] - g_x[atom_id];
    const double dy = ghost_pos[tid * 3 + 1] - g_y[atom_id];
    const double dz = ghost_pos[tid * 3 + 2] - g_z[atom_id];
    local_fx = kx * dx;
    local_fy = ky * dy;
    local_fz = kz * dz;
    local_e = 0.5 * (kx * dx * dx + ky * dy * dy + kz * dz * dz);
    g_fx[atom_id] += local_fx;
    g_fy[atom_id] += local_fy;
    g_fz[atom_id] += local_fz;
  }

  // Store to shared memory
  s[lid * 4 + 0] = local_fx;
  s[lid * 4 + 1] = local_fy;
  s[lid * 4 + 2] = local_fz;
  s[lid * 4 + 3] = local_e;
  __syncthreads();

  // Reduction
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      s[lid * 4 + 0] += s[(lid + stride) * 4 + 0];
      s[lid * 4 + 1] += s[(lid + stride) * 4 + 1];
      s[lid * 4 + 2] += s[(lid + stride) * 4 + 2];
      s[lid * 4 + 3] += s[(lid + stride) * 4 + 3];
    }
    __syncthreads();
  }

  // Write results to global memory
  if (lid == 0) {
    atomicAdd(&d_sum_force[0], s[0]);
    atomicAdd(&d_sum_force[1], s[1]);
    atomicAdd(&d_sum_force[2], s[2]);
    atomicAdd(&d_sum_energy[0], s[3]);
  }
}

// Couple COM: apply equal-opposite forces to two groups
static void __global__ gpu_add_spring_couple_com_force(
  const int* __restrict__ g_group_contents,
  const int group1_size,
  const int group1_size_sum,
  const int group2_size,
  const int group2_size_sum,
  const double* __restrict__ g_mass,
  const double fx,
  const double fy,
  const double fz,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  const double sum_mass1_inv,
  const double sum_mass2_inv)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Apply force to group 1
  if (tid < group1_size) {
    const int atom_id = g_group_contents[group1_size_sum + tid];
    const double mass_frac = g_mass[atom_id] * sum_mass1_inv;
    g_fx[atom_id] += fx * mass_frac;
    g_fy[atom_id] += fy * mass_frac;
    g_fz[atom_id] += fz * mass_frac;
  }

  // Apply opposite force to group 2
  if (tid < group2_size) {
    const int atom_id = g_group_contents[group2_size_sum + tid];
    const double mass_frac = g_mass[atom_id] * sum_mass2_inv;
    g_fx[atom_id] -= fx * mass_frac;
    g_fy[atom_id] -= fy * mass_frac;
    g_fz[atom_id] -= fz * mass_frac;
  }
}

void Add_Spring::parse(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom)
{
  printf("Add spring [%d call(s)].\n", num_calls_);
  if (atom.unwrapped_position.size() < atom.position_per_atom.size()) {
    atom.unwrapped_position.resize(atom.position_per_atom.size());
    atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
  }
  if (atom.position_temp.size() < atom.position_per_atom.size()) {
    atom.position_temp.resize(atom.position_per_atom.size());
  }

  if (num_calls_ >= MAX_SPRING_CALLS) {
    std::string error_msg = "add_spring cannot be used more than " + std::to_string(MAX_SPRING_CALLS) + " times.\n";
    PRINT_INPUT_ERROR(error_msg.c_str());
  }

  const char* mode_str = param[1];
  const int id = num_calls_;

  if (strcmp(mode_str, "ghost_com") == 0) {
    // Syntax:
    //   add_spring ghost_com gm gid vx vy vz couple   k  R0  x0 y0 z0
    //   add_spring ghost_com gm gid vx vy vz decouple kx ky kz x0 y0 z0

    mode_[id] = MODE_GHOST_COM;

    // Parse group info
    if (!is_valid_int(param[2], &grouping_method_[id])) {
      PRINT_INPUT_ERROR("grouping method should be an integer.\n");
    }
    if (grouping_method_[id] < 0 || grouping_method_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("grouping method is out of range.\n");
    }

    if (!is_valid_int(param[3], &group_id_[id])) {
      PRINT_INPUT_ERROR("group id should be an integer.\n");
    }
    if (group_id_[id] < 0 || group_id_[id] >= groups[grouping_method_[id]].number) {
      PRINT_INPUT_ERROR("group id is out of range.\n");
    }
    if (groups[grouping_method_[id]].cpu_size[group_id_[id]] <= 0) {
      PRINT_INPUT_ERROR("The group for add_spring is empty.\n");
    }

    // Parse velocity
    if (!is_valid_real(param[4], &velocity_[id][0]) ||
        !is_valid_real(param[5], &velocity_[id][1]) ||
        !is_valid_real(param[6], &velocity_[id][2])) {
      PRINT_INPUT_ERROR("velocity should be three numbers.\n");
    }

    const char* stiff_str = param[7];

    if (strcmp(stiff_str, "couple") == 0) {
      if (num_param != 13) {
        PRINT_INPUT_ERROR("add_spring ghost_com couple requires 13 parameters.\n");
      }
      stiffness_mode_[id] = STIFFNESS_COUPLE;

      if (!is_valid_real(param[8], &k_couple_[id]) || k_couple_[id] <= 0.0) {
        PRINT_INPUT_ERROR("spring constant k should be positive.\n");
      }
      if (!is_valid_real(param[9], &R0_[id]) || R0_[id] < -1.0e-20) {
        PRINT_INPUT_ERROR("R0 should be a non-negative number.\n");
      }
      if (!is_valid_real(param[10], &offset_[id][0]) ||
          !is_valid_real(param[11], &offset_[id][1]) ||
          !is_valid_real(param[12], &offset_[id][2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }
      
      if (offset_[id][0] == 0.0 && offset_[id][1] == 0.0 && offset_[id][2] == 0.0
          && R0_[id] > 1.0e-20) {
        printf("    Warning: zero offset with positive R0 may lead to weird forces at the beginning of the simulation. So the forces are set to zero initially.\n");
      }

      printf("    ghost_com couple: grouping_method=%d, group_id=%d\n", grouping_method_[id], group_id_[id]);
      printf("    velocity=(%g,%g,%g) Å/step, k=%g eV/Å^2, R0=%g Å\n",
             velocity_[id][0], velocity_[id][1], velocity_[id][2], k_couple_[id], R0_[id]);
      printf("    offset=(%g,%g,%g) Å\n", offset_[id][0], offset_[id][1], offset_[id][2]);

    } else if (strcmp(stiff_str, "decouple") == 0) {
      if (num_param != 14) {
        PRINT_INPUT_ERROR("add_spring ghost_com decouple requires 14 parameters.\n");
      }
      stiffness_mode_[id] = STIFFNESS_DECOUPLE;

      if (!is_valid_real(param[8], &k_decouple_[id][0]) || k_decouple_[id][0] < -1.0e-20 ||
          !is_valid_real(param[9], &k_decouple_[id][1]) || k_decouple_[id][1] < -1.0e-20 ||
          !is_valid_real(param[10], &k_decouple_[id][2]) || k_decouple_[id][2] < -1.0e-20) {
        PRINT_INPUT_ERROR("k components should be non-negative numbers.\n");
      }

      if (!is_valid_real(param[11], &offset_[id][0]) ||
          !is_valid_real(param[12], &offset_[id][1]) ||
          !is_valid_real(param[13], &offset_[id][2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }

      printf("    ghost_com decouple: grouping_method=%d, group_id=%d\n", grouping_method_[id], group_id_[id]);
      printf("    velocity=(%g,%g,%g) Å/step, k=(%g,%g,%g) eV/Å^2\n",
             velocity_[id][0], velocity_[id][1], velocity_[id][2],
             k_decouple_[id][0], k_decouple_[id][1], k_decouple_[id][2]);
      printf("    offset=(%g,%g,%g) Å\n", offset_[id][0], offset_[id][1], offset_[id][2]);

    } else {
      PRINT_INPUT_ERROR("stiffness mode should be 'couple' or 'decouple'.\n");
    }

    init_origin_[id] = 0;

  } else if (strcmp(mode_str, "ghost_atom") == 0) {
    // Syntax:
    //   add_spring ghost_atom gm gid vx vy vz couple   k  R0  x0 y0 z0
    //   add_spring ghost_atom gm gid vx vy vz decouple kx ky kz x0 y0 z0

    mode_[id] = MODE_GHOST_ATOM;

    // Parse group info
    if (!is_valid_int(param[2], &grouping_method_[id])) {
      PRINT_INPUT_ERROR("grouping method should be an integer.\n");
    }
    if (grouping_method_[id] < 0 || grouping_method_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("grouping method is out of range.\n");
    }

    if (!is_valid_int(param[3], &group_id_[id])) {
      PRINT_INPUT_ERROR("group id should be an integer.\n");
    }
    if (group_id_[id] < 0 || group_id_[id] >= groups[grouping_method_[id]].number) {
      PRINT_INPUT_ERROR("group id is out of range.\n");
    }

    // Parse velocity
    if (!is_valid_real(param[4], &velocity_[id][0]) ||
        !is_valid_real(param[5], &velocity_[id][1]) ||
        !is_valid_real(param[6], &velocity_[id][2])) {
      PRINT_INPUT_ERROR("velocity should be three numbers.\n");
    }

    const char* stiff_str = param[7];

    if (strcmp(stiff_str, "couple") == 0) {
      if (num_param != 13) {
        PRINT_INPUT_ERROR("add_spring ghost_atom couple requires 13 parameters.\n");
      }
      stiffness_mode_[id] = STIFFNESS_COUPLE;

      if (!is_valid_real(param[8], &k_couple_[id]) || k_couple_[id] <= 0.0) {
        PRINT_INPUT_ERROR("spring constant k should be positive.\n");
      }
      // calculate the k for each ghost atom
      double k_total = k_couple_[id];
      k_couple_[id] /= groups[grouping_method_[id]].cpu_size[group_id_[id]];
      printf("    total k=%g eV/Å^2, k per atom=%g eV/Å^2\n", k_total, k_couple_[id]);


      if (!is_valid_real(param[9], &R0_[id]) || R0_[id] < -1.0e-20) {
        PRINT_INPUT_ERROR("R0 should be a non-negative number.\n");
      }
      if (!is_valid_real(param[10], &offset_[id][0]) ||
          !is_valid_real(param[11], &offset_[id][1]) ||
          !is_valid_real(param[12], &offset_[id][2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }

      if (offset_[id][0] == 0.0 && offset_[id][1] == 0.0 && offset_[id][2] == 0.0
          && R0_[id] > 1.0e-20) {
        printf("    Warning: zero offset with positive R0 may lead to weird forces at the beginning of the simulation. So the forces are set to zero initially.\n");
      }

      printf("    ghost_atom couple: grouping_method=%d, group_id=%d\n", grouping_method_[id], group_id_[id]);
      printf("    velocity=(%g,%g,%g) Å/step, k=%g eV/Å^2, R0=%g Å\n",
             velocity_[id][0], velocity_[id][1], velocity_[id][2], k_couple_[id], R0_[id]);
      printf("    offset=(%g,%g,%g) Å\n", offset_[id][0], offset_[id][1], offset_[id][2]);

    } else if (strcmp(stiff_str, "decouple") == 0) {
      if (num_param != 14) {
        PRINT_INPUT_ERROR("add_spring ghost_atom decouple requires 14 parameters.\n");
      }
      stiffness_mode_[id] = STIFFNESS_DECOUPLE;

      if (!is_valid_real(param[8], &k_decouple_[id][0]) ||
          !is_valid_real(param[9], &k_decouple_[id][1]) ||
          !is_valid_real(param[10], &k_decouple_[id][2]) ||
          k_decouple_[id][1] < -1.0e-20 || k_decouple_[id][2] < -1.0e-20 ||
          k_decouple_[id][0] < -1.0e-20) {
        PRINT_INPUT_ERROR("k components should be non-negative numbers.\n");
      }
      // calculate the k for each ghost atom
      double k_total[3] = {k_decouple_[id][0], k_decouple_[id][1], k_decouple_[id][2]};
      k_decouple_[id][0] /= groups[grouping_method_[id]].cpu_size[group_id_[id]];
      k_decouple_[id][1] /= groups[grouping_method_[id]].cpu_size[group_id_[id]];
      k_decouple_[id][2] /= groups[grouping_method_[id]].cpu_size[group_id_[id]];
      printf("    total k=(%g,%g,%g) eV/Å^2, k per atom=(%g,%g,%g) eV/Å^2\n",
             k_total[0], k_total[1], k_total[2],
             k_decouple_[id][0], k_decouple_[id][1], k_decouple_[id][2]);

      if (!is_valid_real(param[11], &offset_[id][0]) ||
          !is_valid_real(param[12], &offset_[id][1]) ||
          !is_valid_real(param[13], &offset_[id][2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }

      printf("    ghost_atom decouple: grouping_method=%d, group_id=%d\n", 
              grouping_method_[id], group_id_[id]);
      printf("    velocity=(%g,%g,%g) Å/step, k=(%g,%g,%g) eV/Å^2\n",
             velocity_[id][0], velocity_[id][1], velocity_[id][2],
             k_decouple_[id][0], k_decouple_[id][1], k_decouple_[id][2]);
      printf("    offset=(%g,%g,%g) Å\n", offset_[id][0], offset_[id][1], offset_[id][2]);

    } else {
      PRINT_INPUT_ERROR("stiffness mode should be 'couple' or 'decouple'.\n");
    }

    // Allocate ghost atom positions (will be initialized at first compute)
    ghost_atom_group_size_[id] = groups[grouping_method_[id]].cpu_size[group_id_[id]];
    if (ghost_atom_group_size_[id] <= 0) {
      PRINT_INPUT_ERROR("The group for add_spring is empty.\n");
    }
    ghost_atom_pos_[id].resize(3 * ghost_atom_group_size_[id]);
    init_origin_[id] = 0;

  } else if (strcmp(mode_str, "couple_com") == 0) {
    // Syntax:
    //   add_spring couple_com gm gid1 gid2 couple   k  R0
    //   add_spring couple_com gm gid1 gid2 decouple kx ky kz

    mode_[id] = MODE_COUPLE_COM;

    // Parse grouping method
    if (!is_valid_int(param[2], &grouping_method_[id])) {
      PRINT_INPUT_ERROR("grouping_method should be an integer.\n");
    }
    if (grouping_method_[id] < 0 || grouping_method_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("grouping_method is out of range.\n");
    }

    // Parse first group id
    if (!is_valid_int(param[3], &group_id_[id])) {
      PRINT_INPUT_ERROR("group_id_1 should be an integer.\n");
    }
    if (group_id_[id] < 0 || group_id_[id] >= groups[grouping_method_[id]].number) {
      PRINT_INPUT_ERROR("group_id_1 is out of range.\n");
    }

    // Parse second group id
    if (!is_valid_int(param[4], &group_id_2_[id])) {
      PRINT_INPUT_ERROR("group_id_2 should be an integer.\n");
    }
    if (group_id_2_[id] < 0 || group_id_2_[id] >= groups[grouping_method_[id]].number) {
      PRINT_INPUT_ERROR("group_id_2 is out of range.\n");
    }

    if (groups[grouping_method_[id]].cpu_size[group_id_[id]] <= 0) {
      PRINT_INPUT_ERROR("The first group for add_spring is empty.\n");
    }
    if (groups[grouping_method_[id]].cpu_size[group_id_2_[id]] <= 0) {
      PRINT_INPUT_ERROR("The second group for add_spring is empty.\n");
    }

    if (group_id_[id] == group_id_2_[id]) {
      PRINT_INPUT_ERROR("group_id_1 and group_id_2 cannot be the same.\n");
    }

    const char* stiff_str = param[5];

    if (strcmp(stiff_str, "couple") == 0) {
      if (num_param != 8) {
        PRINT_INPUT_ERROR("add_spring couple_com couple requires 8 parameters.\n");
      }
      stiffness_mode_[id] = STIFFNESS_COUPLE;

      if (!is_valid_real(param[6], &k_couple_[id]) || k_couple_[id] <= 0.0) {
        PRINT_INPUT_ERROR("spring constant k should be positive.\n");
      }
      if (!is_valid_real(param[7], &R0_[id]) || R0_[id] < -1.0e-20) {
        PRINT_INPUT_ERROR("R0 should be a non-negative number.\n");
      }

      printf("    couple_com couple: gm=%d, gid1=%d, gid2=%d\n",
             grouping_method_[id], group_id_[id], group_id_2_[id]);
      printf("    k=%g eV/Å^2, R0=%g Å\n", k_couple_[id], R0_[id]);

    } else if (strcmp(stiff_str, "decouple") == 0) {
      if (num_param != 9) {
        PRINT_INPUT_ERROR("add_spring couple_com decouple requires 9 parameters.\n");
      }
      stiffness_mode_[id] = STIFFNESS_DECOUPLE;

      if (!is_valid_real(param[6], &k_decouple_[id][0]) ||
          !is_valid_real(param[7], &k_decouple_[id][1]) ||
          !is_valid_real(param[8], &k_decouple_[id][2]) ||
          k_decouple_[id][0] < -1.0e-20 || k_decouple_[id][1] < -1.0e-20 ||
          k_decouple_[id][2] < -1.0e-20) {
        PRINT_INPUT_ERROR("k components should be non-negative numbers.\n");
      }

      printf("    couple_com decouple: gm=%d, gid1=%d, gid2=%d\n",
             grouping_method_[id], group_id_[id], group_id_2_[id]);
      printf("    k=(%g,%g,%g) eV/Å^2\n", k_decouple_[id][0], k_decouple_[id][1], k_decouple_[id][2]);

    } else {
      PRINT_INPUT_ERROR("stiffness mode should be 'couple' or 'decouple'.\n");
    }

    // No velocity for couple_com
    velocity_[id][0] = 0.0;
    velocity_[id][1] = 0.0;
    velocity_[id][2] = 0.0;

  } else {
    PRINT_INPUT_ERROR("Unknown mode. Use ghost_com, ghost_atom, or couple_com.\n");
  }

  // Initialize output file
  energy_[id] = 0.0;
  force_[id][0] = 0.0;
  force_[id][1] = 0.0;
  force_[id][2] = 0.0;
  total_force_[id] = 0.0;

  if (output_stride_ > 0) {
    std::string filename = "spring_force_" + std::to_string(id) + ".out";
    fp_out_[id] = fopen(filename.c_str(), "w");
    if (fp_out_[id]) {
      fprintf(fp_out_[id], "# step  mode  Fx  Fy  Fz Ftotal (eV/Å) energy (eV)\n");
      fflush(fp_out_[id]);
    }
  }

  ++num_calls_;
}

void Add_Spring::compute(const int step, const std::vector<Group>& groups, Atom& atom)
{
  for (int c = 0; c < num_calls_; ++c) {
    // Allocate temp buffers on first call
    if (d_tmp_vec3_.size() == 0) {
      d_tmp_vec3_.resize(3);
      d_tmp_scalar_.resize(1);
      d_tmp_force3_.resize(3);
    }

    const int block_size = 64;
    if (atom.unwrapped_position.size() < atom.position_per_atom.size()) {
      printf("Warning: unwrapped_position size is less than position_per_atom size.\n");
    }
    const int num_atoms_total = atom.type.size();
    double *g_x = atom.unwrapped_position.data();
    double *g_y = atom.unwrapped_position.data() + num_atoms_total;
    double *g_z = atom.unwrapped_position.data() + num_atoms_total * 2;

    const int group_size = groups[grouping_method_[c]].cpu_size[group_id_[c]];
    const int group_size_sum = groups[grouping_method_[c]].cpu_size_sum[group_id_[c]];
    const int grid_size = (group_size - 1) / block_size + 1;

    if (mode_[c] == MODE_GHOST_COM) {
      // Zero out buffers before atomic reduction
      d_tmp_vec3_.fill(0.0);
      d_tmp_scalar_.fill(0.0);

      // Compute COM position of the group
      gpu_sum_group_mass_pos_reduce<<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
        group_size,
        group_size_sum,
        groups[grouping_method_[c]].contents.data(),
        g_x,
        g_y,
        g_z,
        atom.mass.data(),
        d_tmp_vec3_.data(),
        d_tmp_scalar_.data());
      GPU_CHECK_KERNEL

      // Copy results to host
      double h_sum_mx[3], h_sum_m;
      d_tmp_vec3_.copy_to_host(h_sum_mx);
      d_tmp_scalar_.copy_to_host(&h_sum_m);
      const double h_sum_m_inv = 1.0 / h_sum_m;

      double com_x = h_sum_mx[0] * h_sum_m_inv;
      double com_y = h_sum_mx[1] * h_sum_m_inv;
      double com_z = h_sum_mx[2] * h_sum_m_inv;

      // Initialize origin at first step
      if (init_origin_[c] == 0) {
        origin_[c][0] = com_x + offset_[c][0];
        origin_[c][1] = com_y + offset_[c][1];
        origin_[c][2] = com_z + offset_[c][2];
        init_origin_[c] = 1;
      }

      // Update ghost position
      double ghost_x = origin_[c][0];
      double ghost_y = origin_[c][1];
      double ghost_z = origin_[c][2];
      origin_[c][0] += velocity_[c][0];
      origin_[c][1] += velocity_[c][1];
      origin_[c][2] += velocity_[c][2];

      // Compute spring force
      double dx = ghost_x - com_x;
      double dy = ghost_y - com_y;
      double dz = ghost_z - com_z;

      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      double e = 0.0;

      if (stiffness_mode_[c] == STIFFNESS_COUPLE) {
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double dr = r - R0_[c];
        if (r > 1.0e-20) {
          double f = k_couple_[c] * dr / r;
          fx = f * dx;
          fy = f * dy;
          fz = f * dz;
        }
        e = 0.5 * k_couple_[c] * dr * dr;
      } else {
        fx = k_decouple_[c][0] * dx;
        fy = k_decouple_[c][1] * dy;
        fz = k_decouple_[c][2] * dz;
        e = 0.5 * (k_decouple_[c][0] * dx * dx + k_decouple_[c][1] * dy * dy + k_decouple_[c][2] * dz * dz);
      }

      // Apply force to group atoms
      gpu_add_force_to_group_mass_weighted<<<grid_size, block_size>>>(
        group_size,
        group_size_sum,
        groups[grouping_method_[c]].contents.data(),
        atom.mass.data(),
        h_sum_m_inv,
        fx,
        fy,
        fz,
        atom.force_per_atom.data(),
        atom.force_per_atom.data() + num_atoms_total,
        atom.force_per_atom.data() + num_atoms_total * 2);
      GPU_CHECK_KERNEL

      force_[c][0] = fx;
      force_[c][1] = fy;
      force_[c][2] = fz;
      energy_[c] = e;
      total_force_[c] = sqrt(fx * fx + fy * fy + fz * fz);

    } else if (mode_[c] == MODE_GHOST_ATOM) {
      if (init_origin_[c] == 0) {
        // Initialize ghost atom positions
        gpu_init_ghost_atom_pos<<<grid_size, block_size>>>(
          group_size,
          group_size_sum,
          groups[grouping_method_[c]].contents.data(),
          g_x,
          g_y,
          g_z,
          offset_[c][0],
          offset_[c][1],
          offset_[c][2],
          ghost_atom_pos_[c].data());
        GPU_CHECK_KERNEL
        init_origin_[c] = 1;
      } else {
        // Update ghost atom positions by adding velocity * 1 step
        gpu_update_ghost_atom_pos<<<grid_size, block_size>>>(
          group_size,
          ghost_atom_pos_[c].data(),
          velocity_[c][0],
          velocity_[c][1],
          velocity_[c][2]);
        GPU_CHECK_KERNEL
      }

      // Apply spring forces
      d_tmp_force3_.fill(0.0);
      d_tmp_scalar_.fill(0.0);
      if (stiffness_mode_[c] == STIFFNESS_COUPLE) {
        gpu_add_spring_ghost_atom_couple
        <<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
          group_size,
          group_size_sum,
          groups[grouping_method_[c]].contents.data(),
          g_x,
          g_y,
          g_z,
          ghost_atom_pos_[c].data(),
          k_couple_[c],
          R0_[c],
          atom.force_per_atom.data(),
          atom.force_per_atom.data() + num_atoms_total,
          atom.force_per_atom.data() + num_atoms_total * 2,
          d_tmp_force3_.data(),
          d_tmp_scalar_.data());
      } else {
        gpu_add_spring_ghost_atom_decouple
        <<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
          group_size,
          group_size_sum,
          groups[grouping_method_[c]].contents.data(),
          g_x,
          g_y,
          g_z,
          ghost_atom_pos_[c].data(),
          k_decouple_[c][0],
          k_decouple_[c][1],
          k_decouple_[c][2],
          atom.force_per_atom.data(),
          atom.force_per_atom.data() + num_atoms_total,
          atom.force_per_atom.data() + num_atoms_total * 2,
          d_tmp_force3_.data(),
          d_tmp_scalar_.data());
      }
      GPU_CHECK_KERNEL

      // Copy force results to host
      double h_force[3];
      d_tmp_force3_.copy_to_host(h_force);

      double h_energy = 0.0;
      d_tmp_scalar_.copy_to_host(&h_energy);

      force_[c][0] = h_force[0];
      force_[c][1] = h_force[1];
      force_[c][2] = h_force[2];
      total_force_[c] = sqrt(h_force[0] * h_force[0] + h_force[1] * h_force[1] + h_force[2] * h_force[2]);

      // Total energy from all springs
      energy_[c] = h_energy;

    } else if (mode_[c] == MODE_COUPLE_COM) {
      // Zero out buffers before atomic reduction
      d_tmp_vec3_.fill(0.0);
      d_tmp_scalar_.fill(0.0);

      // Compute COM for group 1
      gpu_sum_group_mass_pos_reduce
      <<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
        group_size,
        groups[grouping_method_[c]].cpu_size_sum[group_id_[c]],
        groups[grouping_method_[c]].contents.data(),
        g_x,
        g_y,
        g_z,
        atom.mass.data(),
        d_tmp_vec3_.data(),
        d_tmp_scalar_.data());
      GPU_CHECK_KERNEL

      double h_sum_mx1[3], h_sum_m1;
      d_tmp_vec3_.copy_to_host(h_sum_mx1);
      d_tmp_scalar_.copy_to_host(&h_sum_m1);

      // Compute COM for group 2
      int group2_size = groups[grouping_method_[c]].cpu_size[group_id_2_[c]];
      int group2_size_sum = groups[grouping_method_[c]].cpu_size_sum[group_id_2_[c]];
      int grid_size_2 = (group2_size - 1) / block_size + 1;

      // Zero out buffers before atomic reduction
      d_tmp_vec3_.fill(0.0);
      d_tmp_scalar_.fill(0.0);

      gpu_sum_group_mass_pos_reduce
      <<<grid_size_2, block_size, block_size * 4 * sizeof(double)>>>(
        group2_size,
        group2_size_sum,
        groups[grouping_method_[c]].contents.data(),
        g_x,
        g_y,
        g_z,
        atom.mass.data(),
        d_tmp_vec3_.data(),
        d_tmp_scalar_.data());
      GPU_CHECK_KERNEL

      double h_sum_mx2[3], h_sum_m2;
      d_tmp_vec3_.copy_to_host(h_sum_mx2);
      d_tmp_scalar_.copy_to_host(&h_sum_m2);

      double h_sum_m1_inv = 1.0 / h_sum_m1;
      double h_sum_m2_inv = 1.0 / h_sum_m2;

      double com1_x = h_sum_mx1[0] * h_sum_m1_inv;
      double com1_y = h_sum_mx1[1] * h_sum_m1_inv;
      double com1_z = h_sum_mx1[2] * h_sum_m1_inv;
      double com2_x = h_sum_mx2[0] * h_sum_m2_inv;
      double com2_y = h_sum_mx2[1] * h_sum_m2_inv;
      double com2_z = h_sum_mx2[2] * h_sum_m2_inv;

      // Compute spring force between COMs
      double dx = com2_x - com1_x;
      double dy = com2_y - com1_y;
      double dz = com2_z - com1_z;

      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      double e = 0.0;

      if (stiffness_mode_[c] == STIFFNESS_COUPLE) {
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double dr = r - R0_[c];
        if (r > 1.0e-20) {
          double f = k_couple_[c] * dr / r;
          fx = f * dx;
          fy = f * dy;
          fz = f * dz;
        }
        e = 0.5 * k_couple_[c] * dr * dr;
      } else {
        fx = k_decouple_[c][0] * dx;
        fy = k_decouple_[c][1] * dy;
        fz = k_decouple_[c][2] * dz;
        e = 0.5 * (k_decouple_[c][0] * dx * dx + k_decouple_[c][1] * dy * dy + k_decouple_[c][2] * dz * dz);
      }

      // Apply equal-opposite forces
      int max_size = std::max(group_size, group2_size);
      int grid_size_max = (max_size - 1) / block_size + 1;
      gpu_add_spring_couple_com_force<<<grid_size_max, block_size>>>(
        groups[grouping_method_[c]].contents.data(),
        group_size,
        groups[grouping_method_[c]].cpu_size_sum[group_id_[c]],
        group2_size,
        group2_size_sum,
        atom.mass.data(),
        fx,
        fy,
        fz,
        atom.force_per_atom.data(),
        atom.force_per_atom.data() + num_atoms_total,
        atom.force_per_atom.data() + num_atoms_total * 2,
        h_sum_m1_inv,
        h_sum_m2_inv);
      GPU_CHECK_KERNEL

      force_[c][0] = fx;
      force_[c][1] = fy;
      force_[c][2] = fz;
      energy_[c] = e;
      total_force_[c] = sqrt(fx * fx + fy * fy + fz * fz);
    }

    // Write output if needed
    if (fp_out_[c] && output_stride_ > 0 && step % output_stride_ == 0) {
      fprintf(fp_out_[c], "%d  %d  %g  %g  %g  %g  %g\n",
              step, mode_[c], force_[c][0], force_[c][1], force_[c][2], total_force_[c], energy_[c]);
      fflush(fp_out_[c]);
    }
  }
}

void Add_Spring::finalize()
{
  // GPU_Vector destructors will automatically free device memory
  d_tmp_vec3_.resize(0);
  d_tmp_scalar_.resize(0);
  d_tmp_force3_.resize(0);

  for (int c = 0; c < MAX_SPRING_CALLS; ++c) {
    ghost_atom_pos_[c].resize(0);
    init_origin_[c] = 0;
    if (fp_out_[c]) {
      fclose(fp_out_[c]);
      fp_out_[c] = nullptr;
    }
  }

  num_calls_ = 0;
}
