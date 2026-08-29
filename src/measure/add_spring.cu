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
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

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
  s[lid * 4 + 0] = local_mx;
  s[lid * 4 + 1] = local_my;
  s[lid * 4 + 2] = local_mz;
  s[lid * 4 + 3] = local_m;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      s[lid * 4 + 0] += s[(lid + stride) * 4 + 0];
      s[lid * 4 + 1] += s[(lid + stride) * 4 + 1];
      s[lid * 4 + 2] += s[(lid + stride) * 4 + 2];
      s[lid * 4 + 3] += s[(lid + stride) * 4 + 3];
    }
    __syncthreads();
  }
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
  s[lid * 4 + 0] = local_fx;
  s[lid * 4 + 1] = local_fy;
  s[lid * 4 + 2] = local_fz;
  s[lid * 4 + 3] = local_e;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      s[lid * 4 + 0] += s[(lid + stride) * 4 + 0];
      s[lid * 4 + 1] += s[(lid + stride) * 4 + 1];
      s[lid * 4 + 2] += s[(lid + stride) * 4 + 2];
      s[lid * 4 + 3] += s[(lid + stride) * 4 + 3];
    }
    __syncthreads();
  }
  if (lid == 0) {
    atomicAdd(&d_sum_force[0], s[0]);
    atomicAdd(&d_sum_force[1], s[1]);
    atomicAdd(&d_sum_force[2], s[2]);
    atomicAdd(&d_sum_energy[0], s[3]);
  }
}

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
  s[lid * 4 + 0] = local_fx;
  s[lid * 4 + 1] = local_fy;
  s[lid * 4 + 2] = local_fz;
  s[lid * 4 + 3] = local_e;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lid < stride) {
      s[lid * 4 + 0] += s[(lid + stride) * 4 + 0];
      s[lid * 4 + 1] += s[(lid + stride) * 4 + 1];
      s[lid * 4 + 2] += s[(lid + stride) * 4 + 2];
      s[lid * 4 + 3] += s[(lid + stride) * 4 + 3];
    }
    __syncthreads();
  }
  if (lid == 0) {
    atomicAdd(&d_sum_force[0], s[0]);
    atomicAdd(&d_sum_force[1], s[1]);
    atomicAdd(&d_sum_force[2], s[2]);
    atomicAdd(&d_sum_energy[0], s[3]);
  }
}

static void __global__ gpu_add_spring_com_com_force(
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
  if (tid < group1_size) {
    const int atom_id = g_group_contents[group1_size_sum + tid];
    const double mass_frac = g_mass[atom_id] * sum_mass1_inv;
    g_fx[atom_id] += fx * mass_frac;
    g_fy[atom_id] += fy * mass_frac;
    g_fz[atom_id] += fz * mass_frac;
  }
  if (tid < group2_size) {
    const int atom_id = g_group_contents[group2_size_sum + tid];
    const double mass_frac = g_mass[atom_id] * sum_mass2_inv;
    g_fx[atom_id] -= fx * mass_frac;
    g_fy[atom_id] -= fy * mass_frac;
    g_fz[atom_id] -= fz * mass_frac;
  }
}

static std::map<std::pair<int, int>, std::set<int>>& active_spring_ids()
{
  static std::map<std::pair<int, int>, std::set<int>> ids;
  return ids;
}

static std::string spring_stem(const int grouping_method, const int group_id, const int spring_id)
{
  return "spring_gm" + std::to_string(grouping_method) + "_g" + std::to_string(group_id) +
         "_s" + std::to_string(spring_id);
}

static bool file_exists(const std::string& filename)
{
  std::ifstream input(filename.c_str());
  return input.good();
}

static int reserve_spring_id(
  const int grouping_method,
  const int group_id,
  const int requested_spring_id)
{
  std::set<int>& ids = active_spring_ids()[std::make_pair(grouping_method, group_id)];
  if (requested_spring_id >= 0) {
    if (ids.count(requested_spring_id) != 0) {
      PRINT_INPUT_ERROR("The same add_spring spring id cannot be used more than once for one group within a run.\n");
    }
    ids.insert(requested_spring_id);
    return requested_spring_id;
  }

  int spring_id = 0;
  while (
    ids.count(spring_id) != 0 ||
    file_exists(spring_stem(grouping_method, group_id, spring_id) + ".restart") ||
    file_exists(spring_stem(grouping_method, group_id, spring_id) + ".out")) {
    ++spring_id;
  }
  ids.insert(spring_id);
  return spring_id;
}

static void release_spring_id(
  const int grouping_method,
  const int group_id,
  const int spring_id)
{
  auto key = std::make_pair(grouping_method, group_id);
  auto map_it = active_spring_ids().find(key);
  if (map_it == active_spring_ids().end()) {
    return;
  }
  map_it->second.erase(spring_id);
  if (map_it->second.empty()) {
    active_spring_ids().erase(map_it);
  }
}

static int parse_continue_option(const char** param, const int num_param, const int base_num_param)
{
  if (num_param == base_num_param) {
    return -1;
  }
  if (num_param != base_num_param + 2) {
    PRINT_INPUT_ERROR(
      "add_spring has an invalid number of parameters. Optional continuation syntax is: continue <spring_id>.\n");
  }
  if (strcmp(param[base_num_param], "continue") != 0) {
    PRINT_INPUT_ERROR("The optional add_spring arguments should be: continue <spring_id>.\n");
  }
  int spring_id = -1;
  if (!is_valid_int(param[base_num_param + 1], &spring_id)) {
    PRINT_INPUT_ERROR("The spring id after 'continue' should be an integer.\n");
  }
  if (spring_id < 0) {
    PRINT_INPUT_ERROR("The spring id after 'continue' should be non-negative.\n");
  }
  return spring_id;
}

static bool read_labeled_int(std::ifstream& input, const char* expected_label, int& value)
{
  std::string label;
  if (!(input >> label >> value)) {
    return false;
  }
  return label == expected_label;
}

Add_Spring::Add_Spring(
  const char** param,
  const int num_param,
  const std::vector<Group>& groups,
  Atom& atom)
{
  action_name = "add_spring";
  printf("Add spring.\n");
  atom.enable_unwrapped_position();

  if (num_param < 2) {
    PRINT_INPUT_ERROR("add_spring requires a mode.\n");
  }
  const char* mode_str = param[1];

  if (strcmp(mode_str, "ghost_com") == 0) {
    mode_ = MODE_GHOST_COM;
    if (num_param < 8) {
      PRINT_INPUT_ERROR("add_spring ghost_com has too few parameters.\n");
    }
    if (!is_valid_int(param[2], &grouping_method_)) {
      PRINT_INPUT_ERROR("grouping method should be an integer.\n");
    }
    if (grouping_method_ < 0 || grouping_method_ >= (int)groups.size()) {
      PRINT_INPUT_ERROR("grouping method is out of range.\n");
    }
    if (!is_valid_int(param[3], &group_id_)) {
      PRINT_INPUT_ERROR("group id should be an integer.\n");
    }
    if (group_id_ < 0 || group_id_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group id is out of range.\n");
    }
    restart_group_size_ = groups[grouping_method_].cpu_size[group_id_];
    if (restart_group_size_ <= 0) {
      PRINT_INPUT_ERROR("The group for add_spring is empty.\n");
    }
    if (!is_valid_real(param[4], &velocity_[0]) ||
        !is_valid_real(param[5], &velocity_[1]) ||
        !is_valid_real(param[6], &velocity_[2])) {
      PRINT_INPUT_ERROR("velocity should be three numbers.\n");
    }

    const char* stiff_str = param[7];
    if (strcmp(stiff_str, "couple") == 0) {
      continue_spring_id_ = parse_continue_option(param, num_param, 13);
      stiffness_mode_ = STIFFNESS_COUPLE;
      if (!is_valid_real(param[8], &k_couple_) || k_couple_ <= 0.0) {
        PRINT_INPUT_ERROR("spring constant k should be positive.\n");
      }
      if (!is_valid_real(param[9], &R0_) || R0_ < -1.0e-20) {
        PRINT_INPUT_ERROR("R0 should be a non-negative number.\n");
      }
      if (!is_valid_real(param[10], &offset_[0]) ||
          !is_valid_real(param[11], &offset_[1]) ||
          !is_valid_real(param[12], &offset_[2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }
      if (continue_spring_id_ < 0 &&
          offset_[0] == 0.0 && offset_[1] == 0.0 && offset_[2] == 0.0 && R0_ > 1.0e-20) {
        printf("    Warning: zero offset with positive R0 may lead to weird forces at the beginning of the simulation. So the forces are set to zero initially.\n");
      }
      printf("    ghost_com couple: grouping_method=%d, group_id=%d\n", grouping_method_, group_id_);
      printf(
        "    velocity=(%g,%g,%g) Å/step, k=%g eV/Å^2, R0=%g Å\n",
        velocity_[0], velocity_[1], velocity_[2], k_couple_, R0_);
      printf("    offset=(%g,%g,%g) Å\n", offset_[0], offset_[1], offset_[2]);
    } else if (strcmp(stiff_str, "decouple") == 0) {
      continue_spring_id_ = parse_continue_option(param, num_param, 14);
      stiffness_mode_ = STIFFNESS_DECOUPLE;
      if (!is_valid_real(param[8], &k_decouple_[0]) || k_decouple_[0] < -1.0e-20 ||
          !is_valid_real(param[9], &k_decouple_[1]) || k_decouple_[1] < -1.0e-20 ||
          !is_valid_real(param[10], &k_decouple_[2]) || k_decouple_[2] < -1.0e-20) {
        PRINT_INPUT_ERROR("k components should be non-negative numbers.\n");
      }
      if (!is_valid_real(param[11], &offset_[0]) ||
          !is_valid_real(param[12], &offset_[1]) ||
          !is_valid_real(param[13], &offset_[2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }
      printf("    ghost_com decouple: grouping_method=%d, group_id=%d\n", grouping_method_, group_id_);
      printf(
        "    velocity=(%g,%g,%g) Å/step, k=(%g,%g,%g) eV/Å^2\n",
        velocity_[0], velocity_[1], velocity_[2],
        k_decouple_[0], k_decouple_[1], k_decouple_[2]);
      printf("    offset=(%g,%g,%g) Å\n", offset_[0], offset_[1], offset_[2]);
    } else {
      PRINT_INPUT_ERROR("stiffness mode should be 'couple' or 'decouple'.\n");
    }
  } else if (strcmp(mode_str, "ghost_atom") == 0) {
    mode_ = MODE_GHOST_ATOM;
    if (num_param < 8) {
      PRINT_INPUT_ERROR("add_spring ghost_atom has too few parameters.\n");
    }
    if (!is_valid_int(param[2], &grouping_method_)) {
      PRINT_INPUT_ERROR("grouping method should be an integer.\n");
    }
    if (grouping_method_ < 0 || grouping_method_ >= (int)groups.size()) {
      PRINT_INPUT_ERROR("grouping method is out of range.\n");
    }
    if (!is_valid_int(param[3], &group_id_)) {
      PRINT_INPUT_ERROR("group id should be an integer.\n");
    }
    if (group_id_ < 0 || group_id_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group id is out of range.\n");
    }
    restart_group_size_ = groups[grouping_method_].cpu_size[group_id_];
    if (restart_group_size_ <= 0) {
      PRINT_INPUT_ERROR("The group for add_spring is empty.\n");
    }
    if (!is_valid_real(param[4], &velocity_[0]) ||
        !is_valid_real(param[5], &velocity_[1]) ||
        !is_valid_real(param[6], &velocity_[2])) {
      PRINT_INPUT_ERROR("velocity should be three numbers.\n");
    }

    const char* stiff_str = param[7];
    if (strcmp(stiff_str, "couple") == 0) {
      continue_spring_id_ = parse_continue_option(param, num_param, 13);
      stiffness_mode_ = STIFFNESS_COUPLE;
      if (!is_valid_real(param[8], &k_couple_) || k_couple_ <= 0.0) {
        PRINT_INPUT_ERROR("spring constant k should be positive.\n");
      }
      const double k_total = k_couple_;
      k_couple_ /= restart_group_size_;
      printf("    total k=%g eV/Å^2, k per atom=%g eV/Å^2\n", k_total, k_couple_);
      if (!is_valid_real(param[9], &R0_) || R0_ < -1.0e-20) {
        PRINT_INPUT_ERROR("R0 should be a non-negative number.\n");
      }
      if (!is_valid_real(param[10], &offset_[0]) ||
          !is_valid_real(param[11], &offset_[1]) ||
          !is_valid_real(param[12], &offset_[2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }
      if (continue_spring_id_ < 0 &&
          offset_[0] == 0.0 && offset_[1] == 0.0 && offset_[2] == 0.0 && R0_ > 1.0e-20) {
        printf("    Warning: zero offset with positive R0 may lead to weird forces at the beginning of the simulation. So the forces are set to zero initially.\n");
      }
      printf("    ghost_atom couple: grouping_method=%d, group_id=%d\n", grouping_method_, group_id_);
      printf(
        "    velocity=(%g,%g,%g) Å/step, k=%g eV/Å^2, R0=%g Å\n",
        velocity_[0], velocity_[1], velocity_[2], k_couple_, R0_);
      printf("    offset=(%g,%g,%g) Å\n", offset_[0], offset_[1], offset_[2]);
    } else if (strcmp(stiff_str, "decouple") == 0) {
      continue_spring_id_ = parse_continue_option(param, num_param, 14);
      stiffness_mode_ = STIFFNESS_DECOUPLE;
      if (!is_valid_real(param[8], &k_decouple_[0]) ||
          !is_valid_real(param[9], &k_decouple_[1]) ||
          !is_valid_real(param[10], &k_decouple_[2]) ||
          k_decouple_[0] < -1.0e-20 || k_decouple_[1] < -1.0e-20 || k_decouple_[2] < -1.0e-20) {
        PRINT_INPUT_ERROR("k components should be non-negative numbers.\n");
      }
      const double k_total[3] = {k_decouple_[0], k_decouple_[1], k_decouple_[2]};
      k_decouple_[0] /= restart_group_size_;
      k_decouple_[1] /= restart_group_size_;
      k_decouple_[2] /= restart_group_size_;
      printf(
        "    total k=(%g,%g,%g) eV/Å^2, k per atom=(%g,%g,%g) eV/Å^2\n",
        k_total[0], k_total[1], k_total[2],
        k_decouple_[0], k_decouple_[1], k_decouple_[2]);
      if (!is_valid_real(param[11], &offset_[0]) ||
          !is_valid_real(param[12], &offset_[1]) ||
          !is_valid_real(param[13], &offset_[2])) {
        PRINT_INPUT_ERROR("offset (x0, y0, z0) should be numbers.\n");
      }
      printf("    ghost_atom decouple: grouping_method=%d, group_id=%d\n", grouping_method_, group_id_);
      printf(
        "    velocity=(%g,%g,%g) Å/step, k=(%g,%g,%g) eV/Å^2\n",
        velocity_[0], velocity_[1], velocity_[2],
        k_decouple_[0], k_decouple_[1], k_decouple_[2]);
      printf("    offset=(%g,%g,%g) Å\n", offset_[0], offset_[1], offset_[2]);
    } else {
      PRINT_INPUT_ERROR("stiffness mode should be 'couple' or 'decouple'.\n");
    }

    ghost_atom_group_size_ = restart_group_size_;
    ghost_atom_pos_.resize(3 * ghost_atom_group_size_);
  } else if (strcmp(mode_str, "com_com") == 0) {
    mode_ = MODE_COM_COM;
    if (num_param < 6) {
      PRINT_INPUT_ERROR("add_spring com_com has too few parameters.\n");
    }
    if (!is_valid_int(param[2], &grouping_method_)) {
      PRINT_INPUT_ERROR("grouping_method should be an integer.\n");
    }
    if (grouping_method_ < 0 || grouping_method_ >= (int)groups.size()) {
      PRINT_INPUT_ERROR("grouping_method is out of range.\n");
    }
    if (!is_valid_int(param[3], &group_id_)) {
      PRINT_INPUT_ERROR("group_id_1 should be an integer.\n");
    }
    if (group_id_ < 0 || group_id_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group_id_1 is out of range.\n");
    }
    if (!is_valid_int(param[4], &group_id_2_)) {
      PRINT_INPUT_ERROR("group_id_2 should be an integer.\n");
    }
    if (group_id_2_ < 0 || group_id_2_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group_id_2 is out of range.\n");
    }
    if (groups[grouping_method_].cpu_size[group_id_] <= 0) {
      PRINT_INPUT_ERROR("The first group for add_spring is empty.\n");
    }
    if (groups[grouping_method_].cpu_size[group_id_2_] <= 0) {
      PRINT_INPUT_ERROR("The second group for add_spring is empty.\n");
    }
    if (group_id_ == group_id_2_) {
      PRINT_INPUT_ERROR("group_id_1 and group_id_2 cannot be the same.\n");
    }

    const char* stiff_str = param[5];
    if (strcmp(stiff_str, "couple") == 0) {
      if (num_param != 8) {
        PRINT_INPUT_ERROR("add_spring com_com couple requires 8 parameters.\n");
      }
      stiffness_mode_ = STIFFNESS_COUPLE;
      if (!is_valid_real(param[6], &k_couple_) || k_couple_ <= 0.0) {
        PRINT_INPUT_ERROR("spring constant k should be positive.\n");
      }
      if (!is_valid_real(param[7], &R0_) || R0_ < -1.0e-20) {
        PRINT_INPUT_ERROR("R0 should be a non-negative number.\n");
      }
      printf("    com_com couple: gm=%d, gid1=%d, gid2=%d\n", grouping_method_, group_id_, group_id_2_);
      printf("    k=%g eV/Å^2, R0=%g Å\n", k_couple_, R0_);
    } else if (strcmp(stiff_str, "decouple") == 0) {
      if (num_param != 9) {
        PRINT_INPUT_ERROR("add_spring com_com decouple requires 9 parameters.\n");
      }
      stiffness_mode_ = STIFFNESS_DECOUPLE;
      if (!is_valid_real(param[6], &k_decouple_[0]) ||
          !is_valid_real(param[7], &k_decouple_[1]) ||
          !is_valid_real(param[8], &k_decouple_[2]) ||
          k_decouple_[0] < -1.0e-20 || k_decouple_[1] < -1.0e-20 || k_decouple_[2] < -1.0e-20) {
        PRINT_INPUT_ERROR("k components should be non-negative numbers.\n");
      }
      printf("    com_com decouple: gm=%d, gid1=%d, gid2=%d\n", grouping_method_, group_id_, group_id_2_);
      printf("    k=(%g,%g,%g) eV/Å^2\n", k_decouple_[0], k_decouple_[1], k_decouple_[2]);
    } else {
      PRINT_INPUT_ERROR("stiffness mode should be 'couple' or 'decouple'.\n");
    }
  } else {
    PRINT_INPUT_ERROR("Unknown mode. Use ghost_com, ghost_atom, or com_com.\n");
  }

  spring_id_ = reserve_spring_id(grouping_method_, group_id_, continue_spring_id_);
  printf("    spring id = %d.\n", spring_id_);

  if (continue_spring_id_ >= 0) {
    if (mode_ == MODE_COM_COM) {
      PRINT_INPUT_ERROR("continue is only available for ghost_com and ghost_atom.\n");
    }
    load_restart();
  }

  const std::string filename = spring_stem(grouping_method_, group_id_, spring_id_) + ".out";
  fp_out_ = fopen(filename.c_str(), continue_spring_id_ >= 0 ? "a" : "w");
  if (!fp_out_) {
    std::string error_msg = "Failed to open add_spring output file: " + filename + "\n";
    PRINT_INPUT_ERROR(error_msg.c_str());
  }
  fprintf(fp_out_, "# step  mode  Fx  Fy  Fz Ftotal (eV/Å) energy (eV)\n");
  fflush(fp_out_);
}

Add_Spring::~Add_Spring()
{
  if (fp_out_) {
    fclose(fp_out_);
    fp_out_ = nullptr;
  }
  if (spring_id_ >= 0) {
    release_spring_id(grouping_method_, group_id_, spring_id_);
  }
}

void Add_Spring::save_restart()
{
  if (mode_ == MODE_COM_COM || !init_origin_) {
    return;
  }

  const std::string filename = spring_stem(grouping_method_, group_id_, spring_id_) + ".restart";
  const std::string tmp_filename = filename + ".tmp";
  std::ofstream output(tmp_filename.c_str(), std::ios::out | std::ios::trunc);
  if (!output) {
    std::string error_msg = "Failed to open add_spring restart file for writing: " + tmp_filename + "\n";
    PRINT_INPUT_ERROR(error_msg.c_str());
  }

  const int state_size = (mode_ == MODE_GHOST_COM) ? 3 : 3 * ghost_atom_group_size_;
  output << "GPUMD_ADD_SPRING_RESTART_V2\n";
  output << "mode " << static_cast<int>(mode_) << "\n";
  output << "grouping_method " << grouping_method_ << "\n";
  output << "group_id " << group_id_ << "\n";
  output << "spring_id " << spring_id_ << "\n";
  output << "group_size " << restart_group_size_ << "\n";
  output << "state_size " << state_size << "\n";
  output << "state\n";
  output << std::setprecision(17);
  if (mode_ == MODE_GHOST_COM) {
    output << origin_[0] << " " << origin_[1] << " " << origin_[2] << "\n";
  } else {
    std::vector<double> host_pos(state_size);
    ghost_atom_pos_.copy_to_host(host_pos.data());
    for (int n = 0; n < ghost_atom_group_size_; ++n) {
      output << host_pos[n * 3 + 0] << " "
             << host_pos[n * 3 + 1] << " "
             << host_pos[n * 3 + 2] << "\n";
    }
  }
  output.close();
  if (!output) {
    std::remove(tmp_filename.c_str());
    std::string error_msg = "Failed while writing add_spring restart file: " + tmp_filename + "\n";
    PRINT_INPUT_ERROR(error_msg.c_str());
  }

  if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0) {
    // POSIX rename replaces the destination atomically. Some platforms do not;
    // fall back to remove + rename there. Keep the temporary file if the second
    // rename fails so the newly written restart state can still be recovered.
    if (std::remove(filename.c_str()) != 0) {
      std::string error_msg =
        "Failed to replace the previous add_spring restart file: " + filename +
        ". The new state is kept in " + tmp_filename + "\n";
      PRINT_INPUT_ERROR(error_msg.c_str());
    }
    if (std::rename(tmp_filename.c_str(), filename.c_str()) != 0) {
      std::string error_msg =
        "Failed to finalize add_spring restart file: " + filename +
        ". The new state is kept in " + tmp_filename + "\n";
      PRINT_INPUT_ERROR(error_msg.c_str());
    }
  }
}

void Add_Spring::load_restart()
{
  const std::string filename = spring_stem(grouping_method_, group_id_, spring_id_) + ".restart";
  std::ifstream input(filename.c_str());
  if (!input) {
    std::string error_msg = "Failed to open add_spring restart file: " + filename + "\n";
    PRINT_INPUT_ERROR(error_msg.c_str());
  }

  std::string magic;
  input >> magic;
  if (!input || magic != "GPUMD_ADD_SPRING_RESTART_V2") {
    PRINT_INPUT_ERROR("Invalid or unsupported add_spring restart file header.\n");
  }

  int saved_mode = -1;
  int saved_grouping_method = -1;
  int saved_group_id = -1;
  int saved_spring_id = -1;
  int saved_group_size = -1;
  int state_size = -1;
  if (!read_labeled_int(input, "mode", saved_mode) ||
      !read_labeled_int(input, "grouping_method", saved_grouping_method) ||
      !read_labeled_int(input, "group_id", saved_group_id) ||
      !read_labeled_int(input, "spring_id", saved_spring_id) ||
      !read_labeled_int(input, "group_size", saved_group_size) ||
      !read_labeled_int(input, "state_size", state_size)) {
    PRINT_INPUT_ERROR("Malformed add_spring restart metadata.\n");
  }

  std::string state_label;
  input >> state_label;
  if (!input || state_label != "state") {
    PRINT_INPUT_ERROR("Malformed add_spring restart state section.\n");
  }
  if (saved_mode != static_cast<int>(mode_)) {
    PRINT_INPUT_ERROR("add_spring continuation requires the same ghost mode (ghost_com or ghost_atom).\n");
  }
  if (saved_grouping_method != grouping_method_ || saved_group_id != group_id_) {
    PRINT_INPUT_ERROR("add_spring continuation requires the same grouping method and group id.\n");
  }
  if (saved_spring_id != spring_id_) {
    PRINT_INPUT_ERROR("add_spring restart spring id does not match the requested spring id.\n");
  }
  if (saved_group_size != restart_group_size_) {
    PRINT_INPUT_ERROR("add_spring continuation requires the same group size.\n");
  }

  const int expected_state_size = (mode_ == MODE_GHOST_COM) ? 3 : 3 * ghost_atom_group_size_;
  if (state_size != expected_state_size) {
    PRINT_INPUT_ERROR("add_spring restart state size does not match the current spring.\n");
  }

  std::vector<double> state(state_size);
  for (int n = 0; n < state_size; ++n) {
    if (!(input >> state[n])) {
      PRINT_INPUT_ERROR("add_spring restart state data are incomplete.\n");
    }
  }

  if (mode_ == MODE_GHOST_COM) {
    origin_[0] = state[0];
    origin_[1] = state[1];
    origin_[2] = state[2];
  } else {
    ghost_atom_pos_.copy_from_host(state.data());
  }
  init_origin_ = true;
  printf("    Continue spring %d using %s.\n", spring_id_, filename.c_str());
  printf("    Restored ghost state; the input offset is not reapplied.\n");
}

void Add_Spring::advance_ghost()
{
  if (!init_origin_) {
    return;
  }
  if (mode_ == MODE_GHOST_COM) {
    origin_[0] += velocity_[0];
    origin_[1] += velocity_[1];
    origin_[2] += velocity_[2];
  } else if (mode_ == MODE_GHOST_ATOM) {
    const int block_size = 64;
    const int grid_size = (ghost_atom_group_size_ - 1) / block_size + 1;
    gpu_update_ghost_atom_pos<<<grid_size, block_size>>>(
      ghost_atom_group_size_,
      ghost_atom_pos_.data(),
      velocity_[0],
      velocity_[1],
      velocity_[2]);
    GPU_CHECK_KERNEL
  }
}

void Add_Spring::setup_force(
  const double,
  Integrate&,
  std::vector<Group>& group,
  Atom& atom,
  Box&,
  Force&)
{
  apply_force(group, atom);
}

void Add_Spring::post_force(
  const int step,
  const double,
  Integrate&,
  std::vector<Group>& group,
  Atom& atom,
  Box&,
  Force&)
{
  advance_ghost();
  apply_force(group, atom);
  write_output(step);
}

void Add_Spring::post_run(
  Atom&,
  Box&,
  Integrate&,
  const int,
  const double,
  const double)
{
  save_restart();
  if (fp_out_) {
    fclose(fp_out_);
    fp_out_ = nullptr;
  }
}

void Add_Spring::write_output(const int step)
{
  if (fp_out_ && output_stride_ > 0 && step % output_stride_ == 0) {
    fprintf(
      fp_out_,
      "%d  %d  %g  %g  %g  %g  %g\n",
      step,
      mode_,
      force_[0],
      force_[1],
      force_[2],
      total_force_,
      energy_);
    fflush(fp_out_);
  }
}

void Add_Spring::apply_force(std::vector<Group>& groups, Atom& atom)
{
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
  double* g_x = atom.unwrapped_position.data();
  double* g_y = atom.unwrapped_position.data() + num_atoms_total;
  double* g_z = atom.unwrapped_position.data() + num_atoms_total * 2;
  const int group_size = groups[grouping_method_].cpu_size[group_id_];
  const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];
  const int grid_size = (group_size - 1) / block_size + 1;

  if (mode_ == MODE_GHOST_COM) {
    d_tmp_vec3_.fill(0.0);
    d_tmp_scalar_.fill(0.0);
    gpu_sum_group_mass_pos_reduce<<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_].contents.data(),
      g_x,
      g_y,
      g_z,
      atom.mass.data(),
      d_tmp_vec3_.data(),
      d_tmp_scalar_.data());
    GPU_CHECK_KERNEL

    double h_sum_mx[3], h_sum_m;
    d_tmp_vec3_.copy_to_host(h_sum_mx);
    d_tmp_scalar_.copy_to_host(&h_sum_m);
    const double h_sum_m_inv = 1.0 / h_sum_m;
    const double com_x = h_sum_mx[0] * h_sum_m_inv;
    const double com_y = h_sum_mx[1] * h_sum_m_inv;
    const double com_z = h_sum_mx[2] * h_sum_m_inv;

    if (!init_origin_) {
      origin_[0] = com_x + offset_[0];
      origin_[1] = com_y + offset_[1];
      origin_[2] = com_z + offset_[2];
      init_origin_ = true;
    }

    const double dx = origin_[0] - com_x;
    const double dy = origin_[1] - com_y;
    const double dz = origin_[2] - com_z;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double e = 0.0;
    if (stiffness_mode_ == STIFFNESS_COUPLE) {
      const double r = sqrt(dx * dx + dy * dy + dz * dz);
      const double dr = r - R0_;
      if (r > 1.0e-20) {
        const double f = k_couple_ * dr / r;
        fx = f * dx;
        fy = f * dy;
        fz = f * dz;
      }
      e = 0.5 * k_couple_ * dr * dr;
    } else {
      fx = k_decouple_[0] * dx;
      fy = k_decouple_[1] * dy;
      fz = k_decouple_[2] * dz;
      e = 0.5 *
          (k_decouple_[0] * dx * dx + k_decouple_[1] * dy * dy + k_decouple_[2] * dz * dz);
    }

    gpu_add_force_to_group_mass_weighted<<<grid_size, block_size>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_].contents.data(),
      atom.mass.data(),
      h_sum_m_inv,
      fx,
      fy,
      fz,
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + num_atoms_total,
      atom.force_per_atom.data() + num_atoms_total * 2);
    GPU_CHECK_KERNEL

    force_[0] = fx;
    force_[1] = fy;
    force_[2] = fz;
    energy_ = e;
    total_force_ = sqrt(fx * fx + fy * fy + fz * fz);
  } else if (mode_ == MODE_GHOST_ATOM) {
    if (!init_origin_) {
      gpu_init_ghost_atom_pos<<<grid_size, block_size>>>(
        group_size,
        group_size_sum,
        groups[grouping_method_].contents.data(),
        g_x,
        g_y,
        g_z,
        offset_[0],
        offset_[1],
        offset_[2],
        ghost_atom_pos_.data());
      GPU_CHECK_KERNEL
      init_origin_ = true;
    }

    d_tmp_force3_.fill(0.0);
    d_tmp_scalar_.fill(0.0);
    if (stiffness_mode_ == STIFFNESS_COUPLE) {
      gpu_add_spring_ghost_atom_couple
      <<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
        group_size,
        group_size_sum,
        groups[grouping_method_].contents.data(),
        g_x,
        g_y,
        g_z,
        ghost_atom_pos_.data(),
        k_couple_,
        R0_,
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
        groups[grouping_method_].contents.data(),
        g_x,
        g_y,
        g_z,
        ghost_atom_pos_.data(),
        k_decouple_[0],
        k_decouple_[1],
        k_decouple_[2],
        atom.force_per_atom.data(),
        atom.force_per_atom.data() + num_atoms_total,
        atom.force_per_atom.data() + num_atoms_total * 2,
        d_tmp_force3_.data(),
        d_tmp_scalar_.data());
    }
    GPU_CHECK_KERNEL

    double h_force[3];
    d_tmp_force3_.copy_to_host(h_force);
    double h_energy = 0.0;
    d_tmp_scalar_.copy_to_host(&h_energy);
    force_[0] = h_force[0];
    force_[1] = h_force[1];
    force_[2] = h_force[2];
    total_force_ = sqrt(h_force[0] * h_force[0] + h_force[1] * h_force[1] + h_force[2] * h_force[2]);
    energy_ = h_energy;
  } else {
    d_tmp_vec3_.fill(0.0);
    d_tmp_scalar_.fill(0.0);
    gpu_sum_group_mass_pos_reduce
    <<<grid_size, block_size, block_size * 4 * sizeof(double)>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_].contents.data(),
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

    const int group2_size = groups[grouping_method_].cpu_size[group_id_2_];
    const int group2_size_sum = groups[grouping_method_].cpu_size_sum[group_id_2_];
    const int grid_size_2 = (group2_size - 1) / block_size + 1;
    d_tmp_vec3_.fill(0.0);
    d_tmp_scalar_.fill(0.0);
    gpu_sum_group_mass_pos_reduce
    <<<grid_size_2, block_size, block_size * 4 * sizeof(double)>>>(
      group2_size,
      group2_size_sum,
      groups[grouping_method_].contents.data(),
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
    const double h_sum_m1_inv = 1.0 / h_sum_m1;
    const double h_sum_m2_inv = 1.0 / h_sum_m2;
    const double com1_x = h_sum_mx1[0] * h_sum_m1_inv;
    const double com1_y = h_sum_mx1[1] * h_sum_m1_inv;
    const double com1_z = h_sum_mx1[2] * h_sum_m1_inv;
    const double com2_x = h_sum_mx2[0] * h_sum_m2_inv;
    const double com2_y = h_sum_mx2[1] * h_sum_m2_inv;
    const double com2_z = h_sum_mx2[2] * h_sum_m2_inv;

    const double dx = com2_x - com1_x;
    const double dy = com2_y - com1_y;
    const double dz = com2_z - com1_z;
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double e = 0.0;
    if (stiffness_mode_ == STIFFNESS_COUPLE) {
      const double r = sqrt(dx * dx + dy * dy + dz * dz);
      const double dr = r - R0_;
      if (r > 1.0e-20) {
        const double f = k_couple_ * dr / r;
        fx = f * dx;
        fy = f * dy;
        fz = f * dz;
      }
      e = 0.5 * k_couple_ * dr * dr;
    } else {
      fx = k_decouple_[0] * dx;
      fy = k_decouple_[1] * dy;
      fz = k_decouple_[2] * dz;
      e = 0.5 *
          (k_decouple_[0] * dx * dx + k_decouple_[1] * dy * dy + k_decouple_[2] * dz * dz);
    }

    const int max_size = std::max(group_size, group2_size);
    const int grid_size_max = (max_size - 1) / block_size + 1;
    gpu_add_spring_com_com_force<<<grid_size_max, block_size>>>(
      groups[grouping_method_].contents.data(),
      group_size,
      group_size_sum,
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

    force_[0] = fx;
    force_[1] = fy;
    force_[2] = fz;
    energy_ = e;
    total_force_ = sqrt(fx * fx + fy * fy + fz * fz);
  }
}
