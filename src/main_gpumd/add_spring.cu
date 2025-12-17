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
#include <limits>

#ifndef SMALL
#define SMALL 1.0e-20
#endif


__global__ void gpu_sum_group_mass_pos_reduce(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* __restrict__ g_mass,   // length = N
  double* __restrict__ d_sum_mr,        // length = 3
  double* __restrict__ d_sum_m)         // length = 1
{
  extern __shared__ double s[];
  double* sx = s;
  double* sy = s + blockDim.x;
  double* sz = s + 2 * blockDim.x;
  double* sm = s + 3 * blockDim.x;

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  double mx = 0.0, my = 0.0, mz = 0.0, m = 0.0;
  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    m  = g_mass[atom_id];
    mx = m * g_x[atom_id];
    my = m * g_y[atom_id];
    mz = m * g_z[atom_id];
  }

  sx[threadIdx.x] = mx;
  sy[threadIdx.x] = my;
  sz[threadIdx.x] = mz;
  sm[threadIdx.x] = m;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sx[threadIdx.x] += sx[threadIdx.x + offset];
      sy[threadIdx.x] += sy[threadIdx.x + offset];
      sz[threadIdx.x] += sz[threadIdx.x + offset];
      sm[threadIdx.x] += sm[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&d_sum_mr[0], sx[0]);
    atomicAdd(&d_sum_mr[1], sy[0]);
    atomicAdd(&d_sum_mr[2], sz[0]);
    atomicAdd(d_sum_m,      sm[0]);
  }
}

__global__ void gpu_add_force_to_group_mass_weighted(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_mass,
  const double inv_M,                // 1 / sum(m)
  const double Fx_cm,
  const double Fy_cm,
  const double Fz_cm,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= group_size) return;

  const int atom_id = g_group_contents[group_size_sum + tid];
  const double a    = g_mass[atom_id] * inv_M;

  g_fx[atom_id] += Fx_cm * a;
  g_fy[atom_id] += Fy_cm * a;
  g_fz[atom_id] += Fz_cm * a;
}

static inline const char* spring_mode_string(const SpringStiffMode m)
{
  if (m == SPRING_COUPLE) return "ghost_com_couple";
  return "ghost_com_decouple";
}

static inline double spring_nan()
{
  return std::numeric_limits<double>::quiet_NaN();
}

static inline double friction_from_force_and_velocity(
  const double Fx, const double Fy, const double Fz,
  const double vx, const double vy, const double vz)
{
  const double v2 = vx * vx + vy * vy + vz * vz;
  if (v2 <= 0.0) return spring_nan();

  const double inv_v = 1.0 / sqrt(v2);
  const double vhatx = vx * inv_v;
  const double vhaty = vy * inv_v;
  const double vhatz = vz * inv_v;

  // friction scalar along driving direction: -F · vhat
  return -(Fx * vhatx + Fy * vhaty + Fz * vhatz);
}

void Add_Spring::parse(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom)
{
  printf("Add spring.\n");
  if (atom.unwrapped_position.size() < atom.position_per_atom.size()) {
    atom.unwrapped_position.resize(atom.position_per_atom.size());
    atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
  }
  if (atom.position_temp.size() < atom.position_per_atom.size()) {
    atom.position_temp.resize(atom.position_per_atom.size());
  }

  if (num_calls_ >= MAX_SPRING_CALLS) {
    PRINT_INPUT_ERROR("add_spring cannot be used more than 10 times in one run.\n");
  }

  const char* mode_str = param[1];
  const int id = num_calls_;

  if (strcmp(mode_str, "ghost_com") != 0) {
    PRINT_INPUT_ERROR("Only add_spring ghost_com is supported now.\n"
                      "Use:\n"
                      "  add_spring ghost_com gm gid vx vy vz couple   k  R0  x0 y0 z0\n"
                      "  add_spring ghost_com gm gid vx vy vz decouple kx ky kz x0 y0 z0\n");
  }

  // ----- gm gid -----
  if (!is_valid_int(param[2], &grouping_method_[id])) {
    PRINT_INPUT_ERROR("group_method should be an integer.\n");
  }
  if (grouping_method_[id] < 0 || grouping_method_[id] >= (int)groups.size()) {
    PRINT_INPUT_ERROR("group_method is out of range.\n");
  }

  if (!is_valid_int(param[3], &group_id_[id])) {
    PRINT_INPUT_ERROR("group_id should be an integer.\n");
  }
  if (group_id_[id] < 0 || group_id_[id] >= groups[grouping_method_[id]].number) {
    PRINT_INPUT_ERROR("group_id is out of range.\n");
  }

  // ----- velocity (Å/step): read first after gm/gid -----
  if (!is_valid_real(param[4], &ghost_velocity_[id][0]) ||
      !is_valid_real(param[5], &ghost_velocity_[id][1]) ||
      !is_valid_real(param[6], &ghost_velocity_[id][2])) {
    PRINT_INPUT_ERROR("velocity (vx, vy, vz) should be numbers.\n");
  }

  // ----- stiffness mode + parameters -----
  const char* stiff_str = param[7];

  if (strcmp(stiff_str, "couple") == 0 || strcmp(stiff_str, "coupled") == 0) {

    // add_spring ghost_com gm gid vx vy vz couple k R0 x0 y0 z0
    if (num_param != 13) {
      PRINT_INPUT_ERROR("add_spring ghost_com couple requires:\n"
                        "  add_spring ghost_com gm gid vx vy vz couple k R0 x0 y0 z0\n");
    }

    stiff_mode_[id] = SPRING_COUPLE;

    if (!is_valid_real(param[8], &k_couple_[id]) || k_couple_[id] <= 0.0) {
      PRINT_INPUT_ERROR("spring constant k should be a positive number.\n");
    }
    if (!is_valid_real(param[9], &R0_[id])) {
      PRINT_INPUT_ERROR("R0 should be a number.\n");
    }

    // offsets relative to initial COM
    if (!is_valid_real(param[10], &ghost_offset_[id][0]) ||
        !is_valid_real(param[11], &ghost_offset_[id][1]) ||
        !is_valid_real(param[12], &ghost_offset_[id][2])) {
      PRINT_INPUT_ERROR("x0, y0, z0 should be numbers (offsets relative to initial COM).\n");
    }

    // origin will be initialized at first compute as: Rg(0) = Rcm(init) + offset
    ghost_origin_[id][0] = ghost_origin_[id][1] = ghost_origin_[id][2] = 0.0;
    init_origin_[id] = 0;

    // decouple params unused
    k_decouple_[id][0] = k_decouple_[id][1] = k_decouple_[id][2] = 0.0;

    printf("  ghost_com couple: group method %d, group id %d, v=(%g,%g,%g)[Å/step], k=%g [eV/Å^2], R0=%g [Å], "
           "offset=(%g,%g,%g) [Å]\n",
           grouping_method_[id], group_id_[id],
           ghost_velocity_[id][0], ghost_velocity_[id][1], ghost_velocity_[id][2],
           k_couple_[id], R0_[id],
           ghost_offset_[id][0], ghost_offset_[id][1], ghost_offset_[id][2]);

  } else if (strcmp(stiff_str, "decouple") == 0 || strcmp(stiff_str, "decoupled") == 0) {

    // add_spring ghost_com gm gid vx vy vz decouple kx ky kz x0 y0 z0
    if (num_param != 14) {
      PRINT_INPUT_ERROR("add_spring ghost_com decouple requires:\n"
                        "  add_spring ghost_com gm gid vx vy vz decouple kx ky kz x0 y0 z0\n");
    }

    stiff_mode_[id] = SPRING_DECOUPLE;

    if (!is_valid_real(param[8],  &k_decouple_[id][0]) ||
        !is_valid_real(param[9],  &k_decouple_[id][1]) ||
        !is_valid_real(param[10], &k_decouple_[id][2])) {
      PRINT_INPUT_ERROR("kx, ky, kz should be numbers.\n");
    }
    if (k_decouple_[id][0] < 0.0 || k_decouple_[id][1] < 0.0 || k_decouple_[id][2] < 0.0) {
      PRINT_INPUT_ERROR("kx, ky, kz should be non-negative.\n");
    }
    if (k_decouple_[id][0] == 0.0 && k_decouple_[id][1] == 0.0 && k_decouple_[id][2] == 0.0) {
      PRINT_INPUT_ERROR("At least one of kx, ky, kz must be > 0.\n");
    }

    // offsets relative to initial COM
    if (!is_valid_real(param[11], &ghost_offset_[id][0]) ||
        !is_valid_real(param[12], &ghost_offset_[id][1]) ||
        !is_valid_real(param[13], &ghost_offset_[id][2])) {
      PRINT_INPUT_ERROR("x0, y0, z0 should be numbers (offsets relative to initial COM).\n");
    }

    // couple params unused
    k_couple_[id] = 0.0;
    R0_[id]       = 0.0;

    // origin will also be initialized at first compute: Rg(0) = Rcm(init) + offset
    ghost_origin_[id][0] = ghost_origin_[id][1] = ghost_origin_[id][2] = 0.0;
    init_origin_[id] = 0;

    printf("  ghost_com decouple: group method %d, group id %d, v=(%g,%g,%g)[Å/step], "
           "k=(%g,%g,%g) [eV/Å^2], offset=(%g,%g,%g) [Å]\n",
           grouping_method_[id], group_id_[id],
           ghost_velocity_[id][0], ghost_velocity_[id][1], ghost_velocity_[id][2],
           k_decouple_[id][0], k_decouple_[id][1], k_decouple_[id][2],
           ghost_offset_[id][0], ghost_offset_[id][1], ghost_offset_[id][2]);

  } else {
    PRINT_INPUT_ERROR("Unknown stiffness mode. Use couple or decouple.\n");
  }

  spring_energy_[id] = 0.0;
  spring_force_[id][0] = spring_force_[id][1] = spring_force_[id][2] = 0.0;
  spring_fric_[id] = spring_nan();


  // output file of id
  if (fp_out_[id] == nullptr && output_stride_ > 0) {
    // filename: spring_force_id.out
    std::string filename = "spring_force_";;
    filename += std::to_string(id);
    filename += ".out"; 
    fp_out_[id] = fopen(filename.c_str(), "w");
    if (fp_out_[id]) {
      fprintf(fp_out_[id], "# step  call  mode  Fx  Fy  Fz  energy  fric\n");
      fflush(fp_out_[id]);
    } else {
      printf("WARNING: cannot open spring_force.out for writing.\n");
    }
  }

  ++num_calls_;
}

void Add_Spring::compute(const int step,
                         const std::vector<Group>& groups,
                         Atom& atom)
{
  if (!d_tmp_vec3_)   gpuMalloc((void**)&d_tmp_vec3_,   3 * sizeof(double));
  if (!d_tmp_scalar_) gpuMalloc((void**)&d_tmp_scalar_, 1 * sizeof(double));

  const int n_atoms_total = (int) atom.position_per_atom.size() / 3;

  double* g_pos = atom.unwrapped_position.data();
  if (atom.unwrapped_position.size() == 0) {
    g_pos = atom.position_per_atom.data();
    if (!printed_use_wrapped_position_) {
      printf("WARNING: Atom::unwrapped_position is empty. add_spring will use wrapped positions.\n");
      printed_use_wrapped_position_ = 1;
    }
  }

  const double* g_x = g_pos;
  const double* g_y = g_pos + n_atoms_total;
  const double* g_z = g_pos + 2 * n_atoms_total;

  double* g_force = atom.force_per_atom.data();
  double* g_fx    = g_force;
  double* g_fy    = g_force + n_atoms_total;
  double* g_fz    = g_force + 2 * n_atoms_total;

  double* g_mass = atom.mass.data();
  if (atom.mass.size() == 0) {
    PRINT_INPUT_ERROR("Atom::mass is empty, but add_spring requires per-atom masses.\n");
  }

  const int block_size = 64;
  const double step_d  = (double) step;
  const size_t shmem_bytes = 4 * block_size * sizeof(double);

  for (int c = 0; c < num_calls_; ++c) {

    spring_energy_[c] = 0.0;
    spring_force_[c][0] = spring_force_[c][1] = spring_force_[c][2] = 0.0;
    spring_fric_[c] = spring_nan();

    const int gm  = grouping_method_[c];
    const int gid = group_id_[c];
    const Group& G = groups[gm];

    const int group_size     = G.cpu_size[gid];
    const int group_size_sum = G.cpu_size_sum[gid];
    if (group_size <= 0) continue;

    const int* g_contents = G.contents.data();
    const int grid_size   = (group_size - 1) / block_size + 1;

    // mass-weighted COM
    gpuMemset(d_tmp_vec3_,   0, 3 * sizeof(double));
    gpuMemset(d_tmp_scalar_, 0,     sizeof(double));

    gpu_sum_group_mass_pos_reduce<<<grid_size, block_size, shmem_bytes>>>(
      group_size, group_size_sum, g_contents,
      g_x, g_y, g_z, g_mass,
      d_tmp_vec3_, d_tmp_scalar_);
    GPU_CHECK_KERNEL

    double sum_mr[3], M;
    gpuMemcpy(sum_mr, d_tmp_vec3_,   3 * sizeof(double), gpuMemcpyDeviceToHost);
    gpuMemcpy(&M,     d_tmp_scalar_, 1 * sizeof(double), gpuMemcpyDeviceToHost);
    if (M <= 0.0) continue;

    const double Rcm[3] = { sum_mr[0] / M, sum_mr[1] / M, sum_mr[2] / M };


    if (!init_origin_[c]) {
      ghost_origin_[c][0] = Rcm[0] + ghost_offset_[c][0] - ghost_velocity_[c][0] * step_d;
      ghost_origin_[c][1] = Rcm[1] + ghost_offset_[c][1] - ghost_velocity_[c][1] * step_d;
      ghost_origin_[c][2] = Rcm[2] + ghost_offset_[c][2] - ghost_velocity_[c][2] * step_d;
      init_origin_[c] = 1;
    }

    // ghost position at current step
    const double Rg[3] = {
      ghost_origin_[c][0] + ghost_velocity_[c][0] * step_d,
      ghost_origin_[c][1] + ghost_velocity_[c][1] * step_d,
      ghost_origin_[c][2] + ghost_velocity_[c][2] * step_d
    };

    const double dx = Rcm[0] - Rg[0];
    const double dy = Rcm[1] - Rg[1];
    const double dz = Rcm[2] - Rg[2];

    double Fx_cm = 0.0, Fy_cm = 0.0, Fz_cm = 0.0, energy = 0.0;

    if (stiff_mode_[c] == SPRING_COUPLE) {

      const double k  = k_couple_[c];
      const double R0 = R0_[c];
      const double r2 = dx * dx + dy * dy + dz * dz;

      if (R0 <= 0.0) {
        // U = 1/2 k |d|^2  (equivalent to R0=0 in radial form)
        Fx_cm  = -k * dx;
        Fy_cm  = -k * dy;
        Fz_cm  = -k * dz;
        energy = 0.5 * k * r2;
      } else {
        const double r    = sqrt(r2) + SMALL;
        const double dr   = r - R0;
        const double coef = -k * dr / r;
        Fx_cm  = coef * dx;
        Fy_cm  = coef * dy;
        Fz_cm  = coef * dz;
        energy = 0.5 * k * dr * dr;
      }

    } else { // SPRING_DECOUPLE

      const double kx = k_decouple_[c][0];
      const double ky = k_decouple_[c][1];
      const double kz = k_decouple_[c][2];

      Fx_cm  = -kx * dx;
      Fy_cm  = -ky * dy;
      Fz_cm  = -kz * dz;
      energy = 0.5 * (kx * dx * dx + ky * dy * dy + kz * dz * dz);
    }

    spring_energy_[c]   = energy;
    spring_force_[c][0] = Fx_cm;
    spring_force_[c][1] = Fy_cm;
    spring_force_[c][2] = Fz_cm;

    spring_fric_[c] = friction_from_force_and_velocity(
      Fx_cm, Fy_cm, Fz_cm,
      ghost_velocity_[c][0],
      ghost_velocity_[c][1],
      ghost_velocity_[c][2]);

    const double inv_M = 1.0 / M;
    gpu_add_force_to_group_mass_weighted<<<grid_size, block_size>>>(
      group_size,
      group_size_sum,
      g_contents,
      g_mass,
      inv_M,
      Fx_cm, Fy_cm, Fz_cm,
      g_fx, g_fy, g_fz);
    GPU_CHECK_KERNEL

    // write output
    if (fp_out_[c] && output_stride_ > 0 && (step % output_stride_ == 0)) {
      fprintf(fp_out_[c], "%d %d %s %.15e %.15e %.15e %.15e %.15e\n",
              step, c, spring_mode_string(stiff_mode_[c]),
              spring_force_[c][0], spring_force_[c][1], spring_force_[c][2],
              spring_energy_[c], spring_fric_[c]);
    }
  }

}

void Add_Spring::finalize()
{
  if (d_tmp_vec3_)   { gpuFree(d_tmp_vec3_);   d_tmp_vec3_   = nullptr; }
  if (d_tmp_scalar_) { gpuFree(d_tmp_scalar_); d_tmp_scalar_ = nullptr; }

  for (int id = 0; id < MAX_SPRING_CALLS; ++id) {
    if (fp_out_[id]) {
      fclose(fp_out_[id]);
      fp_out_[id] = nullptr;
    }
  }
  num_calls_ = 0;
  printed_use_wrapped_position_ = 0;

  for (int c = 0; c < MAX_SPRING_CALLS; ++c) {
    init_origin_[c] = 0;
  }
}
