/*
   add_spring.cu

   Three spring modes:
     1) ghost_com   : moving ghost point coupled to the COM of a group
     2) ghost_atom  : moving ghost anchors (one per atom) for a group
     3) couple_com  : a spring between the COMs of two groups

   Syntax (step-based velocities):

     add_spring ghost_com  gm gid  k  xg yg zg  R0  vx vy vz
     add_spring ghost_atom gm gid  k  R0        vx vy vz
     add_spring couple_com gm1 gid1  gm2 gid2  k  R0

   Here (vx, vy, vz) are in units of Å per MD step, i.e.,
     R_g(step) = R_g(0) + v * step
*/

#include "add_spring.cuh"

#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef SMALL
#define SMALL 1.0e-20
#endif

// ---------------------------------------------------------------------------
// device kernels
// ---------------------------------------------------------------------------

// Sum positions of atoms in a group: d_sum[0..2], d_count[0] = number of atoms
__global__ void kernel_sum_group_pos(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* __restrict__ d_sum,
  double* __restrict__ d_count)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= group_size) return;

  const int atom_id = g_group_contents[group_size_sum + tid];

  const double x = g_x[atom_id];
  const double y = g_y[atom_id];
  const double z = g_z[atom_id];

  atomicAdd(&d_sum[0], x);
  atomicAdd(&d_sum[1], y);
  atomicAdd(&d_sum[2], z);
  atomicAdd(d_count, 1.0);
}

// Initialize ghost_atom anchors at step=0 as the current positions
__global__ void kernel_init_ghost_atom_pos(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* __restrict__ ghost_pos)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= group_size) return;

  const int atom_id = g_group_contents[group_size_sum + tid];

  ghost_pos[tid]                 = g_x[atom_id];
  ghost_pos[group_size + tid]    = g_y[atom_id];
  ghost_pos[2 * group_size + tid] = g_z[atom_id];
}

// Distribute a given total COM spring force (Fx_cm, Fy_cm, Fz_cm)
// equally to all atoms in the group.
__global__ void kernel_add_spring_ghost_com(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
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

  const double fx_atom = Fx_cm / (double) group_size;
  const double fy_atom = Fy_cm / (double) group_size;
  const double fz_atom = Fz_cm / (double) group_size;

  g_fx[atom_id] += fx_atom;
  g_fy[atom_id] += fy_atom;
  g_fz[atom_id] += fz_atom;
}

// ghost_atom: each atom is connected to its ghost anchor,
// which moves at a constant velocity (vx_g, vy_g, vz_g).
// The initial anchors at step=0 are stored in ghost_pos (device).
// We also accumulate spring energy to d_energy by atomicAdd.
__global__ void kernel_add_spring_ghost_atom(
  const int group_size,
  const int group_size_sum,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ ghost_pos,  // length = 3 * group_size (x0, y0, z0)
  const double k,
  const double R0,
  const double vx_g,
  const double vy_g,
  const double vz_g,
  const double step,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  double* __restrict__ d_energy)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= group_size) return;

  const int atom_id = g_group_contents[group_size_sum + tid];

  const double x = g_x[atom_id];
  const double y = g_y[atom_id];
  const double z = g_z[atom_id];

  const double xg0 = ghost_pos[tid];
  const double yg0 = ghost_pos[group_size + tid];
  const double zg0 = ghost_pos[2 * group_size + tid];

  // ghost anchor position at this step
  const double xg = xg0 + vx_g * step;
  const double yg = yg0 + vy_g * step;
  const double zg = zg0 + vz_g * step;

  double dx = x - xg;
  double dy = y - yg;
  double dz = z - zg;

  const double r2 = dx * dx + dy * dy + dz * dz;

  double fx, fy, fz, e;

  if (R0 <= 0.0) {
    // Vector spring: U = 0.5 k |d|^2, F = -k d
    fx = -k * dx;
    fy = -k * dy;
    fz = -k * dz;
    e  = 0.5 * k * r2;
  } else {
    const double r  = sqrt(r2) + SMALL;
    const double dr = r - R0;
    const double coef = -k * dr / r;  // F = -k (r - R0) d / r
    fx = coef * dx;
    fy = coef * dy;
    fz = coef * dz;
    e  = 0.5 * k * dr * dr;
  }

  g_fx[atom_id] += fx;
  g_fy[atom_id] += fy;
  g_fz[atom_id] += fz;

  atomicAdd(d_energy, e);
}

// couple_com: group1 COM to group2 COM, with total spring force F1 = +Fcm on group1,
// F2 = -Fcm on group2. We distribute F1/F2 evenly across atoms in each group.
__global__ void kernel_add_spring_couple_com(
  const int group1_size,
  const int group1_size_sum,
  const int* __restrict__ g_contents1,
  const int group2_size,
  const int group2_size_sum,
  const int* __restrict__ g_contents2,
  const double Fx_cm,
  const double Fy_cm,
  const double Fz_cm,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < group1_size) {
    const int atom_id = g_contents1[group1_size_sum + tid];
    const double fx_atom = Fx_cm / (double) group1_size;
    const double fy_atom = Fy_cm / (double) group1_size;
    const double fz_atom = Fz_cm / (double) group1_size;
    g_fx[atom_id] += fx_atom;
    g_fy[atom_id] += fy_atom;
    g_fz[atom_id] += fz_atom;
  }

  if (tid < group2_size) {
    const int atom_id = g_contents2[group2_size_sum + tid];
    const double fx_atom = Fx_cm / (double) group2_size;
    const double fy_atom = Fy_cm / (double) group2_size;
    const double fz_atom = Fz_cm / (double) group2_size;
    g_fx[atom_id] -= fx_atom;
    g_fy[atom_id] -= fy_atom;
    g_fz[atom_id] -= fz_atom;
  }
}

// ---------------------------------------------------------------------------
// Add_Spring::parse
// ---------------------------------------------------------------------------

void Add_Spring::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  printf("Add spring.\n");

  if (num_calls_ >= MAX_SPRING_CALLS) {
    PRINT_INPUT_ERROR("add_spring cannot be used more than 10 times in one run.\n");
  }

  const char* mode_str = param[1];
  const int id = num_calls_;

  if (strcmp(mode_str, "ghost_com") == 0) {

    // add_spring ghost_com gm gid k xg yg zg R0 vx vy vz
    if (num_param != 12) {
      PRINT_INPUT_ERROR("add_spring ghost_com requires: "
                        "add_spring ghost_com gm gid k xg yg zg R0 vx vy vz\n");
    }

    mode_[id] = SPRING_GHOST_COM;

    // group method & id
    if (!is_valid_int(param[2], &grouping_method1_[id])) {
      PRINT_INPUT_ERROR("group_method should be an integer.\n");
    }
    if (grouping_method1_[id] < 0 || grouping_method1_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("group_method is out of range.\n");
    }

    if (!is_valid_int(param[3], &group_id1_[id])) {
      PRINT_INPUT_ERROR("group_id should be an integer.\n");
    }
    if (group_id1_[id] < 0 ||
        group_id1_[id] >= groups[grouping_method1_[id]].number) {
      PRINT_INPUT_ERROR("group_id is out of range.\n");
    }

    // k
    if (!is_valid_real(param[4], &k_[id]) || k_[id] <= 0.0) {
      PRINT_INPUT_ERROR("spring constant k should be a positive number.\n");
    }

    // ghost origin
    if (!is_valid_real(param[5], &ghost_com_origin_[id][0]) ||
        !is_valid_real(param[6], &ghost_com_origin_[id][1]) ||
        !is_valid_real(param[7], &ghost_com_origin_[id][2])) {
      PRINT_INPUT_ERROR("ghost_com origin (xg, yg, zg) should be numbers.\n");
    }

    // R0
    if (!is_valid_real(param[8], &R0_[id])) {
      PRINT_INPUT_ERROR("R0 should be a number.\n");
    }

    // ghost velocity (per step)
    if (!is_valid_real(param[9],  &ghost_com_velocity_[id][0]) ||
        !is_valid_real(param[10], &ghost_com_velocity_[id][1]) ||
        !is_valid_real(param[11], &ghost_com_velocity_[id][2])) {
      PRINT_INPUT_ERROR("ghost_com velocity (vx, vy, vz) should be numbers.\n");
    }

    // not used for this mode
    grouping_method2_[id] = -1;
    group_id2_[id]        = -1;

    printf("  ghost_com: gm=%d gid=%d, k=%g, origin=(%g,%g,%g), R0=%g, v=(%g,%g,%g) [Å/step]\n",
           grouping_method1_[id], group_id1_[id], k_[id],
           ghost_com_origin_[id][0],
           ghost_com_origin_[id][1],
           ghost_com_origin_[id][2],
           R0_[id],
           ghost_com_velocity_[id][0],
           ghost_com_velocity_[id][1],
           ghost_com_velocity_[id][2]);

  } else if (strcmp(mode_str, "ghost_atom") == 0) {

    // add_spring ghost_atom gm gid k R0 vx vy vz
    if (num_param != 10) {
      PRINT_INPUT_ERROR("add_spring ghost_atom requires: "
                        "add_spring ghost_atom gm gid k R0 vx vy vz\n");
    }

    mode_[id] = SPRING_GHOST_ATOM;

    if (!is_valid_int(param[2], &grouping_method1_[id])) {
      PRINT_INPUT_ERROR("group_method should be an integer.\n");
    }
    if (grouping_method1_[id] < 0 || grouping_method1_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("group_method is out of range.\n");
    }

    if (!is_valid_int(param[3], &group_id1_[id])) {
      PRINT_INPUT_ERROR("group_id should be an integer.\n");
    }
    if (group_id1_[id] < 0 ||
        group_id1_[id] >= groups[grouping_method1_[id]].number) {
      PRINT_INPUT_ERROR("group_id is out of range.\n");
    }

    if (!is_valid_real(param[4], &k_[id]) || k_[id] <= 0.0) {
      PRINT_INPUT_ERROR("spring constant k should be a positive number.\n");
    }

    if (!is_valid_real(param[5], &R0_[id])) {
      PRINT_INPUT_ERROR("R0 should be a number.\n");
    }

    if (!is_valid_real(param[6], &ghost_atom_velocity_[id][0]) ||
        !is_valid_real(param[7], &ghost_atom_velocity_[id][1]) ||
        !is_valid_real(param[8], &ghost_atom_velocity_[id][2])) {
      PRINT_INPUT_ERROR("ghost_atom velocity (vx, vy, vz) should be numbers.\n");
    }

    grouping_method2_[id] = -1;
    group_id2_[id]        = -1;

    d_ghost_atom_pos_[id]      = nullptr;
    ghost_atom_group_size_[id] = 0;

    printf("  ghost_atom: gm=%d gid=%d, k=%g, R0=%g, v=(%g,%g,%g) [Å/step]\n",
           grouping_method1_[id], group_id1_[id], k_[id], R0_[id],
           ghost_atom_velocity_[id][0],
           ghost_atom_velocity_[id][1],
           ghost_atom_velocity_[id][2]);

  } else if (strcmp(mode_str, "couple_com") == 0) {

    // add_spring couple_com gm1 gid1 gm2 gid2 k R0
    if (num_param != 9) {
      PRINT_INPUT_ERROR("add_spring couple_com requires: "
                        "add_spring couple_com gm1 gid1 gm2 gid2 k R0\n");
    }

    mode_[id] = SPRING_COUPLE_COM;

    // first group
    if (!is_valid_int(param[2], &grouping_method1_[id])) {
      PRINT_INPUT_ERROR("group_method1 should be an integer.\n");
    }
    if (grouping_method1_[id] < 0 || grouping_method1_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("group_method1 is out of range.\n");
    }

    if (!is_valid_int(param[3], &group_id1_[id])) {
      PRINT_INPUT_ERROR("group_id1 should be an integer.\n");
    }
    if (group_id1_[id] < 0 ||
        group_id1_[id] >= groups[grouping_method1_[id]].number) {
      PRINT_INPUT_ERROR("group_id1 is out of range.\n");
    }

    // second group
    if (!is_valid_int(param[4], &grouping_method2_[id])) {
      PRINT_INPUT_ERROR("group_method2 should be an integer.\n");
    }
    if (grouping_method2_[id] < 0 || grouping_method2_[id] >= (int)groups.size()) {
      PRINT_INPUT_ERROR("group_method2 is out of range.\n");
    }

    if (!is_valid_int(param[5], &group_id2_[id])) {
      PRINT_INPUT_ERROR("group_id2 should be an integer.\n");
    }
    if (group_id2_[id] < 0 ||
        group_id2_[id] >= groups[grouping_method2_[id]].number) {
      PRINT_INPUT_ERROR("group_id2 is out of range.\n");
    }

    if (!is_valid_real(param[6], &k_[id]) || k_[id] <= 0.0) {
      PRINT_INPUT_ERROR("spring constant k should be a positive number.\n");
    }

    if (!is_valid_real(param[7], &R0_[id])) {
      PRINT_INPUT_ERROR("R0 should be a number.\n");
    }

    printf("  couple_com: (gm1,gid1)=(%d,%d), (gm2,gid2)=(%d,%d), k=%g, R0=%g\n",
           grouping_method1_[id], group_id1_[id],
           grouping_method2_[id], group_id2_[id],
           k_[id], R0_[id]);

  } else {
    PRINT_INPUT_ERROR("Unknown add_spring mode. Use ghost_com, ghost_atom, or couple_com.\n");
  }

  spring_energy_[id] = 0.0;
  ++num_calls_;
}

// ---------------------------------------------------------------------------
// Add_Spring::compute
// ---------------------------------------------------------------------------

void Add_Spring::compute(const int step,
                         const std::vector<Group>& groups,
                         Atom& atom)
{
  if (num_calls_ == 0) return;

  // lazily allocate scratch buffers
  if (!d_tmp_vec3_) {
    cudaMalloc((void**)&d_tmp_vec3_,   3 * sizeof(double));
  }
  if (!d_tmp_scalar_) {
    cudaMalloc((void**)&d_tmp_scalar_, sizeof(double));
  }

  const int n_atoms_total = (int) atom.position_per_atom.size() / 3;

  double* g_pos = atom.position_per_atom.data();
  double* g_x   = g_pos;
  double* g_y   = g_pos + n_atoms_total;
  double* g_z   = g_pos + 2 * n_atoms_total;

  double* g_force = atom.force_per_atom.data();
  double* g_fx    = g_force;
  double* g_fy    = g_force + n_atoms_total;
  double* g_fz    = g_force + 2 * n_atoms_total;

  const int block_size = 64;
  const double step_d  = (double) step;

  for (int c = 0; c < num_calls_; ++c) {
    spring_energy_[c] = 0.0;

    if (mode_[c] == SPRING_GHOST_COM) {

      const int gm  = grouping_method1_[c];
      const int gid = group_id1_[c];
      const Group& G = groups[gm];

      const int group_size     = G.cpu_size[gid];
      const int group_size_sum = G.cpu_size_sum[gid];

      if (group_size <= 0) continue;

      const int* g_contents = G.contents.data();
      const int grid_size   = (group_size - 1) / block_size + 1;

      // compute COM (geometric center) on device
      cudaMemset(d_tmp_vec3_,   0, 3 * sizeof(double));
      cudaMemset(d_tmp_scalar_, 0,     sizeof(double));

      kernel_sum_group_pos<<<grid_size, block_size>>>(
        group_size, group_size_sum,
        g_contents,
        g_x, g_y, g_z,
        d_tmp_vec3_, d_tmp_scalar_);
      GPU_CHECK_KERNEL

      double sum_pos[3];
      double count;
      cudaMemcpy(sum_pos,  d_tmp_vec3_,   3 * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(&count,   d_tmp_scalar_, sizeof(double),     cudaMemcpyDeviceToHost);

      if (count <= 0.0) continue;

      double Rcm[3];
      Rcm[0] = sum_pos[0] / count;
      Rcm[1] = sum_pos[1] / count;
      Rcm[2] = sum_pos[2] / count;

      // ghost COM position at current step: Rg = Rg0 + v * step
      double Rg[3];
      Rg[0] = ghost_com_origin_[c][0] + ghost_com_velocity_[c][0] * step_d;
      Rg[1] = ghost_com_origin_[c][1] + ghost_com_velocity_[c][1] * step_d;
      Rg[2] = ghost_com_origin_[c][2] + ghost_com_velocity_[c][2] * step_d;

      double dx = Rcm[0] - Rg[0];
      double dy = Rcm[1] - Rg[1];
      double dz = Rcm[2] - Rg[2];

      const double k  = k_[c];
      const double R0 = R0_[c];

      const double r2 = dx * dx + dy * dy + dz * dz;

      double Fx_cm, Fy_cm, Fz_cm, energy;

      if (R0 == 0.0) {
        // vector spring
        Fx_cm = -k * dx;
        Fy_cm = -k * dy;
        Fz_cm = -k * dz;
        energy = 0.5 * k * r2;
      } else {
        const double r  = sqrt(r2) + SMALL;
        const double dr = r - R0;
        const double coef = -k * dr / r;
        Fx_cm = coef * dx;
        Fy_cm = coef * dy;
        Fz_cm = coef * dz;
        energy = 0.5 * k * dr * dr;
      }

      spring_energy_[c] = energy;

      kernel_add_spring_ghost_com<<<grid_size, block_size>>>(
        group_size, group_size_sum,
        g_contents,
        Fx_cm, Fy_cm, Fz_cm,
        g_fx, g_fy, g_fz);
      GPU_CHECK_KERNEL;

    } else if (mode_[c] == SPRING_GHOST_ATOM) {

      const int gm  = grouping_method1_[c];
      const int gid = group_id1_[c];
      const Group& G = groups[gm];

      const int group_size     = G.cpu_size[gid];
      const int group_size_sum = G.cpu_size_sum[gid];

      if (group_size <= 0) continue;

      const int* g_contents = G.contents.data();

      // allocate & init anchors if needed
      if (!d_ghost_atom_pos_[c] || ghost_atom_group_size_[c] != group_size) {
        if (d_ghost_atom_pos_[c]) {
          cudaFree(d_ghost_atom_pos_[c]);
          d_ghost_atom_pos_[c] = nullptr;
        }

        ghost_atom_group_size_[c] = group_size;
        cudaMalloc((void**)&d_ghost_atom_pos_[c], 3 * group_size * sizeof(double));

        const int grid_init = (group_size - 1) / block_size + 1;
        kernel_init_ghost_atom_pos<<<grid_init, block_size>>>(
          group_size, group_size_sum,
          g_contents,
          g_x, g_y, g_z,
          d_ghost_atom_pos_[c]);
        GPU_CHECK_KERNEL
      }

      cudaMemset(d_tmp_scalar_, 0, sizeof(double));  // use as energy accumulator

      const int grid_size = (group_size - 1) / block_size + 1;

      kernel_add_spring_ghost_atom<<<grid_size, block_size>>>(
        group_size, group_size_sum,
        g_contents,
        d_ghost_atom_pos_[c],
        k_[c],
        R0_[c],
        ghost_atom_velocity_[c][0],
        ghost_atom_velocity_[c][1],
        ghost_atom_velocity_[c][2],
        step_d,
        g_x, g_y, g_z,
        g_fx, g_fy, g_fz,
        d_tmp_scalar_);
      GPU_CHECK_KERNEL

      cudaMemcpy(&spring_energy_[c], d_tmp_scalar_, sizeof(double), cudaMemcpyDeviceToHost);

    } else if (mode_[c] == SPRING_COUPLE_COM) {

      const int gm1  = grouping_method1_[c];
      const int gid1 = group_id1_[c];
      const int gm2  = grouping_method2_[c];
      const int gid2 = group_id2_[c];

      const Group& G1 = groups[gm1];
      const Group& G2 = groups[gm2];

      const int group1_size     = G1.cpu_size[gid1];
      const int group1_size_sum = G1.cpu_size_sum[gid1];

      const int group2_size     = G2.cpu_size[gid2];
      const int group2_size_sum = G2.cpu_size_sum[gid2];

      if (group1_size <= 0 || group2_size <= 0) continue;

      const int* g_contents1 = G1.contents.data();
      const int* g_contents2 = G2.contents.data();

      // COM1
      cudaMemset(d_tmp_vec3_,   0, 3 * sizeof(double));
      cudaMemset(d_tmp_scalar_, 0,     sizeof(double));

      int grid1 = (group1_size - 1) / block_size + 1;
      kernel_sum_group_pos<<<grid1, block_size>>>(
        group1_size, group1_size_sum,
        g_contents1,
        g_x, g_y, g_z,
        d_tmp_vec3_, d_tmp_scalar_);
      GPU_CHECK_KERNEL

      double sum1[3], count1;
      cudaMemcpy(sum1,  d_tmp_vec3_,   3 * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(&count1, d_tmp_scalar_, sizeof(double),   cudaMemcpyDeviceToHost);
      if (count1 <= 0.0) continue;

      double R1[3];
      R1[0] = sum1[0] / count1;
      R1[1] = sum1[1] / count1;
      R1[2] = sum1[2] / count1;

      // COM2
      cudaMemset(d_tmp_vec3_,   0, 3 * sizeof(double));
      cudaMemset(d_tmp_scalar_, 0,     sizeof(double));

      int grid2 = (group2_size - 1) / block_size + 1;
      kernel_sum_group_pos<<<grid2, block_size>>>(
        group2_size, group2_size_sum,
        g_contents2,
        g_x, g_y, g_z,
        d_tmp_vec3_, d_tmp_scalar_);
      GPU_CHECK_KERNEL

      double sum2[3], count2;
      cudaMemcpy(sum2,  d_tmp_vec3_,   3 * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(&count2, d_tmp_scalar_, sizeof(double),   cudaMemcpyDeviceToHost);
      if (count2 <= 0.0) continue;

      double R2[3];
      R2[0] = sum2[0] / count2;
      R2[1] = sum2[1] / count2;
      R2[2] = sum2[2] / count2;

      // displacement from COM1 -> COM2
      double dx = R2[0] - R1[0];
      double dy = R2[1] - R1[1];
      double dz = R2[2] - R1[2];

      const double k  = k_[c];
      const double R0 = R0_[c];

      const double r2 = dx * dx + dy * dy + dz * dz;

      double Fx_cm, Fy_cm, Fz_cm, energy;

      if (R0 == 0.0) {
        // U = 0.5 k r^2, F1 = +k d, F2 = -k d
        Fx_cm = k * dx;
        Fy_cm = k * dy;
        Fz_cm = k * dz;
        energy = 0.5 * k * r2;
      } else {
        const double r  = sqrt(r2) + SMALL;
        const double dr = r - R0;
        const double coef = k * dr / r; // F1 = k (r - R0) d / r
        Fx_cm = coef * dx;
        Fy_cm = coef * dy;
        Fz_cm = coef * dz;
        energy = 0.5 * k * dr * dr;
      }

      spring_energy_[c] = energy;

      const int max_size  = (group1_size > group2_size) ? group1_size : group2_size;
      const int grid_size = (max_size   - 1) / block_size + 1;

      kernel_add_spring_couple_com<<<grid_size, block_size>>>(
        group1_size, group1_size_sum, g_contents1,
        group2_size, group2_size_sum, g_contents2,
        Fx_cm, Fy_cm, Fz_cm,
        g_fx, g_fy, g_fz);
      GPU_CHECK_KERNEL;
    }
  }
}

// ---------------------------------------------------------------------------
// Add_Spring::finalize
// ---------------------------------------------------------------------------

void Add_Spring::finalize()
{
  // free ghost_atom device buffers
  for (int c = 0; c < num_calls_; ++c) {
    if (d_ghost_atom_pos_[c]) {
      cudaFree(d_ghost_atom_pos_[c]);
      d_ghost_atom_pos_[c]      = nullptr;
      ghost_atom_group_size_[c] = 0;
    }
  }

  // free scratch buffers
  if (d_tmp_vec3_) {
    cudaFree(d_tmp_vec3_);
    d_tmp_vec3_ = nullptr;
  }
  if (d_tmp_scalar_) {
    cudaFree(d_tmp_scalar_);
    d_tmp_scalar_ = nullptr;
  }

  num_calls_ = 0;
}
