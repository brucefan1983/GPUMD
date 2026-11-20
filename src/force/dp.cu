/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The class dealing with the Deep Potential(DP).

Major contributors:
  Ke Xu: Initial DP support and periodic-image force corrections.
  Hekai Bu: DP interface rewrite with ghost-atom handling.
  Ming-Yu Guo: PyTorch models, GPU edge inference, and triclinic-cell support.
  Ming-Yu Guo: Canonical-graph buffers and removal of the unused ghost-atom path.
  Ming-Yu Guo: Reuse DP device buffers across force evaluations.
  Ming-Yu Guo: Share the FP32 and FP64 edge-list kernels.
------------------------------------------------------------------------------*/

#ifdef USE_DEEPMD
#include "dp.cuh"
#include "model/box.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdint>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#define BLOCK_SIZE_FORCE 128
#define MAX_NEIGH_NUM_DP 512 // max neighbor number of an atom for DP

DP::DP(const char* filename_dp, int num_atoms)
{
  // DP setting
  set_dp_coeff();

  // init DP from potential file
  initialize_dp(filename_dp);

  // Global neighbor list at rc + skin (Neighbor owns the cell-list and
  // reference-position buffers); MAX_NEIGH_NUM_DP is the rc-cutoff estimate
  // that Neighbor pads by (rc + skin)^3 / rc^3. The per-step rc-filtered list
  // reuses the same column-major (stride N) layout.
  dp_neighbor.initialize(rc, num_atoms, MAX_NEIGH_NUM_DP);
  dp_NN_local.resize(num_atoms);
  dp_NL_local.resize(dp_neighbor.NL.size());
  type_cpu.resize(num_atoms);
}

void DP::initialize_dp(const char* filename_dp)
{
  int num_gpus;
  CHECK(gpuGetDeviceCount(&num_gpus));
  printf("\nInitialize deep potential by the file: %s and %d gpu(s).\n\n", filename_dp, num_gpus);
  deep_pot.init(filename_dp, 0);
  rc = deep_pot.cutoff();
  int numb_types = deep_pot.numb_types();
  int numb_types_spin = deep_pot.numb_types_spin();
  int dim_fparam = deep_pot.dim_fparam();
  int dim_aparam = deep_pot.dim_aparam();

  std::string type_map_str;
  deep_pot.get_type_map(type_map_str);

  printf("---------------------------------------------------------------\n");
  printf("  ++ cutoff: %f ++ \n", rc);
  printf("  ++ numb_types: %d ++ \n", numb_types);
  printf("  ++ numb_types_spin: %d ++ \n", numb_types_spin);
  printf("  ++ dim_fparam: %d ++ \n", dim_fparam);
  printf("  ++ dim_aparam: %d ++ \n", dim_aparam);
  printf("  ++ %s ++\n", type_map_str.c_str());
  printf("---------------------------------------------------------------\n");
}

DP::~DP(void)
{
  // none
}

void DP::set_dp_coeff(void)
{
  ener_unit_cvt_factor = 1; // 1.0 / 8.617343e-5;
  dist_unit_cvt_factor = 1; // gpumd: angstrom, dp: angstrom;
  force_unit_cvt_factor = ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor = 1; // ener_unit_cvt_factor
}

namespace
{
template <typename T>
void grow_gpu(GPU_Vector<T>& values, const size_t count)
{
  if (values.size() < count) {
    values.resize(count);
  }
}

static __global__ void dp_position_transpose(const double* position, double* position_trans, int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    position_trans[n1 * 3] = position[n1];
    position_trans[n1 * 3 + 1] = position[n1 + N];
    position_trans[n1 * 3 + 2] = position[n1 + 2 * N];
  }
}

// Simplified transpose kernel for fully-periodic path (no ghost atoms to fold back)
static __global__ void transpose_and_update_unit_no_ghost(
  const double* e_f_v_in,
  double* e_out,
  double* f_out,
  double* v_out,
  double e_factor,
  double f_factor,
  double v_factor,
  const int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int f_in_offset = N;
    const int v_in_offset = N * 4;
    e_out[n1] = e_f_v_in[n1] * e_factor;

    double fx = e_f_v_in[f_in_offset + n1 * 3];
    double fy = e_f_v_in[f_in_offset + n1 * 3 + 1];
    double fz = e_f_v_in[f_in_offset + n1 * 3 + 2];

    f_out[n1] = fx * f_factor;
    f_out[n1 + N] = fy * f_factor;
    f_out[n1 + N * 2] = fz * f_factor;

    // virial layout from DeePMD atomic virial (same convention as old path):
    // Index mapping: xx=0, yy=4, zz=8, xy=3, xz=6, yz=7, yx=1, zx=2, zy=5
    double vxx = e_f_v_in[v_in_offset + n1 * 9] * v_factor;
    double vyy = e_f_v_in[v_in_offset + n1 * 9 + 4] * v_factor;
    double vzz = e_f_v_in[v_in_offset + n1 * 9 + 8] * v_factor;
    double vxy = e_f_v_in[v_in_offset + n1 * 9 + 3] * v_factor;
    double vxz = e_f_v_in[v_in_offset + n1 * 9 + 6] * v_factor;
    double vyz = e_f_v_in[v_in_offset + n1 * 9 + 7] * v_factor;
    double vyx = e_f_v_in[v_in_offset + n1 * 9 + 1] * v_factor;
    double vzx = e_f_v_in[v_in_offset + n1 * 9 + 2] * v_factor;
    double vzy = e_f_v_in[v_in_offset + n1 * 9 + 5] * v_factor;

    // GPUMD virial_per_atom layout (SoA): xx, yy, zz, xy, xz, yz, yx, zx, zy
    v_out[n1] = vxx;
    v_out[n1 + N] = vyy;
    v_out[n1 + N * 2] = vzz;
    v_out[n1 + N * 3] = vxy;
    v_out[n1 + N * 4] = vxz;
    v_out[n1 + N * 5] = vyz;
    v_out[n1 + N * 6] = vyx;
    v_out[n1 + N * 7] = vzx;
    v_out[n1 + N * 8] = vzy;
  }
}

// force and virial need transpose from dp to gpumd
// Periodic cell grids with fewer cells than the five-cell search stencil can
// visit the same wrapped cell more than once.  The global-list sorter ranks
// equal atom indices at the same position, so such duplicates are not
// guaranteed to remain adjacent: colliding writes can leave older valid
// neighbor entries in the intervening slots.  Compact the whole DP-local list
// by atom index before turning it into graph edges.  The GPU-edge path is only
// used when each periodic box thickness exceeds twice the neighbor cutoff, so
// distinct periodic images of the same atom cannot be valid separate edges.
static __global__ void dp_compact_duplicate_neighbors(const int N, const int nloc, int* NN, int* NL)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < nloc) {
    const int count = NN[n1];
    int count_unique = 0;
    for (int c = 0; c < count; ++c) {
      const int n2 = NL[static_cast<size_t>(c) * N + n1];
      bool duplicate = false;
      for (int u = 0; u < count_unique; ++u) {
        if (NL[static_cast<size_t>(u) * N + n1] == n2) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        NL[static_cast<size_t>(count_unique++) * N + n1] = n2;
      }
    }
    NN[n1] = count_unique;
  }
}

// Materialize the compact edge schema from GPUMD's neighbor list (NN, NL).
// For each local atom n1 and each of its NN[n1] neighbors n2 (a local index),
// emit one edge (src = n2, dst = n1) with the minimum-image bond vector
// r(n2) - r(n1).  ``edge_index`` is the flattened [2, nedge] graph:
// [0, nedge) holds the source row, [nedge, 2 * nedge) the destination row.
// Displacement stays in double precision and is converted when it is stored.
template <typename Vec>
static __global__ void dp_fill_edges(
  const int N,
  const int nloc,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const int* __restrict__ edge_offset,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const Box box,
  const int nedge,
  int* edge_index,
  Vec* edge_vec)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < nloc) {
    const int count = NN[n1];
    const int base = edge_offset[n1];
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    for (int c = 0; c < count; ++c) {
      const int n2 = NL[c * N + n1];
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double z12 = z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      const int e = base + c;
      edge_index[e] = n2;
      edge_index[nedge + e] = n1;
      edge_vec[e * 3] = static_cast<Vec>(x12);
      edge_vec[e * 3 + 1] = static_cast<Vec>(y12);
      edge_vec[e * 3 + 2] = static_cast<Vec>(z12);
    }
  }
}

// Build the compact canonical graph ABI used by compressed DPA4C artifacts.
// The neighbor list is already destination-major (one contiguous row per
// center), so its exclusive offsets are the destination CSR.  This kernel
// writes the uint32 source row, float32 bond vectors, and per-source counts.
static __global__ void dp_fill_canonical_edges(
  const int N,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const int* __restrict__ edge_offset,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const Box box,
  int* source,
  float* edge_vec,
  int* source_count)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int count = NN[n1];
    const int base = edge_offset[n1];
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    for (int c = 0; c < count; ++c) {
      const int n2 = NL[static_cast<size_t>(c) * N + n1];
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double z12 = z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      const int edge = base + c;
      source[edge] = n2;
      edge_vec[edge * 3] = static_cast<float>(x12);
      edge_vec[edge * 3 + 1] = static_cast<float>(y12);
      edge_vec[edge * 3 + 2] = static_cast<float>(z12);
      atomicAdd(source_count + n2, 1);
    }
  }
}

static __global__ void dp_prepare_canonical_nodes(
  const int N,
  const int nedge,
  const int* __restrict__ type,
  const int* __restrict__ destination_offset,
  std::int64_t* model_type,
  std::int64_t* destination_row_ptr)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    model_type[n1] = static_cast<std::int64_t>(type[n1]);
    destination_row_ptr[n1] = static_cast<std::int64_t>(destination_offset[n1]);
  }
  if (n1 == N - 1) {
    destination_row_ptr[N] = static_cast<std::int64_t>(nedge);
  }
}

static __global__ void dp_prepare_canonical_source_rows(
  const int N,
  const int nedge,
  const int* __restrict__ source_offset,
  std::int64_t* source_row_ptr,
  int* source_cursor)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int offset = source_offset[n1];
    source_row_ptr[n1] = static_cast<std::int64_t>(offset);
    source_cursor[n1] = offset;
  }
  if (n1 == N - 1) {
    source_row_ptr[N] = static_cast<std::int64_t>(nedge);
  }
}

static __global__ void dp_build_canonical_source_order(
  const int nedge, const int* __restrict__ source, int* source_cursor, std::int64_t* source_order)
{
  const int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge < nedge) {
    const int position = atomicAdd(source_cursor + source[edge], 1);
    source_order[position] = static_cast<std::int64_t>(edge);
  }
}

static __global__ void dp_fill_canonical_guards(
  const int edge_begin,
  const int edge_storage,
  int* source,
  float* edge_vec,
  std::int64_t* source_order)
{
  const int edge = edge_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (edge < edge_storage) {
    source[edge] = 0;
    edge_vec[edge * 3] = 0.0f;
    edge_vec[edge * 3 + 1] = 0.0f;
    edge_vec[edge * 3 + 2] = 0.0f;
    source_order[edge] = static_cast<std::int64_t>(edge);
  }
}

// Scatter the device model outputs into GPUMD's per-atom arrays.  Force and
// virial are converted from the model's row-major layout to GPUMD's
// structure-of-arrays layout, applying the unit-conversion factors.  The
// virial index map mirrors the standalone path: model column-major
// (xx, yx, zx, xy, yy, zy, xz, yz, zz) -> GPUMD (xx, yy, zz, xy, xz, yz, yx,
// zx, zy).
static __global__ void dp_scatter_outputs(
  const int nloc,
  const double* __restrict__ atom_energy,
  const double* __restrict__ force_rm,
  const double* __restrict__ atom_virial,
  const double e_factor,
  const double f_factor,
  const double v_factor,
  double* potential,
  double* force,
  double* virial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < nloc) {
    potential[n1] = atom_energy[n1] * e_factor;

    force[n1] = force_rm[n1 * 3] * f_factor;
    force[n1 + nloc] = force_rm[n1 * 3 + 1] * f_factor;
    force[n1 + nloc * 2] = force_rm[n1 * 3 + 2] * f_factor;

    const double* v = atom_virial + n1 * 9;
    virial[n1] = v[0] * v_factor;
    virial[n1 + nloc] = v[4] * v_factor;
    virial[n1 + nloc * 2] = v[8] * v_factor;
    virial[n1 + nloc * 3] = v[3] * v_factor;
    virial[n1 + nloc * 4] = v[6] * v_factor;
    virial[n1 + nloc * 5] = v[7] * v_factor;
    virial[n1 + nloc * 6] = v[1] * v_factor;
    virial[n1 + nloc * 7] = v[2] * v_factor;
    virial[n1 + nloc * 8] = v[5] * v_factor;
  }
}

} // namespace

void DP::compute_gpu_edges(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = type.size();
  const int grid_size = (N - 1) / BLOCK_SIZE_FORCE + 1;

  // The global list is (re)built at rc + skin only when an atom has drifted
  // more than skin/2, then filtered to the true cutoff rc for this step's
  // edges. The rebuild is amortized over many steps instead of running every
  // step.
  dp_neighbor.find_neighbor_global(rc, box, type, position_per_atom);
  dp_neighbor.find_local_neighbor_from_global(rc, box, position_per_atom, dp_NN_local, dp_NL_local);

  // find_neighbor_global uses cells of width (rc + skin) / 2 and a five-cell
  // stencil.  Wrapped cells can alias only when a periodic direction has
  // fewer than five cells; avoid the quadratic compaction cost for the common
  // non-aliasing case.  The 1.0 A skin must match Neighbor::skin.
  int num_bins[3];
  box.get_num_bins(0.5 * (rc + 1.0), num_bins);
  const bool wrapped_cells_alias = (box.pbc_x && num_bins[0] < 5) ||
                                   (box.pbc_y && num_bins[1] < 5) || (box.pbc_z && num_bins[2] < 5);
  if (wrapped_cells_alias) {
    dp_compact_duplicate_neighbors<<<grid_size, BLOCK_SIZE_FORCE>>>(
      N, N, dp_NN_local.data(), dp_NL_local.data());
    GPU_CHECK_KERNEL
  }

  // Exclusive scan of the per-atom neighbor counts gives each atom's edge
  // offset; the total edge count is the reduction of the counts.
  grow_gpu(dp_edge_offset, static_cast<size_t>(N));
  thrust::exclusive_scan(
    thrust::device, dp_NN_local.data(), dp_NN_local.data() + N, dp_edge_offset.data());
  const int nedge = thrust::reduce(
    thrust::device, dp_NN_local.data(), dp_NN_local.data() + N, 0, thrust::plus<int>());

  grow_gpu(dp_atom_energy_gpu, static_cast<size_t>(N));
  grow_gpu(dp_force_rowmajor, static_cast<size_t>(N) * 3);
  grow_gpu(dp_atom_virial_gpu, static_cast<size_t>(N) * 9);

  if (deep_pot.uses_canonical_graph_inference()) {
    // Compact canonical ABI (compressed DPA1 / DPA4C and the same family).
    // Single rank, no domain halo: nall == nloc and the halo exchange is null.
    // Source indices stay 32-bit because the neighbor list itself is int.
    // Source CSR positions are int64. Scratch buffers are reused by width:
    //   dp_position_gpu_trans  int64 atom type
    //   dp_position_gpu        int64 destination row pointer
    //   e_f_v_gpu              int64 source row pointer
    //   dp_edge_index          uint32 source index
    //   dp_edge_vec            float32 bond vector
    //   source_order           int64 source order
    //   canonical_coord        float64 coordinates, only if the model has a
    //                          long-range module
    const int edge_storage = nedge < 2 ? 2 : nedge;
    if (
      deepmd_compat::canonical_index_width(static_cast<std::int64_t>(N)) != DP_CANONICAL_INDEX32) {
      PRINT_INPUT_ERROR("DP canonical graph source indices do not fit in 32 bits.");
    }
    grow_gpu(dp_edge_index, static_cast<size_t>(edge_storage));
    grow_gpu(dp_edge_vec, static_cast<size_t>(3 * edge_storage));
    grow_gpu(dp_position_gpu_trans, static_cast<size_t>(N));
    grow_gpu(dp_position_gpu, static_cast<size_t>(N + 1));
    grow_gpu(e_f_v_gpu, static_cast<size_t>(N + 1));
    grow_gpu(source_order, static_cast<size_t>(edge_storage));
    grow_gpu(source_count, static_cast<size_t>(N));
    grow_gpu(source_cursor, static_cast<size_t>(N));

    auto* model_type = reinterpret_cast<std::int64_t*>(dp_position_gpu_trans.data());
    auto* destination_row_ptr = reinterpret_cast<std::int64_t*>(dp_position_gpu.data());
    auto* source_row_ptr = reinterpret_cast<std::int64_t*>(e_f_v_gpu.data());
    int* source = dp_edge_index.data();
    float* edge_vec = reinterpret_cast<float*>(dp_edge_vec.data());

    dp_prepare_canonical_nodes<<<grid_size, BLOCK_SIZE_FORCE>>>(
      N, nedge, type.data(), dp_edge_offset.data(), model_type, destination_row_ptr);
    GPU_CHECK_KERNEL
    CHECK(gpuMemset(source_count.data(), 0, sizeof(int) * N));
    if (nedge > 0) {
      dp_fill_canonical_edges<<<grid_size, BLOCK_SIZE_FORCE>>>(
        N,
        dp_NN_local.data(),
        dp_NL_local.data(),
        dp_edge_offset.data(),
        position_per_atom.data(),
        position_per_atom.data() + N,
        position_per_atom.data() + N * 2,
        box,
        source,
        edge_vec,
        source_count.data());
      GPU_CHECK_KERNEL
    }
    thrust::exclusive_scan(
      thrust::device, source_count.data(), source_count.data() + N, dp_edge_offset.data());
    dp_prepare_canonical_source_rows<<<grid_size, BLOCK_SIZE_FORCE>>>(
      N, nedge, dp_edge_offset.data(), source_row_ptr, source_cursor.data());
    GPU_CHECK_KERNEL
    if (nedge > 0) {
      const int edge_grid = (nedge - 1) / BLOCK_SIZE_FORCE + 1;
      dp_build_canonical_source_order<<<edge_grid, BLOCK_SIZE_FORCE>>>(
        nedge, source, source_cursor.data(), source_order.data());
      GPU_CHECK_KERNEL
    }
    if (edge_storage > nedge) {
      const int guard_grid = (edge_storage - nedge - 1) / BLOCK_SIZE_FORCE + 1;
      dp_fill_canonical_guards<<<guard_grid, BLOCK_SIZE_FORCE>>>(
        nedge, edge_storage, source, edge_vec, source_order.data());
      GPU_CHECK_KERNEL
    }

    // A long-range module reads node positions and, for a fully periodic
    // cell, the lattice with vectors as rows. Every other model ignores them.
    double* d_coord = nullptr;
    double host_box[9];
    const double* box_ptr = nullptr;
    if (deep_pot.has_long_range()) {
      grow_gpu(canonical_coord, static_cast<size_t>(N) * 3);
      dp_position_transpose<<<grid_size, BLOCK_SIZE_FORCE>>>(
        position_per_atom.data(), canonical_coord.data(), N);
      GPU_CHECK_KERNEL
      d_coord = canonical_coord.data();
      if (box.pbc_x && box.pbc_y && box.pbc_z) {
        host_box[0] = box.cpu_h[0];
        host_box[1] = box.cpu_h[3];
        host_box[2] = box.cpu_h[6];
        host_box[3] = box.cpu_h[1];
        host_box[4] = box.cpu_h[4];
        host_box[5] = box.cpu_h[7];
        host_box[6] = box.cpu_h[2];
        host_box[7] = box.cpu_h[5];
        host_box[8] = box.cpu_h[8];
        box_ptr = host_box;
      }
    }

    // Per-atom virial is what GPUMD reduces into the stress, so the source
    // CSR is passed whenever the artifact can consume it. The total virial
    // is left null; GPUMD sums the atomic contribution itself.
    deep_pot.compute_canonical_graph_gpu(
      dp_atom_energy_gpu.data(),
      dp_force_rowmajor.data(),
      nullptr,
      dp_atom_virial_gpu.data(),
      model_type,
      reinterpret_cast<const std::uint32_t*>(source),
      DP_CANONICAL_INDEX32,
      edge_vec,
      destination_row_ptr,
      source_row_ptr,
      source_order.data(),
      d_coord,
      nullptr,
      nullptr,
      box_ptr,
      static_cast<std::int64_t>(N),
      static_cast<std::int64_t>(N),
      static_cast<std::int64_t>(edge_storage),
      nullptr);
  } else {
    // Transpose positions from GPUMD SoA (x1..xN, y1..yN, z1..zN) to the
    // row-major (x1, y1, z1, ...) layout the edge model consumes.
    grow_gpu(dp_position_gpu_trans, static_cast<size_t>(N) * 3);
    dp_position_transpose<<<grid_size, BLOCK_SIZE_FORCE>>>(
      position_per_atom.data(), dp_position_gpu_trans.data(), N);
    GPU_CHECK_KERNEL
    const bool edge_fp32 = deep_pot.uses_fp32_edge_vectors();
    if (nedge > 0) {
      grow_gpu(dp_edge_index, static_cast<size_t>(2 * nedge));
      grow_gpu(dp_edge_vec, static_cast<size_t>(3 * nedge));
      if (edge_fp32) {
        dp_fill_edges<float><<<grid_size, BLOCK_SIZE_FORCE>>>(
          N,
          N,
          dp_NN_local.data(),
          dp_NL_local.data(),
          dp_edge_offset.data(),
          position_per_atom.data(),
          position_per_atom.data() + N,
          position_per_atom.data() + N * 2,
          box,
          nedge,
          dp_edge_index.data(),
          reinterpret_cast<float*>(dp_edge_vec.data()));
      } else {
        dp_fill_edges<double><<<grid_size, BLOCK_SIZE_FORCE>>>(
          N,
          N,
          dp_NN_local.data(),
          dp_NL_local.data(),
          dp_edge_offset.data(),
          position_per_atom.data(),
          position_per_atom.data() + N,
          position_per_atom.data() + N * 2,
          box,
          nedge,
          dp_edge_index.data(),
          dp_edge_vec.data());
      }
      GPU_CHECK_KERNEL
    }
    if (edge_fp32) {
      const std::vector<double> no_param;
      deep_pot.compute_edges_gpu(
        dp_atom_energy_gpu.data(),
        dp_force_rowmajor.data(),
        dp_atom_virial_gpu.data(),
        dp_position_gpu_trans.data(),
        type.data(),
        dp_edge_index.data(),
        nedge > 0 ? reinterpret_cast<const float*>(dp_edge_vec.data()) : nullptr,
        N,
        nedge,
        no_param,
        no_param);
    } else {
      deep_pot.compute_edges_gpu(
        dp_atom_energy_gpu.data(),
        dp_force_rowmajor.data(),
        dp_atom_virial_gpu.data(),
        dp_position_gpu_trans.data(),
        type.data(),
        dp_edge_index.data(),
        dp_edge_vec.data(),
        N,
        nedge);
    }
  }

  // Scatter the device outputs into GPUMD's per-atom arrays.
  dp_scatter_outputs<<<grid_size, BLOCK_SIZE_FORCE>>>(
    N,
    dp_atom_energy_gpu.data(),
    dp_force_rowmajor.data(),
    dp_atom_virial_gpu.data(),
    ener_unit_cvt_factor,
    force_unit_cvt_factor,
    virial_unit_cvt_factor,
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}

void DP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  if (number_of_atoms <= 0)
    return;
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE_FORCE + 1;

  // Build the neighbor list and edge schema on the GPU and run the exported
  // model on device tensors (no host neighbor-list build, no per-step
  // host-device transfers).  This path resolves neighbors by the minimum-image
  // convention, which is valid only when the cutoff is smaller than half the
  // box thickness along every periodic direction.  Smaller cells
  // (cutoff >= L/2) use the host fallback below.  Box thicknesses are computed
  // directly because Box::thickness_* is only populated as a side effect of
  // get_num_bins, which has not run yet at this point.
  {
    // The reused global list spans rc + skin, so the minimum-image build is
    // only unambiguous when every periodic thickness exceeds 2 * (rc + skin);
    // thinner cells use the host fallback below. The skin here
    // must match Neighbor::skin (1.0 A).
    const double neighbor_rc = rc + 1.0;
    const double volume = box.get_volume();
    const double thickness_x = volume / box.get_area(0);
    const double thickness_y = volume / box.get_area(1);
    const double thickness_z = volume / box.get_area(2);
    const bool mic_valid = (box.pbc_x == 0 || thickness_x > 2.0 * neighbor_rc) &&
                           (box.pbc_y == 0 || thickness_y > 2.0 * neighbor_rc) &&
                           (box.pbc_z == 0 || thickness_z > 2.0 * neighbor_rc);
    if (mic_valid && deep_pot.supports_device_edge_inference()) {
      compute_gpu_edges(
        box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
      return;
    }
  }

  // Always use DeePMD's internal neighbor list construction (bypass path).
  // This avoids ghost atom construction which causes force inconsistency
  // for message-passing networks on both bulk and slab systems.
  //
  // For non-periodic directions, we inflate the box and center atoms so that
  // DeePMD's internal PBC won't create spurious periodic images in those directions.
  // DeePMD's C++ API only supports box=all-zero (cluster) or box=non-zero (full PBC).
  // We choose the latter and ensure non-periodic directions have sufficient vacuum.
  {
    // Transpose positions from GPUMD layout (x1..xN, y1..yN, z1..zN) to
    // row-major (x1,y1,z1, x2,y2,z2, ...)
    grow_gpu(dp_position_gpu_trans, static_cast<size_t>(number_of_atoms) * 3);
    dp_position_transpose<<<grid_size, BLOCK_SIZE_FORCE>>>(
      position_per_atom.data(), dp_position_gpu_trans.data(), number_of_atoms);
    GPU_CHECK_KERNEL
    dp_position_cpu.resize(static_cast<size_t>(number_of_atoms) * 3);
    CHECK(gpuMemcpy(
      dp_position_cpu.data(),
      dp_position_gpu_trans.data(),
      sizeof(double) * static_cast<size_t>(number_of_atoms) * 3,
      gpuMemcpyDeviceToHost));

    // Copy types to CPU (type is const, so use gpuMemcpy directly)
    type_cpu.resize(number_of_atoms);
    CHECK(gpuMemcpy(
      type_cpu.data(), type.data(), sizeof(int) * number_of_atoms, gpuMemcpyDeviceToHost));

    // Build the box for DeePMD, handling non-periodic directions.
    // For non-periodic directions, inflate box vectors so that the effective
    // thickness in that direction >= atom_extent + 4*rc, preventing DeePMD
    // from finding periodic-image neighbors across the free boundary.
    // We also shift atoms to be centered in the inflated box.
    //
    // Box matrix layout in GPUMD (column-major lattice vectors):
    //   a = (cpu_h[0], cpu_h[3], cpu_h[6])  -- 1st lattice vector
    //   b = (cpu_h[1], cpu_h[4], cpu_h[7])  -- 2nd lattice vector
    //   c = (cpu_h[2], cpu_h[5], cpu_h[8])  -- 3rd lattice vector
    // Thickness of direction d = Volume / Area(d).

    double dp_h[9]; // local copy of box matrix for DeePMD
    for (int i = 0; i < 9; ++i)
      dp_h[i] = box.cpu_h[i];

    if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
      if (box.is_orthogonal) {
        // Orthogonal box: use simple Cartesian shift approach.
        // pbc_x/y/z directly maps to Cartesian x/y/z axes for orthogonal boxes.
        double xmin = 1e30, xmax = -1e30;
        double ymin = 1e30, ymax = -1e30;
        double zmin = 1e30, zmax = -1e30;
        for (int i = 0; i < number_of_atoms; ++i) {
          double x = dp_position_cpu[i * 3];
          double y = dp_position_cpu[i * 3 + 1];
          double z = dp_position_cpu[i * 3 + 2];
          if (x < xmin)
            xmin = x;
          if (x > xmax)
            xmax = x;
          if (y < ymin)
            ymin = y;
          if (y > ymax)
            ymax = y;
          if (z < zmin)
            zmin = z;
          if (z > zmax)
            zmax = z;
        }

        double shift_x = 0.0, shift_y = 0.0, shift_z = 0.0;

        if (box.pbc_x == 0) {
          double extent = xmax - xmin;
          double needed = extent + 4.0 * rc;
          if (needed > dp_h[0]) {
            double scale = needed / dp_h[0];
            dp_h[0] *= scale;
          }
          shift_x = dp_h[0] * 0.5 - (xmin + xmax) * 0.5;
        }
        if (box.pbc_y == 0) {
          double extent = ymax - ymin;
          double needed = extent + 4.0 * rc;
          if (needed > dp_h[4]) {
            double scale = needed / dp_h[4];
            dp_h[4] *= scale;
          }
          shift_y = dp_h[4] * 0.5 - (ymin + ymax) * 0.5;
        }
        if (box.pbc_z == 0) {
          double extent = zmax - zmin;
          double needed = extent + 4.0 * rc;
          if (needed > dp_h[8]) {
            double scale = needed / dp_h[8];
            dp_h[8] *= scale;
          }
          shift_z = dp_h[8] * 0.5 - (zmin + zmax) * 0.5;
        }

        // Apply shifts only to non-periodic directions
        for (int i = 0; i < number_of_atoms; ++i) {
          dp_position_cpu[i * 3] += shift_x;
          dp_position_cpu[i * 3 + 1] += shift_y;
          dp_position_cpu[i * 3 + 2] += shift_z;
        }
      } else {
        // Triclinic box: use fractional coordinate approach.
        // pbc_x/y/z maps to lattice vector a/b/c directions, not Cartesian axes.
        // We must work in fractional space to correctly inflate the right direction.
        for (int i = 0; i < number_of_atoms; ++i) {
          double x = dp_position_cpu[i * 3];
          double y = dp_position_cpu[i * 3 + 1];
          double z = dp_position_cpu[i * 3 + 2];
          dp_position_cpu[i * 3] = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
          dp_position_cpu[i * 3 + 1] = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
          dp_position_cpu[i * 3 + 2] = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
        }

        double smin[3] = {1e30, 1e30, 1e30};
        double smax[3] = {-1e30, -1e30, -1e30};
        for (int i = 0; i < number_of_atoms; ++i) {
          for (int d = 0; d < 3; ++d) {
            double s = dp_position_cpu[i * 3 + d];
            if (s < smin[d])
              smin[d] = s;
            if (s > smax[d])
              smax[d] = s;
          }
        }

        auto cross_norm = [](double u0, double u1, double u2, double v0, double v1, double v2) {
          double c0 = u1 * v2 - u2 * v1;
          double c1 = u2 * v0 - u0 * v2;
          double c2 = u0 * v1 - u1 * v0;
          return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
        };
        double volume = fabs(
          dp_h[0] * (dp_h[4] * dp_h[8] - dp_h[5] * dp_h[7]) +
          dp_h[1] * (dp_h[5] * dp_h[6] - dp_h[3] * dp_h[8]) +
          dp_h[2] * (dp_h[3] * dp_h[7] - dp_h[4] * dp_h[6]));
        double area_x = cross_norm(dp_h[1], dp_h[4], dp_h[7], dp_h[2], dp_h[5], dp_h[8]);
        double area_y = cross_norm(dp_h[2], dp_h[5], dp_h[8], dp_h[0], dp_h[3], dp_h[6]);
        double area_z = cross_norm(dp_h[0], dp_h[3], dp_h[6], dp_h[1], dp_h[4], dp_h[7]);

        int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
        double thickness[3] = {volume / area_x, volume / area_y, volume / area_z};

        for (int d = 0; d < 3; ++d) {
          if (pbc[d] == 0) {
            double frac_extent = smax[d] - smin[d];
            double cart_extent = frac_extent * thickness[d];
            double needed = cart_extent + 4.0 * rc;
            if (needed > thickness[d]) {
              double scale = needed / thickness[d];
              dp_h[d] *= scale;
              dp_h[d + 3] *= scale;
              dp_h[d + 6] *= scale;
              thickness[d] = needed;
              for (int i = 0; i < number_of_atoms; ++i) {
                dp_position_cpu[i * 3 + d] /= scale;
              }
              smin[d] /= scale;
              smax[d] /= scale;
            }
            double frac_center = (smin[d] + smax[d]) * 0.5;
            double shift = 0.5 - frac_center;
            for (int i = 0; i < number_of_atoms; ++i) {
              dp_position_cpu[i * 3 + d] += shift;
            }
          }
        }

        // Convert back to Cartesian using (possibly inflated) dp_h
        for (int i = 0; i < number_of_atoms; ++i) {
          double sx = dp_position_cpu[i * 3];
          double sy = dp_position_cpu[i * 3 + 1];
          double sz = dp_position_cpu[i * 3 + 2];
          dp_position_cpu[i * 3] = dp_h[0] * sx + dp_h[1] * sy + dp_h[2] * sz;
          dp_position_cpu[i * 3 + 1] = dp_h[3] * sx + dp_h[4] * sy + dp_h[5] * sz;
          dp_position_cpu[i * 3 + 2] = dp_h[6] * sx + dp_h[7] * sy + dp_h[8] * sz;
        }
      }
    }

    // Construct dp_box in DeePMD row-major format from (possibly inflated) dp_h
    // DeePMD box layout: row0=a, row1=b, row2=c (each as x,y,z)
    if (fallback_box.size() != 9) {
      fallback_box.assign(9, 0.0);
    }
    fallback_box[0] = dp_h[0];
    fallback_box[1] = dp_h[3];
    fallback_box[2] = dp_h[6];
    fallback_box[3] = dp_h[1];
    fallback_box[4] = dp_h[4];
    fallback_box[5] = dp_h[7];
    fallback_box[6] = dp_h[2];
    fallback_box[7] = dp_h[5];
    fallback_box[8] = dp_h[8];

    // Allocate output buffers
    dp_ene_all.resize(1, 0.0);
    dp_ene_atom.resize(number_of_atoms, 0.0);
    dp_force.resize(number_of_atoms * 3, 0.0);
    dp_vir_all.resize(9, 0.0);
    dp_vir_atom.resize(number_of_atoms * 9, 0.0);

    // Call DeePMD compute WITHOUT neighbor list — DeePMD handles PBC internally
    deep_pot.compute(
      dp_ene_all,
      dp_force,
      dp_vir_all,
      dp_ene_atom,
      dp_vir_atom,
      dp_position_cpu,
      type_cpu,
      fallback_box);

    // Copy results to GPU. The 13*N packed buffer is allocated on first use.
    // Memory layout of e_f_v_gpu: e1..eN, fx1,fy1,fz1,...fxN,fyN,fzN, vxx1...
    grow_gpu(e_f_v_gpu, static_cast<size_t>(number_of_atoms) * (1 + 3 + 9));
    e_f_v_gpu.copy_from_host(dp_ene_atom.data(), number_of_atoms, 0);
    e_f_v_gpu.copy_from_host(dp_force.data(), number_of_atoms * 3, number_of_atoms);
    e_f_v_gpu.copy_from_host(dp_vir_atom.data(), number_of_atoms * 9, number_of_atoms * 4);

    // Transpose forces/virials from DeePMD layout to GPUMD layout (no ghost folding)
    transpose_and_update_unit_no_ghost<<<grid_size, BLOCK_SIZE_FORCE>>>(
      e_f_v_gpu.data(),
      potential_per_atom.data(),
      force_per_atom.data(),
      virial_per_atom.data(),
      ener_unit_cvt_factor,
      force_unit_cvt_factor,
      virial_unit_cvt_factor,
      number_of_atoms);
    GPU_CHECK_KERNEL

    return;
  }
}
#endif
