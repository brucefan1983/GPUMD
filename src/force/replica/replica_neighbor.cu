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
Global neighbor list for replica dynamics, see replica_neighbor.cuh.
------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "replica_neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

namespace
{

// Index of the replica owning an atom: always 0 for a single system.
__host__ __device__ inline int replica_of(const Box&, const int) { return 0; }
__host__ __device__ inline int replica_of(const Batch_Box& box, const int atom)
{
  return atom / box.atoms_per_replica;
}

template <typename Box_View>
__global__ void gpu_check_atom_distance(
  const Box_View box,
  const int N,
  const double d2,
  const double* x0,
  const double* y0,
  const double* z0,
  const double* x,
  const double* y,
  const double* z,
  int* rebuild)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    float dx = x[n] - x0[n];
    float dy = y[n] - y0[n];
    float dz = z[n] - z0[n];
    apply_mic(box_of(box, n), dx, dy, dz);
    if (dx * dx + dy * dy + dz * dz > d2) {
      *rebuild = 1;
    }
  }
}

__global__ void gpu_zero_cells(const int* rebuild, const int size, int* data)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (*rebuild && n < size) {
    data[n] = 0;
  }
}

// Cell (replica r, local cell c) has the global index r * N_cells + c.
template <typename Box_View>
__global__ void gpu_find_cell_counts(
  const int* rebuild,
  const Box_View box,
  const int N,
  const int N_cells,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (*rebuild && n1 < N) {
    int cell_id_x, cell_id_y, cell_id_z, cell_id;
    find_cell_id(
      box_of(box, n1), x[n1], y[n1], z[n1], rc_inv, nx, ny, nz,
      cell_id_x, cell_id_y, cell_id_z, cell_id);
    atomicAdd(&cell_count[replica_of(box, n1) * N_cells + cell_id], 1);
  }
}

template <typename Box_View>
__global__ void gpu_find_cell_contents(
  const int* rebuild,
  const Box_View box,
  const int N,
  const int N_cells,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (*rebuild && n1 < N) {
    int cell_id_x, cell_id_y, cell_id_z, cell_id;
    find_cell_id(
      box_of(box, n1), x[n1], y[n1], z[n1], rc_inv, nx, ny, nz,
      cell_id_x, cell_id_y, cell_id_z, cell_id);
    cell_id += replica_of(box, n1) * N_cells;
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

template <typename Box_View>
__global__ void gpu_find_neighbor_ON1(
  const int* rebuild,
  const Box_View box_view,
  const int N,
  const int N_cells,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const float cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (!*rebuild || n1 >= N) {
    return;
  }
  const Box& box = box_of(box_view, n1);
  const int replica_cells = replica_of(box_view, n1) * N_cells;
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  int z_begin, z_end;
  int y_begin, y_end;
  int x_begin, x_end;
  get_neighbor_cell_offset_range(box.pbc_z, nz, z_begin, z_end);
  get_neighbor_cell_offset_range(box.pbc_y, ny, y_begin, y_end);
  get_neighbor_cell_offset_range(box.pbc_x, nx, x_begin, x_end);

  int count = 0;
  for (int k = z_begin; k < z_end; ++k) {
    for (int j = y_begin; j < y_end; ++j) {
      for (int i = x_begin; i < x_end; ++i) {
        int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
        if (cell_id_x + i < 0)
          neighbor_cell += nx;
        else if (cell_id_x + i >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + j < 0)
          neighbor_cell += ny * nx;
        else if (cell_id_y + j >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + k < 0)
          neighbor_cell += nz * ny * nx;
        else if (cell_id_z + k >= nz)
          neighbor_cell -= nz * ny * nx;
        neighbor_cell += replica_cells;

        const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
        const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = cell_contents[num_atoms_previous_cells + m];
          if (n1 != n2) {
            float x12 = x[n2] - x1;
            float y12 = y[n2] - y1;
            float z12 = z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            const float d2 = x12 * x12 + y12 * y12 + z12 * z12;
            if (d2 < cutoff_square) {
              NL[static_cast<size_t>(N) * count++ + n1] = n2;
            }
          }
        }
      }
    }
  }
  NN[n1] = count;
}

// gpu_sort_neighbor_list of force/neighbor.cuh, skipped when no rebuild is needed.
__global__ void gpu_sort_neighbor_list_if_rebuilt(
  const int* rebuild, const int N, const int* NN, int* NL)
{
  if (!*rebuild) {
    return;
  }
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int neighbor_number = NN[bid];
  extern __shared__ int atom_index_copy[];
  for (int i = tid; i < neighbor_number; i += blockDim.x) {
    atom_index_copy[i] = NL[static_cast<size_t>(N) * i + bid];
  }
  __syncthreads();
  for (int i = tid; i < neighbor_number; i += blockDim.x) {
    const int atom_index = atom_index_copy[i];
    int count = 0;
    for (int j = 0; j < neighbor_number; ++j) {
      if (atom_index > atom_index_copy[j]) {
        count++;
      }
    }
    NL[static_cast<size_t>(N) * count + bid] = atom_index;
  }
}

__global__ void gpu_save_positions(
  const int* rebuild, const int N, const int capacity, const double* position, double* position0)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (*rebuild && n < N) {
    position0[n] = position[n];
    position0[n + capacity] = position[n + N];
    position0[n + capacity * 2] = position[n + N * 2];
  }
}

} // namespace

void Replica_Neighbor::initialize(const double rc, const int num_atoms, const int num_neighbors)
{
  const double rc_plus_skin = rc + skin_;
  const int MN = num_neighbors * rc_plus_skin * rc_plus_skin * rc_plus_skin / (rc * rc * rc);
  NN.resize(num_atoms);
  NL.resize(static_cast<size_t>(num_atoms) * MN);
  cell_count_.resize(num_atoms);
  cell_count_sum_.resize(num_atoms);
  cell_contents_.resize(num_atoms);
  position_at_build_.resize(static_cast<size_t>(num_atoms) * 3);
  rebuild_.resize(1);
  valid_ = false;
}

void Replica_Neighbor::update(
  const double rc,
  const Box& box,
  const int number_of_atoms,
  const double* position_per_atom,
  gpuStream_t stream)
{
  Box cell_box = box;
  update_list(rc, cell_box, cell_box, 1, number_of_atoms, position_per_atom, stream);
}

void Replica_Neighbor::update(
  const double rc,
  const Batch_Box& box,
  const int number_of_atoms,
  const double* position_per_atom,
  gpuStream_t stream)
{
  if (box.atoms_per_replica <= 0 || number_of_atoms % box.atoms_per_replica != 0)
    PRINT_INPUT_ERROR("Invalid batched neighbor-list size.");
  Batch_Box batch_box = box;
  update_list(
    rc,
    batch_box,
    batch_box.box,
    number_of_atoms / box.atoms_per_replica,
    number_of_atoms,
    position_per_atom,
    stream);
}

// cell_box is the (shared) box inside box: get_num_bins() also sets its thickness, which
// find_cell_id reads on the device, so it must be the box that the kernels receive.
template <typename Box_View>
void Replica_Neighbor::update_list(
  const double rc,
  Box_View& box,
  Box& cell_box,
  const int number_of_replicas,
  const int number_of_atoms,
  const double* position_per_atom,
  gpuStream_t stream)
{
  const int N = number_of_atoms;
  const int capacity = NN.size();
  if (N <= 0 || N > capacity)
    PRINT_INPUT_ERROR("The replica neighbor list is smaller than the system.");

  for (int i = 0; i < 9; ++i) {
    if (cell_box.cpu_h[i] != last_box_[i])
      valid_ = false;
    last_box_[i] = cell_box.cpu_h[i];
  }
  const int pbc[3] = {cell_box.pbc_x, cell_box.pbc_y, cell_box.pbc_z};
  for (int i = 0; i < 3; ++i) {
    if (pbc[i] != last_pbc_[i])
      valid_ = false;
    last_pbc_[i] = pbc[i];
  }

  const double rc_neighbor = rc + skin_;
  const double rc_cell_list = 0.5 * rc_neighbor;
  const double rc_inv_cell_list = 2.0 / rc_neighbor;
  int num_bins[3];
  cell_box.get_num_bins(rc_cell_list, num_bins);
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];
  const int total_cells = N_cells * number_of_replicas;

  // The cell count of the last build is reused when no rebuild is needed.
  if (total_cells > cell_count_.size()) {
    CHECK(gpuStreamSynchronize(stream));
    cell_count_.resize(total_cells);
    cell_count_sum_.resize(total_cells);
    valid_ = false;
  }
  if (N != built_number_of_atoms_) {
    valid_ = false;
  }

  const double* x = position_per_atom;
  const double* y = position_per_atom + N;
  const double* z = position_per_atom + N * 2;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  const int cell_grid_size = (total_cells - 1) / block_size + 1;

  if (valid_) {
    rebuild_.fill_async(0, stream);
    gpu_check_atom_distance<<<grid_size, block_size, 0, stream>>>(
      box,
      N,
      skin_ * skin_ * 0.25,
      position_at_build_.data(),
      position_at_build_.data() + capacity,
      position_at_build_.data() + capacity * 2,
      x,
      y,
      z,
      rebuild_.data());
    GPU_CHECK_KERNEL
  } else {
    rebuild_.fill_async(1, stream);
  }

  gpu_zero_cells<<<cell_grid_size, block_size, 0, stream>>>(
    rebuild_.data(), total_cells, cell_count_.data());
  GPU_CHECK_KERNEL

  gpu_find_cell_counts<<<grid_size, block_size, 0, stream>>>(
    rebuild_.data(), box, N, N_cells, cell_count_.data(), x, y, z,
    num_bins[0], num_bins[1], num_bins[2], rc_inv_cell_list);
  GPU_CHECK_KERNEL

  // Without a rebuild the counts are those of the last build, so the sums are unchanged.
  scan_.run(cell_count_.data(), cell_count_sum_.data(), total_cells, stream);

  gpu_zero_cells<<<cell_grid_size, block_size, 0, stream>>>(
    rebuild_.data(), total_cells, cell_count_.data());
  GPU_CHECK_KERNEL

  gpu_find_cell_contents<<<grid_size, block_size, 0, stream>>>(
    rebuild_.data(), box, N, N_cells, cell_count_.data(), cell_count_sum_.data(),
    cell_contents_.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv_cell_list);
  GPU_CHECK_KERNEL

  gpu_find_neighbor_ON1<<<grid_size, block_size, 0, stream>>>(
    rebuild_.data(), box, N, N_cells, cell_count_.data(), cell_count_sum_.data(),
    cell_contents_.data(), NN.data(), NL.data(), x, y, z,
    num_bins[0], num_bins[1], num_bins[2], rc_inv_cell_list, rc_neighbor * rc_neighbor);
  GPU_CHECK_KERNEL

  // the list capacity per atom is fixed by initialize()
  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list_if_rebuilt<<<N, min(1024, MN), MN * sizeof(int), stream>>>(
    rebuild_.data(), N, NN.data(), NL.data());
  GPU_CHECK_KERNEL

  gpu_save_positions<<<grid_size, block_size, 0, stream>>>(
    rebuild_.data(), N, capacity, position_per_atom, position_at_build_.data());
  GPU_CHECK_KERNEL

  valid_ = true;
  built_number_of_atoms_ = N;
}
