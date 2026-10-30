/* Single-process spatial decomposition for NEP-Spin. LGPL-3.0-or-later. */
#include "model/box.cuh"
#include "nep_spin_multigpu.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <future>
#include <limits>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace
{
static __device__ void find_cell_id(
  const int partition_direction,
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  cell_id_x = floor(sx * box.thickness_x * rc_inv);
  cell_id_y = floor(sy * box.thickness_y * rc_inv);
  cell_id_z = floor(sz * box.thickness_z * rc_inv);

  while (cell_id_x < 0)
    cell_id_x += nx;
  while (cell_id_x >= nx)
    cell_id_x -= nx;
  while (cell_id_y < 0)
    cell_id_y += ny;
  while (cell_id_y >= ny)
    cell_id_y -= ny;
  while (cell_id_z < 0)
    cell_id_z += nz;
  while (cell_id_z >= nz)
    cell_id_z -= nz;
  if (partition_direction == 0) {
    cell_id = cell_id_y + ny * (cell_id_z + nz * cell_id_x);
  } else if (partition_direction == 1) {
    cell_id = cell_id_x + nx * (cell_id_z + nz * cell_id_y);
  } else {
    cell_id = cell_id_x + nx * (cell_id_y + ny * cell_id_z);
  }
}

static __device__ void find_cell_id(
  const int partition_direction,
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(
    partition_direction,
    box,
    x,
    y,
    z,
    rc_inv,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);
}

static __global__ void find_cell_counts(
  const int partition_direction,
  const Box box,
  const int N,
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
  if (n1 < N) {
    int cell_id;
    find_cell_id(partition_direction, box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents(
  const int partition_direction,
  const Box box,
  const int N,
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
  if (n1 < N) {
    int cell_id;
    find_cell_id(partition_direction, box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

static void __global__ set_to_zero(int size, int* data)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    data[n] = 0;
  }
}

static void find_cell_list(
  gpuStream_t& stream,
  const int partition_direction,
  const double rc,
  const int* num_bins,
  Box& box,
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int offset = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + offset;
  const double* z = position_per_atom.data() + offset * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_count_sum.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count_sum.size(), cell_count_sum.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_contents.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_contents.size(), cell_contents.data());
  GPU_CHECK_KERNEL

  find_cell_counts<<<grid_size, block_size, 0, stream>>>(
    partition_direction,
    box,
    N,
    cell_count.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
#ifdef USE_HIP
    thrust::hip::par.on(stream),
#else
    thrust::cuda::par.on(stream),
#endif
    cell_count.data(),
    cell_count.data() + N_cells,
    cell_count_sum.data());

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  find_cell_contents<<<grid_size, block_size, 0, stream>>>(
    partition_direction,
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL
}

// The sorted index map stays on GPU 0. Geometry and spins always travel together,
// including repeated force evaluations at unchanged positions.
__global__ void pack_domain(
  int n,
  int local,
  int start,
  const int* order,
  const int* type,
  const double* position,
  const double* spin,
  int* local_type,
  double* local_position,
  double* local_spin)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= local)
    return;
  int id = order[(start + i) % n];
  local_type[i] = type[id];
  for (int d = 0; d < 3; ++d) {
    local_position[i + size_t(d) * local] = position[id + size_t(d) * n];
    local_spin[i + size_t(d) * local] = spin[id + size_t(d) * n];
  }
}

// Ownership may stay fixed while atoms move through slab faces. The 2.5*rc
// halo contains every two-hop environment while each atom moves < 0.2*rc:
// an initial separation can grow to at most 2*rc + 2*(0.2*rc) < 2.5*rc.
__global__ void check_partition_displacement(
  int n,
  Box box,
  double limit_squared,
  const double* position,
  const double* reference,
  int* changed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double dx = position[i] - reference[i];
  double dy = position[i + n] - reference[i + n];
  double dz = position[i + 2 * n] - reference[i + 2 * n];
  apply_mic(box, dx, dy, dz);
  if (dx * dx + dy * dy + dz * dz >= limit_squared)
    atomicExch(changed, 1);
}

__global__ void collect_domain(
  int n,
  int count,
  int first,
  const int* order,
  const double* packed,
  double* energy,
  double* force,
  double* virial,
  double* mforce)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int id = order[first + i];
  energy[id] += packed[i];
  for (int d = 0; d < 3; ++d) {
    force[id + size_t(d) * n] += packed[i + size_t(1 + d) * count];
    mforce[id + size_t(d) * n] += packed[i + size_t(4 + d) * count];
  }
  for (int d = 0; d < 9; ++d)
    virial[id + size_t(d) * n] += packed[i + size_t(7 + d) * count];
}

void transfer(void* dst, int dst_gpu, const void* src, int src_gpu, size_t bytes)
{
  if (!bytes)
    return;
  if (dst_gpu == src_gpu) {
    CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToDevice));
  } else {
    // The runtime stages through host memory if direct peer transfer is unavailable.
#ifdef USE_HIP
    CHECK(hipMemcpyPeer(dst, dst_gpu, src, src_gpu, bytes));
#else
    CHECK(cudaMemcpyPeer(dst, dst_gpu, src, src_gpu, bytes));
#endif
  }
}
} // namespace

void NEP_Spin_MultiGPU::Domain::resize(int count)
{
  if (type.size() == size_t(count))
    return;
  if (type.size() != (count)) {
    type.resize(count);
  }
  if (position.size() != (size_t(count) * 3)) {
    position.resize(size_t(count) * 3);
  }
  if (spin.size() != (size_t(count) * 3)) {
    spin.resize(size_t(count) * 3);
  }
  if (energy.size() != (count)) {
    energy.resize(count);
  }
  if (force.size() != (size_t(count) * 3)) {
    force.resize(size_t(count) * 3);
  }
  if (mforce.size() != (size_t(count) * 3)) {
    mforce.resize(size_t(count) * 3);
  }
  if (virial.size() != (size_t(count) * 9)) {
    virial.resize(size_t(count) * 9);
  }
  potential->resize_domain(count);
}

NEP_Spin_MultiGPU::NEP_Spin_MultiGPU(int devices, const char* model, int atoms, int direction)
  : devices_(devices), atoms_(atoms), direction_(direction)
{
  if (devices < 2 || atoms <= 0 || direction < -1 || direction > 2)
    PRINT_INPUT_ERROR("Invalid NEP-Spin multi-GPU partition.");
  for (int gpu = 0; gpu < devices_; ++gpu) {
    CHECK(gpuSetDevice(gpu));
    domains_.emplace_back(new Domain);
    domains_.back()->potential.reset(new NEP_Spin(model, 1));
  }
  rc = domains_[0]->potential->rc;
  CHECK(gpuSetDevice(0));
  CHECK(gpuStreamCreate(&partition_stream_));
  order_.resize(atoms_);
  partition_position_.resize(size_t(atoms_) * 3);
  partition_changed_.resize(1);
  cell_count_.resize(1);
  cell_sum_.resize(1);
  printf(
    "NEP-Spin: spatial decomposition over %d GPUs; two-cutoff geometry/spin halos.\n", devices_);
}

NEP_Spin_MultiGPU::~NEP_Spin_MultiGPU()
{
  for (int gpu = 0; gpu < devices_; ++gpu) {
    CHECK(gpuSetDevice(gpu));
    domains_[gpu].reset();
  }
  CHECK(gpuSetDevice(0));
  CHECK(gpuStreamDestroy(partition_stream_));
}

void NEP_Spin_MultiGPU::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  GPU_Vector<double>& energy,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& mforce)
{
  CHECK(gpuSetDevice(0));
  const int n = type.size();
  if (n != atoms_ || position.size() != size_t(n) * 3 || spin.size() != size_t(n) * 3)
    PRINT_INPUT_ERROR("NEP-Spin multi-GPU atom count mismatch.");
  if (!(box.pbc_x && box.pbc_y && box.pbc_z))
    PRINT_INPUT_ERROR("NEP_Spin requires periodic boundary conditions in all three directions.");
  bool repartition = !partition_valid_ || !std::equal(box.cpu_h, box.cpu_h + 9, partition_box_);
  if (!repartition) {
    partition_changed_.fill(0);
    const double displacement_limit = 0.2 * rc;
    check_partition_displacement<<<(n + 127) / 128, 128>>>(
      n,
      box,
      displacement_limit * displacement_limit,
      position.data(),
      partition_position_.data(),
      partition_changed_.data());
    GPU_CHECK_KERNEL
    int changed = 0;
    partition_changed_.copy_to_host(&changed);
    repartition = changed != 0;
  }
  if (repartition) {
    int bins[3];
    const double cell_width = 0.5 * rc;
    box.get_num_bins(cell_width, bins);
    for (int d = 0; d < 3; ++d)
      bins[d] = std::max(1, bins[d]);
    const long long cells = static_cast<long long>(bins[0]) * bins[1] * bins[2];
    if (cells > std::numeric_limits<int>::max())
      PRINT_INPUT_ERROR("NEP-Spin partition cell count exceeds the supported int range.");
    if (direction_ < 0)
      direction_ = std::max_element(bins, bins + 3) - bins;
    find_cell_list(
      partition_stream_,
      direction_,
      cell_width,
      bins,
      box,
      n,
      position,
      cell_count_,
      cell_sum_,
      order_);
#ifdef USE_HIP
    CHECK(hipStreamSynchronize(partition_stream_));
#else
    CHECK(cudaStreamSynchronize(partition_stream_));
#endif
    prefix_.resize(cells + 1);
    cell_sum_.copy_to_host(prefix_.data(), cells);
    prefix_[cells] = n;
    const int longitudinal = bins[direction_];
    const int transverse = cells / longitudinal;
    // Four half-cutoff cells span 2*rc; the fifth supplies displacement
    // skin plus a rounding margin. If halos overlap, copy each global atom once and retain full
    // PBC.
    constexpr int halo = 5;
    for (int gpu = 0; gpu < devices_; ++gpu) {
      Domain& dom = *domains_[gpu];
      int b0 = static_cast<long long>(gpu) * longitudinal / devices_;
      int b1 = static_cast<long long>(gpu + 1) * longitudinal / devices_;
      dom.global_first = prefix_[b0 * transverse];
      dom.owned = prefix_[b1 * transverse] - dom.global_first;
      if (!dom.owned) {
        dom.size = 0;
        continue;
      }
      if (b1 - b0 + 2 * halo >= longitudinal) {
        dom.start = 0;
        dom.size = n;
        dom.first = dom.global_first;
      } else {
        int lo = (b0 - halo + longitudinal) % longitudinal;
        int hi = (b1 + halo) % longitudinal;
        dom.start = prefix_[lo * transverse];
        dom.size = prefix_[hi * transverse] - dom.start;
        if (hi < lo)
          dom.size += n;
        dom.first = (dom.global_first - dom.start + n) % n;
      }
    }
    partition_position_.copy_from_device(position.data());
    std::copy(box.cpu_h, box.cpu_h + 9, partition_box_);
    partition_valid_ = true;
  }
  for (int gpu = 0; gpu < devices_; ++gpu) {
    Domain& dom = *domains_[gpu];
    if (!dom.owned)
      continue;
    CHECK(gpuSetDevice(gpu));
    dom.resize(dom.size);
    CHECK(gpuSetDevice(0));
    if (packed_type_.size() != (dom.size)) {
      packed_type_.resize(dom.size);
    }
    if (packed_position_.size() != (size_t(dom.size) * 3)) {
      packed_position_.resize(size_t(dom.size) * 3);
    }
    if (packed_spin_.size() != (size_t(dom.size) * 3)) {
      packed_spin_.resize(size_t(dom.size) * 3);
    }
    pack_domain<<<(dom.size + 127) / 128, 128>>>(
      n,
      dom.size,
      dom.start,
      order_.data(),
      type.data(),
      position.data(),
      spin.data(),
      packed_type_.data(),
      packed_position_.data(),
      packed_spin_.data());
    GPU_CHECK_KERNEL
    CHECK(gpuDeviceSynchronize());
    transfer(dom.type.data(), gpu, packed_type_.data(), 0, size_t(dom.size) * sizeof(int));
    transfer(
      dom.position.data(), gpu, packed_position_.data(), 0, size_t(dom.size) * 3 * sizeof(double));
    transfer(dom.spin.data(), gpu, packed_spin_.data(), 0, size_t(dom.size) * 3 * sizeof(double));
  }
  std::vector<std::future<void>> workers;
  for (int gpu = 0; gpu < devices_; ++gpu) {
    if (!domains_[gpu]->owned)
      continue;
    workers.emplace_back(std::async(std::launch::async, [&, gpu] {
      CHECK(gpuSetDevice(gpu));
      Domain& dom = *domains_[gpu];
      dom.energy.fill(0.0);
      dom.force.fill(0.0);
      dom.virial.fill(0.0);
      dom.mforce.fill(0.0);
      if (repartition)
        dom.potential->invalidate_neighbors();
      Box local_box = box; // Neighbor/box bookkeeping is local to each host worker.
      dom.potential->compute(
        local_box, dom.type, dom.position, dom.spin, dom.energy, dom.force, dom.virial, dom.mforce);
      CHECK(gpuDeviceSynchronize());
    }));
  }
  for (auto& worker : workers)
    worker.get();
  CHECK(gpuSetDevice(0));
  for (int gpu = 0; gpu < devices_; ++gpu) {
    const Domain& dom = *domains_[gpu];
    if (!dom.owned)
      continue;
    if (collected_.size() != (size_t(dom.owned) * 16)) {
      collected_.resize(size_t(dom.owned) * 16);
    }
    auto copy_planes = [&](const GPU_Vector<double>& src, int planes, int offset) {
      for (int d = 0; d < planes; ++d)
        transfer(
          collected_.data() + size_t(offset + d) * dom.owned,
          0,
          src.data() + size_t(d) * dom.size + dom.first,
          gpu,
          size_t(dom.owned) * sizeof(double));
    };
    copy_planes(dom.energy, 1, 0);
    copy_planes(dom.force, 3, 1);
    copy_planes(dom.mforce, 3, 4);
    copy_planes(dom.virial, 9, 7);
    // Only owner outputs are collected. Every owner's neighbors and their
    // descriptors are complete inside 2*rc; ghost output must never be summed.
    collect_domain<<<(dom.owned + 127) / 128, 128>>>(
      n,
      dom.owned,
      dom.global_first,
      order_.data(),
      collected_.data(),
      energy.data(),
      force.data(),
      virial.data(),
      mforce.data());
    GPU_CHECK_KERNEL
    CHECK(gpuDeviceSynchronize());
  }
}
