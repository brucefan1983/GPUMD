/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
// Standalone CUDA/HIP regression: logical tiles, upper wave64 lanes and tails.
#include "force/nep_spin/spin_warp.cuh"
#include <cstdio>
#include <cstdlib>
#ifdef USE_HIP
#define GPU(name) hip##name
#else
#define GPU(name) cuda##name
#endif
#define CHECK(call) do { auto e = (call); if (e != GPU(Success)) { \
  std::fprintf(stderr, "%s\n", GPU(GetErrorString)(e)); std::exit(1); } } while (0)

template<int Width>
__global__ void check_tiles(int centers, unsigned long long* masks, double* sums,
                            int* broadcasts, int* shared_values, int* wave_size) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  const int center = id / Width, lane = threadIdx.x % Width;
  const auto participating = nep_spin::warp_ballot(center < centers);
  const auto tile = nep_spin::tile_mask(Width);
  if (center >= centers) return;
  __shared__ int shared[128];
  if (lane == 0) shared[threadIdx.x / Width] = 700 + center;
  __syncwarp(tile);
  double sum = id;
  for (int offset = Width / 2; offset; offset /= 2)
    sum += __shfl_down_sync(participating, sum, offset, Width);
  const int source = __shfl_sync(tile, 1000 + id, 0, Width);
  masks[id] = tile;
  broadcasts[id] = source;
  shared_values[id] = shared[threadIdx.x / Width];
  if (lane == 0) sums[center] = sum;
  if (id == 0) *wave_size = warpSize;
}

template<int Width>
void run(int centers) {
  constexpr int capacity = 256;
  unsigned long long *masks, host_masks[capacity];
  double *sums, host_sums[capacity];
  int *broadcasts, *shared, *wave, host_broadcasts[capacity], host_shared[capacity], host_wave;
  CHECK(GPU(Malloc)(&masks, sizeof(host_masks)));
  CHECK(GPU(Malloc)(&sums, sizeof(host_sums)));
  CHECK(GPU(Malloc)(&broadcasts, sizeof(host_broadcasts)));
  CHECK(GPU(Malloc)(&shared, sizeof(host_shared)));
  CHECK(GPU(Malloc)(&wave, sizeof(int)));
  check_tiles<Width><<<(centers * Width + 127) / 128, 128>>>(centers, masks, sums, broadcasts, shared, wave);
  CHECK(GPU(GetLastError)());
  CHECK(GPU(DeviceSynchronize)());
  CHECK(GPU(Memcpy)(host_masks, masks, sizeof(host_masks), GPU(MemcpyDeviceToHost)));
  CHECK(GPU(Memcpy)(host_sums, sums, sizeof(host_sums), GPU(MemcpyDeviceToHost)));
  CHECK(GPU(Memcpy)(host_broadcasts, broadcasts, sizeof(host_broadcasts), GPU(MemcpyDeviceToHost)));
  CHECK(GPU(Memcpy)(host_shared, shared, sizeof(host_shared), GPU(MemcpyDeviceToHost)));
  CHECK(GPU(Memcpy)(&host_wave, wave, sizeof(int), GPU(MemcpyDeviceToHost)));
  for (int id = 0; id < centers * Width; ++id) {
    const int center = id / Width;
    unsigned long long expected_mask = 0;
    for (int j = center * Width; j < (center + 1) * Width; ++j)
      expected_mask |= 1ull << (j % host_wave);
    double expected_sum = 0;
    for (int j = center * Width; j < (center + 1) * Width; ++j) expected_sum += j;
    if (host_masks[id] != expected_mask || host_broadcasts[id] != 1000 + center * Width ||
        host_shared[id] != 700 + center || host_sums[center] != expected_sum) {
      std::fprintf(stderr, "FAIL width=%d centers=%d lane=%d wave=%d\n", Width, centers, id, host_wave);
      std::exit(2);
    }
  }
  CHECK(GPU(Free)(masks)); CHECK(GPU(Free)(sums)); CHECK(GPU(Free)(broadcasts));
  CHECK(GPU(Free)(shared)); CHECK(GPU(Free)(wave));
  std::printf("PASS width=%d centers=%d wave=%d\n", Width, centers, host_wave);
}
int main() {
  run<4>(13); run<4>(47); run<8>(7); run<8>(23); run<32>(3); run<32>(7);
}
