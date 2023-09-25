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

#include "utilities/error.cuh"
#include "vector.cuh"
#include <string.h>    // memcpy
#define BLOCK_SIZE 512 // optimized

__global__ void gpu_set_zero(int number_of_elements, real* g_state_real, real* g_state_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_elements) {
    g_state_real[n] = 0;
    g_state_imag[n] = 0;
  }
}

void Vector::initialize_gpu(int n)
{
  this->n = n;
  array_size = n * sizeof(real);
  CHECK(cudaMalloc((void**)&real_part, array_size));
  CHECK(cudaMalloc((void**)&imag_part, array_size));
}

Vector::Vector(int n)
{
  initialize_gpu(n);
  gpu_set_zero<<<(n - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(n, real_part, imag_part);
  CHECK(cudaGetLastError());
}

__global__ void gpu_copy_state(int N, real* in_real, real* in_imag, real* out_real, real* out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    out_real[n] = in_real[n];
    out_imag[n] = in_imag[n];
  }
}

Vector::Vector(Vector& original)
{
  // Just teach myself: one can access private members of another instance
  // of the class from within the class
  initialize_gpu(original.n);
  gpu_copy_state<<<(n - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    n, original.real_part, original.imag_part, real_part, imag_part);
  CHECK(cudaGetLastError());
}

Vector::~Vector()
{
  CHECK(cudaFree(real_part));
  CHECK(cudaFree(imag_part));
}

__global__ void gpu_add_state(int n, real* in_real, real* in_imag, real* out_real, real* out_imag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out_real[i] += in_real[i];
    out_imag[i] += in_imag[i];
  }
}

void Vector::add(Vector& other)
{
  gpu_add_state<<<(n - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    n, other.real_part, other.imag_part, real_part, imag_part);
  CHECK(cudaGetLastError());
}

void Vector::copy(Vector& other)
{
  gpu_copy_state<<<(n - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    n, other.real_part, other.imag_part, real_part, imag_part);
  CHECK(cudaGetLastError());
}

__global__ void gpu_apply_sz(int n, real* in_real, real* in_imag, real* out_real, real* out_imag)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (i % 2 == 0) {
      out_real[i] = in_real[i];
      out_imag[i] = in_imag[i];
    } else {
      out_real[i] = -in_real[i];
      out_imag[i] = -in_imag[i];
    }
  }
}

void Vector::apply_sz(Vector& other)
{
  gpu_apply_sz<<<(n - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    n, other.real_part, other.imag_part, real_part, imag_part);
  CHECK(cudaGetLastError());
}

void Vector::copy_from_host(real* other_real, real* other_imag)
{
  CHECK(cudaMemcpy(real_part, other_real, array_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(imag_part, other_imag, array_size, cudaMemcpyHostToDevice));
}

void Vector::copy_to_host(real* target_real, real* target_imag)
{
  CHECK(cudaMemcpy(target_real, real_part, array_size, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(target_imag, imag_part, array_size, cudaMemcpyDeviceToHost));
}

void Vector::swap(Vector& other)
{
  real* tmp_real = real_part;
  real* tmp_imag = imag_part;
  real_part = other.real_part, imag_part = other.imag_part;
  other.real_part = tmp_real;
  other.imag_part = tmp_imag;
}

__device__ void warp_reduce(volatile real* s, int t)
{
  s[t] += s[t + 32];
  s[t] += s[t + 16];
  s[t] += s[t + 8];
  s[t] += s[t + 4];
  s[t] += s[t + 2];
  s[t] += s[t + 1];
}

__global__ void gpu_find_inner_product_1(
  int number_of_atoms,
  real* g_final_state_real,
  real* g_final_state_imag,
  real* g_random_state_real,
  real* g_random_state_imag,
  real* g_inner_product_real,
  real* g_inner_product_imag,
  int g_offset)
{
  int tid = threadIdx.x;
  int n = blockIdx.x * blockDim.x + tid;
  int m;
  real a, b, c, d;
  __shared__ real s_data_real[BLOCK_SIZE];
  __shared__ real s_data_imag[BLOCK_SIZE];
  s_data_real[tid] = 0.0;
  s_data_imag[tid] = 0.0;

  if (n < number_of_atoms) {
    a = g_final_state_real[n];
    b = g_final_state_imag[n];
    c = g_random_state_real[n];
    d = g_random_state_imag[n];
    s_data_real[tid] = (a * c + b * d);
    s_data_imag[tid] = (b * c - a * d);
  }
  __syncthreads();

  if (tid < 256) {
    m = tid + 256;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 128) {
    m = tid + 128;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 64) {
    m = tid + 64;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 32) {
    warp_reduce(s_data_real, tid);
    warp_reduce(s_data_imag, tid);
  }
  if (tid == 0) {
    g_inner_product_real[blockIdx.x + g_offset] = s_data_real[0];
    g_inner_product_imag[blockIdx.x + g_offset] = s_data_imag[0];
  }
}

void Vector::inner_product_1(int number_of_atoms, Vector& other, Vector& target, int offset)
{
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE + 1;
  gpu_find_inner_product_1<<<grid_size, BLOCK_SIZE>>>(
    number_of_atoms,
    real_part,
    imag_part,
    other.real_part,
    other.imag_part,
    target.real_part,
    target.imag_part,
    offset);
  CHECK(cudaGetLastError());
}

__global__ void gpu_find_inner_product_2(
  int number_of_atoms,
  real* g_inner_product_1_real,
  real* g_inner_product_1_imag,
  real* g_inner_product_2_real,
  real* g_inner_product_2_imag)
{
  //<<<para.number_of_energy_points, BLOCK_SIZE)>>>
  int tid = threadIdx.x;
  int patch, n, m;

  __shared__ real s_data_real[BLOCK_SIZE];
  __shared__ real s_data_imag[BLOCK_SIZE];
  s_data_real[tid] = 0.0;
  s_data_imag[tid] = 0.0;
  int number_of_blocks = (number_of_atoms - 1) / BLOCK_SIZE + 1;
  int number_of_patches = (number_of_blocks - 1) / BLOCK_SIZE + 1;

  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * BLOCK_SIZE;
    if (n < number_of_blocks) {
      m = blockIdx.x * number_of_blocks + n;
      s_data_real[tid] += g_inner_product_1_real[m];
      s_data_imag[tid] += g_inner_product_1_imag[m];
    }
  }
  __syncthreads();

  if (tid < 256) {
    m = tid + 256;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 128) {
    m = tid + 128;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 64) {
    m = tid + 64;
    s_data_real[tid] += s_data_real[m];
    s_data_imag[tid] += s_data_imag[m];
  }
  __syncthreads();
  if (tid < 32) {
    warp_reduce(s_data_real, tid);
    warp_reduce(s_data_imag, tid);
  }
  if (tid == 0) {
    g_inner_product_2_real[blockIdx.x] = s_data_real[0];
    g_inner_product_2_imag[blockIdx.x] = s_data_imag[0];
  }
}

void Vector::inner_product_2(int number_of_atoms, int number_of_moments, Vector& target)
{
  gpu_find_inner_product_2<<<number_of_moments, BLOCK_SIZE>>>(
    number_of_atoms, real_part, imag_part, target.real_part, target.imag_part);
  CHECK(cudaGetLastError());
}
