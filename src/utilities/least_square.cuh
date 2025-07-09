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

#include "utilities/gpu_vector.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA_ERROR(call)                                                                     \
  do {                                                                                             \
    cudaError_t err = call;                                                                        \
    if (err != cudaSuccess) {                                                                      \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (err_num=" << err << ") at "     \
                << __FILE__ << ":" << __LINE__ << std::endl;                                       \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  } while (0)

#define CHECK_CUSOLVER(call)                                                                       \
  {                                                                                                \
    cusolverStatus_t status = call;                                                                \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                                       \
      fprintf(stderr, "cuSOLVER错误: %d 于文件 %s 行 %d\n", status, __FILE__, __LINE__);           \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }

// -------------------------------------------
// - A_d：整型数组，大小 n，存放 type_sum (每种原子类型的个数)
// - b_d：单个 double，存放 energy_ref (总能量)
// - x_d：double 数组，大小 n，输出每种类型对应的能量 E[i]
// - n  ：类型数 num_types
//
// 数学公式：
//   E_i = (A[i] * b) / ( ∑(A[j]^2) )    (j=0..n-1)
// -------------------------------------------
__global__ void kernelComputePseudoInverse1xN(const int* A_d, const float* b_d, float* x_d, int n)
{
  __shared__ float sum_of_squares;

  // 线程0 串行计算 ∑(A[j]^2)，并写入 sum_of_squares
  if (threadIdx.x == 0) {
    float tmp = 0.0f;
    for (int i = 0; i < n; i++) {
      float val = static_cast<float>(A_d[i]);
      tmp += val * val;
    }
    sum_of_squares = tmp;
  }
  __syncthreads();

  // 每个线程计算 x_d[i] = (A_d[i] * b) / sum_of_squares
  int i = threadIdx.x;
  if (i < n) {
    float val_i = static_cast<float>(A_d[i]);
    float b_val = b_d[0];
    x_d[i] = (val_i * b_val) / sum_of_squares;
  }
}

__global__ void kernelComputePseudoInverseReg1xN(
  const int* A_d,
  const float* b_d,
  const float* lambda_d, // 分母加入l2正则化
  float* x_d,
  int n)
{
  __shared__ float sum_of_squares;

  // (1) 线程0 计算 ∑(A_d[j]^2)
  if (threadIdx.x == 0) {
    float tmp = 0.0f;
    for (int j = 0; j < n; j++) {
      float val = static_cast<float>(A_d[j]);
      tmp += val * val;
    }
    sum_of_squares = tmp;
  }
  __syncthreads();

  // (2) 每个线程计算 x_d[i]
  int i = threadIdx.x;
  if (i < n) {
    float val_i = static_cast<float>(A_d[i]); // type_sum[i]
    float b_val = b_d[0];                     // energy_ref
    float lambda_val = lambda_d[0];           // λ

    // 分母: sum_of_squares + λ
    x_d[i] = (val_i * b_val) / (sum_of_squares + lambda_val);
  }
}

// -------------------------------------------
// 封装函数：在 GPU 上计算每种类型的能量 E[i]
// 输入：
//  - num_types           : 原子类型数
//  - type_sum[]     : 整型数组(长度 num_types)，各类型原子个数
//  - energy_ref     : 总能量 (float)
// 输出：
//  - energy_per_type[] : float 数组(长度 num_types)，每种类型的能量
// -------------------------------------------
void computeEnergyPerType(
  int num_types, const int* type_sum, float energy_ref, float* energy_per_type)
{
  float* b_d = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&b_d, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(b_d, &energy_ref, sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(num_types);
  dim3 grid(1);
  kernelComputePseudoInverse1xN<<<grid, block>>>(type_sum, b_d, energy_per_type, num_types);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUDA_ERROR(cudaFree(b_d));
}

void computeEnergyPerTypeReg(
  int num_types, const int* type_sum, float energy_ref, float lambda, float* energy_per_type)
{
  float *b_d = nullptr, *lambda_d = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&b_d, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&lambda_d, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMemcpy(b_d, &energy_ref, sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(lambda_d, &lambda, sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(num_types);
  dim3 grid(1);
  kernelComputePseudoInverseReg1xN<<<grid, block>>>(
    type_sum, b_d, lambda_d, energy_per_type, num_types);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaFree(b_d));
  CHECK_CUDA_ERROR(cudaFree(lambda_d));
}

void computeMultiBatchEnergyShift(
  int num_types,
  int num_batches,
  const std::vector<std::vector<int>>& batch_type_sums,
  const std::vector<float>& batch_energies,
  float* energy_per_type,
  bool verbose)
{
  if (verbose) {
    printf(
      "Computing energy shifts for %d batches (using direct least squares method):\n", num_batches);
    for (int i = 0; i < num_batches; i++) {
      printf("Batch %d: ", i);
      for (int t = 0; t < num_types; t++) {
        printf("Type%d=%d ", t, batch_type_sums[i][t]);
      }
      printf("Energy=%.6f\n", batch_energies[i]);
    }
  }

  std::vector<float> energy_per_atom(num_batches);
  float min_epa = 1e10f;
  float max_epa = -1e10f;
  float total_energy = 0.0f;
  int total_atoms = 0;

  std::vector<int> total_atoms_by_type(num_types, 0);

  for (int i = 0; i < num_batches; i++) {
    int atoms_in_batch = 0;
    for (int t = 0; t < num_types; t++) {
      atoms_in_batch += batch_type_sums[i][t];
      total_atoms_by_type[t] += batch_type_sums[i][t];
    }

    energy_per_atom[i] = batch_energies[i] / atoms_in_batch;
    min_epa = std::min(min_epa, energy_per_atom[i]);
    max_epa = std::max(max_epa, energy_per_atom[i]);

    total_energy += batch_energies[i];
    total_atoms += atoms_in_batch;
  }

  float avg_epa = total_energy / total_atoms;
  float range_epa = max_epa - min_epa;

  if (verbose) {
    printf("Energy per atom statistics:\n");
    printf(
      "  Min: %.6f, Max: %.6f, Avg: %.6f, Range: %.6f eV/atom\n",
      min_epa,
      max_epa,
      avg_epa,
      range_epa);
    printf("Atom type distribution:\n");
    for (int t = 0; t < num_types; t++) {
      printf(
        "  Type %d: %d atoms (%.1f%%)\n",
        t,
        total_atoms_by_type[t],
        100.0f * total_atoms_by_type[t] / total_atoms);
    }
  }

  // Build regularized least squares system
  std::vector<float> AtA(num_types * num_types, 0.0f);
  std::vector<float> Atb(num_types, 0.0f);

  // Populate A^T*A and A^T*b
  for (int i = 0; i < num_types; i++) {
    for (int j = 0; j < num_types; j++) {
      float sum = 0.0f;
      for (int k = 0; k < num_batches; k++) {
        sum += batch_type_sums[k][i] * batch_type_sums[k][j];
      }

      // Add small regularization term for numerical stability
      if (i == j) {
        sum += 0.001f;
      }

      AtA[i * num_types + j] = sum;
    }

    float sum = 0.0f;
    for (int k = 0; k < num_batches; k++) {
      sum += batch_type_sums[k][i] * batch_energies[k];
    }
    Atb[i] = sum;
  }

  // Solve the equation
  std::vector<float> x(num_types, 0.0f);
  bool system_solved = false;

  // Attempt direct solution
  try {
    std::vector<float> A = AtA; // Make a copy
    std::vector<float> b = Atb;

    // Gaussian elimination
    for (int i = 0; i < num_types; i++) {
      // Find pivot element
      int max_row = i;
      float max_val = fabs(A[i * num_types + i]);
      for (int j = i + 1; j < num_types; j++) {
        if (fabs(A[j * num_types + i]) > max_val) {
          max_row = j;
          max_val = fabs(A[j * num_types + i]);
        }
      }

      // Check if pivot is too small (near-singular)
      if (max_val < 1e-10f) {
        if (verbose) {
          printf("Warning: Near-singular matrix detected during elimination step %d\n", i);
        }
        break;
      }

      // Swap rows
      if (max_row != i) {
        for (int j = i; j < num_types; j++) {
          std::swap(A[i * num_types + j], A[max_row * num_types + j]);
        }
        std::swap(b[i], b[max_row]);
      }

      // Elimination
      for (int j = i + 1; j < num_types; j++) {
        float factor = A[j * num_types + i] / A[i * num_types + i];
        b[j] -= factor * b[i];
        for (int k = i; k < num_types; k++) {
          A[j * num_types + k] -= factor * A[i * num_types + k];
        }
      }
    }

    // Back substitution
    for (int i = num_types - 1; i >= 0; i--) {
      float sum = 0.0f;
      for (int j = i + 1; j < num_types; j++) {
        sum += A[i * num_types + j] * x[j];
      }

      // Check diagonal element
      if (fabs(A[i * num_types + i]) < 1e-10f) {
        if (verbose) {
          printf(
            "Warning: Near-zero diagonal element detected during back substitution step %d\n", i);
        }
        break;
      }

      x[i] = (b[i] - sum) / A[i * num_types + i];
    }

    system_solved = true;
  } catch (const std::exception& e) {
    if (verbose) {
      printf("Error solving equations: %s\n", e.what());
    }
  }

  // If solution fails, use fallback method
  if (!system_solved) {
    if (verbose) {
      printf("Using fallback method: based on average energy per atom\n");
    }

    // Apply slightly different average energy per atom for each type
    for (int t = 0; t < num_types; t++) {
      // Introduce small variations to avoid perfect symmetry
      x[t] = avg_epa * (0.95f + 0.02f * t);
    }
  }

  CHECK_CUDA_ERROR(
    cudaMemcpy(energy_per_type, x.data(), sizeof(float) * num_types, cudaMemcpyHostToDevice));

  if (verbose) {
    printf("Computed energy shift values:\n");
    for (int t = 0; t < num_types; t++) {
      printf("  Type %d: %.6f\n", t, x[t]);
    }

    // Calculate fitting quality
    float total_residual = 0.0f;

    for (int i = 0; i < num_batches && i < 10; i++) {
      float computed_energy = 0.0f;
      for (int j = 0; j < num_types; j++) {
        computed_energy += batch_type_sums[i][j] * x[j];
      }
      float residual = batch_energies[i] - computed_energy;
      total_residual += residual * residual;

      printf(
        "Batch %d: Original=%.6f, Computed=%.6f, Residual=%.6f\n",
        i,
        batch_energies[i],
        computed_energy,
        residual);
    }

    // Show last few batches
    if (num_batches > 10) {
      printf("...\n");
      for (int i = std::max(10, num_batches - 5); i < num_batches; i++) {
        float computed_energy = 0.0f;
        for (int j = 0; j < num_types; j++) {
          computed_energy += batch_type_sums[i][j] * x[j];
        }
        float residual = batch_energies[i] - computed_energy;
        total_residual += residual * residual;

        printf(
          "Batch %d: Original=%.6f, Computed=%.6f, Residual=%.6f\n",
          i,
          batch_energies[i],
          computed_energy,
          residual);
      }
    }

    // Complete residual calculation for all batches
    for (int i = 10; i < std::max(10, num_batches - 5); i++) {
      float computed_energy = 0.0f;
      for (int j = 0; j < num_types; j++) {
        computed_energy += batch_type_sums[i][j] * x[j];
      }
      float residual = batch_energies[i] - computed_energy;
      total_residual += residual * residual;
    }

    printf("Total residual squared: %.6f\n", total_residual);
  }
}

void computeMultiBatchEnergyShiftUniform(
  int num_types,
  int num_batches,
  const std::vector<std::vector<int>>& batch_type_sums,
  const std::vector<float>& batch_energies,
  float* energy_per_type,
  bool verbose)
{
  if (verbose) {
    printf(
      "Computing energy shifts for %d batches (using direct least squares method):\n", num_batches);
    for (int i = 0; i < num_batches; i++) {
      printf("Batch %d: ", i);
      for (int t = 0; t < num_types; t++) {
        printf("Type%d=%d ", t, batch_type_sums[i][t]);
      }
      printf("Energy=%.6f\n", batch_energies[i]);
    }
  }

  std::vector<float> energy_per_atom(num_batches);
  float min_epa = 1e10f;
  float max_epa = -1e10f;
  float total_energy = 0.0f;
  int total_atoms = 0;
  float lambda = 0.001f;
  std::vector<int> total_atoms_by_type(num_types, 0);

  for (int i = 0; i < num_batches; i++) {
    int atoms_in_batch = 0;
    for (int t = 0; t < num_types; t++) {
      atoms_in_batch += batch_type_sums[i][t];
      total_atoms_by_type[t] += batch_type_sums[i][t];
    }

    energy_per_atom[i] = batch_energies[i] / atoms_in_batch;
    min_epa = std::min(min_epa, energy_per_atom[i]);
    max_epa = std::max(max_epa, energy_per_atom[i]);

    total_energy += batch_energies[i];
    total_atoms += atoms_in_batch;
  }

  float avg_epa = total_energy / total_atoms;
  float range_epa = max_epa - min_epa;

  if (verbose) {
    printf("Energy per atom statistics:\n");
    printf(
      "  Min: %.6f, Max: %.6f, Avg: %.6f, Range: %.6f, Total: %.6f eV/atom\n",
      min_epa,
      max_epa,
      avg_epa,
      range_epa,
      total_energy);
    printf("Atom type distribution:\n");
    for (int t = 0; t < num_types; t++) {
      printf(
        "  Type %d: %d atoms (%.1f%%)\n",
        t,
        total_atoms_by_type[t],
        100.0f * total_atoms_by_type[t] / total_atoms);
    }
  }

  int* type_sum_d = nullptr;
  float *energy_d = nullptr, *lambda_d = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc((void**)&type_sum_d, sizeof(int) * num_types));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&energy_d, sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&lambda_d, sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(
    type_sum_d, total_atoms_by_type.data(), sizeof(int) * num_types, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(energy_d, &total_energy, sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(lambda_d, &lambda, sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(num_types);
  dim3 grid(1);

  kernelComputePseudoInverseReg1xN<<<grid, block>>>(
    type_sum_d, energy_d, lambda_d, energy_per_type, num_types);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  CHECK_CUDA_ERROR(cudaFree(type_sum_d));
  CHECK_CUDA_ERROR(cudaFree(energy_d));
  CHECK_CUDA_ERROR(cudaFree(lambda_d));
  if (verbose) {
    std::vector<float> result(num_types);
    CHECK_CUDA_ERROR(cudaMemcpy(
      result.data(), energy_per_type, sizeof(float) * num_types, cudaMemcpyDeviceToHost));

    printf("Computed energy shift values:\n");
    for (int t = 0; t < num_types; t++) {
      printf("  Type %d: %.6f\n", t, result[t]);
    }

    // Verify the result
    float computed_total = 0.0f;
    for (int t = 0; t < num_types; t++) {
      computed_total += total_atoms_by_type[t] * result[t];
    }

    printf(
      "Verification: original total energy = %.6f, computed total energy = %.6f, difference = "
      "%.6f\n",
      total_energy,
      computed_total,
      total_energy - computed_total);
  }
}