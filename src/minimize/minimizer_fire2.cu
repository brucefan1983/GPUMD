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
The FIRE2 (fast inertial relaxation engine) minimizer
Reference: Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire2.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{

__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

__global__ void filter_v(const int size, const double dt, const double maxstep, double* v)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size) {
    if (abs(v[n]) * dt > maxstep) {
      v[n] = maxstep / dt * v[n] / abs(v[n]);
    }
  }
}

// Transpose a 3x3 matrix stored as 9-element array
void transpose9(double* matrix)
{
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      std::swap(matrix[i * 3 + j], matrix[j * 3 + i]);
    }
  }
}

// Matrix multiplication: C = A * B (3x3 matrices as 9-element arrays)
void matrix_multiply(const double* A, const double* B, double* C)
{
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i * 3 + j] = 0.0;
      for (int k = 0; k < 3; ++k) {
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
      }
    }
  }
}

// Solve linear equation: A * X = B for 3x3 matrices using Gaussian elimination
void solve_linear_equation_3x3(const double* A, const double* B, double* X)
{
  double a[3][3], b[3][3];

  // Copy input matrices (row-major)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      a[i][j] = A[i * 3 + j];
      b[i][j] = B[i * 3 + j];
    }
  }

  // Gaussian elimination with partial pivoting
  for (int col = 0; col < 3; ++col) {
    // Find pivot
    int pivot_row = col;
    for (int i = col + 1; i < 3; ++i) {
      if (fabs(a[i][col]) > fabs(a[pivot_row][col])) {
        pivot_row = i;
      }
    }

    if (fabs(a[pivot_row][col]) < 1e-9) {
      printf("Matrix is singular or nearly singular!\n");
      return;
    }

    // Swap rows if needed
    if (pivot_row != col) {
      for (int j = 0; j < 3; ++j) {
        std::swap(a[col][j], a[pivot_row][j]);
        std::swap(b[col][j], b[pivot_row][j]);
      }
    }

    // Scale pivot row
    double diag = a[col][col];
    for (int j = 0; j < 3; ++j) {
      a[col][j] /= diag;
      b[col][j] /= diag;
    }

    // Eliminate column
    for (int row = 0; row < 3; ++row) {
      if (row != col) {
        double factor = a[row][col];
        for (int j = 0; j < 3; ++j) {
          a[row][j] -= factor * a[col][j];
          b[row][j] -= factor * b[col][j];
        }
      }
    }
  }

  // Copy result (row-major)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      X[i * 3 + j] = b[i][j];
    }
  }
}

// Sum virial components from per-atom data
static __global__ void gpu_sum_virial(
  int N,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_s)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_s[1024];
  double s = 0.0;

  switch (bid) {
    case 0:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_sxx[n];
      }
      break;
    case 1:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_sxy[n];
      }
      break;
    case 2:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_sxz[n];
      }
      break;
    case 3:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_syx[n];
      }
      break;
    case 4:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_syy[n];
      }
      break;
    case 5:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_syz[n];
      }
      break;
    case 6:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_szx[n];
      }
      break;
    case 7:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_szy[n];
      }
      break;
    case 8:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          s += g_szz[n];
      }
      break;
  }
  s_s[tid] = s;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_s[tid] += s_s[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_s[bid] = s_s[0];
  }
}

// Kernel to transform forces: force_transformed = force @ deform_grad
__global__ void transform_forces_kernel(
  const double* force_per_atom, double* force_transformed, const double* d_deform_grad, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    // Read force components
    double fx = force_per_atom[idx];
    double fy = force_per_atom[idx + N];
    double fz = force_per_atom[idx + 2 * N];

    // Read deformation gradient (row-major)
    double d00 = d_deform_grad[0], d01 = d_deform_grad[1], d02 = d_deform_grad[2];
    double d10 = d_deform_grad[3], d11 = d_deform_grad[4], d12 = d_deform_grad[5];
    double d20 = d_deform_grad[6], d21 = d_deform_grad[7], d22 = d_deform_grad[8];

    // Transform: f_new = f @ deform_grad
    force_transformed[idx] = fx * d00 + fy * d01 + fz * d02;
    force_transformed[idx + N] = fx * d10 + fy * d11 + fz * d12;
    force_transformed[idx + 2 * N] = fx * d20 + fy * d21 + fz * d22;
  }
}

// Kernel to update positions in unstrained coordinates
__global__ void update_positions_unstrained_kernel(double* pos, const double* dr, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    // dr is already in unstrained coordinates, just add
    pos[idx] += dr[idx];
    pos[N + idx] += dr[N + idx];
    pos[2 * N + idx] += dr[2 * N + idx];
  }
}

// Kernel to transform positions: pos_strained = pos_unstrained @ (I + deform)
__global__ void transform_positions_kernel(double* pos, const double* d_deform, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    double x = pos[idx];
    double y = pos[N + idx];
    double z = pos[2 * N + idx];

    // (I + deform) transformation
    double d00 = 1.0 + d_deform[0], d01 = d_deform[1], d02 = d_deform[2];
    double d10 = d_deform[3], d11 = 1.0 + d_deform[4], d12 = d_deform[5];
    double d20 = d_deform[6], d21 = d_deform[7], d22 = 1.0 + d_deform[8];

    pos[idx] = x * d00 + y * d01 + z * d02;
    pos[N + idx] = x * d10 + y * d11 + z * d12;
    pos[2 * N + idx] = x * d20 + y * d21 + z * d22;
  }
}

// Kernel for velocity update: v_new = (1 - a) * v + a * |v| / |f| * f
__global__ void
update_velocity_kernel(double* v, const double* f, double a, double v_norm, double f_norm, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    v[idx] = (1.0 - a) * v[idx] + a * (v_norm / f_norm) * f[idx];
  }
}

// Kernel for velocity update after acceleration: v += dt * f
__global__ void accelerate_velocity_kernel(double* v, const double* f, double dt, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    v[idx] += dt * f[idx];
  }
}

// Kernel for computing dr = dt * v
__global__ void compute_dr_kernel(const double* v, double* dr, double dt, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dr[idx] = dt * v[idx];
  }
}

// GPU reduction kernels
__global__ void gpu_dot_product_kernel(const double* a, const double* b, double* result, int size)
{
  int tid = threadIdx.x;
  int number_of_patches = (size - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  double sum = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 1024;
    if (n < size) {
      sum += a[n] * b[n];
    }
  }
  s_data[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = s_data[0];
  }
}

__global__ void gpu_norm_squared_kernel(const double* a, double* result, int size)
{
  int tid = threadIdx.x;
  int number_of_patches = (size - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  double sum = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 1024;
    if (n < size) {
      sum += a[n] * a[n];
    }
  }
  s_data[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = s_data[0];
  }
}

__global__ void gpu_max_force_kernel(const double* f, double* result, int N)
{
  int tid = threadIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  double max_val = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 1024;
    if (n < N) {
      double fx = f[n];
      double fy = f[n + N];
      double fz = f[n + 2 * N];
      double f_mag_sq = fx * fx + fy * fy + fz * fz;
      if (f_mag_sq > max_val) {
        max_val = f_mag_sq;
      }
    }
  }
  s_data[tid] = max_val;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_data[tid + offset] > s_data[tid]) {
        s_data[tid] = s_data[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *result = s_data[0];
  }
}

} // namespace

void Minimizer_FIRE2::compute(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = number_of_atoms_;
  const int size = N * 3;
  const int ndof = optimize_cell_ ? (size + 9) : size;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;

  // Initialize velocities and working arrays
  GPU_Vector<double> v(ndof, 0.0);
  GPU_Vector<double> dr(ndof);
  GPU_Vector<double> f_extended(ndof);
  GPU_Vector<double> f_transformed(size);
  GPU_Vector<double> virial_tot(9);
  GPU_Vector<double> d_temp(1);
  GPU_Vector<double> pos_unstrained(size); // Store unstrained coordinates
  scalar_pressure_ /= 160.2176621;

  double orig_box[9];
  if (optimize_cell_) {
    for (int i = 0; i < 9; i++) {
      orig_box[i] = box.cpu_h[i];
    }
    cudaMemcpy(
      pos_unstrained.data(),
      position_per_atom.data(),
      size * sizeof(double),
      cudaMemcpyDeviceToDevice);
  }

  // FIRE2 parameters
  dt_ /= 1;
  dtmax_ /= 1;  
  dtmin_ /= 1; //
  double dt = dt_ ;
  double alpha = astart_;
  int Nsteps = 0;
  printf("\ncell factor is %f.\n", cell_factor_);
  printf("\nFIRE2 energy minimization started.\n");
  double cur_pe;
  for (int step = 0; step < number_of_steps_; ++step) {
    // Compute forces and virial
    force.compute(
      box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);

    if (optimize_cell_) {
      // Sum virial tensor
      gpu_sum_virial<<<9, 1024>>>(
        N,
        virial_per_atom.data() + 0 * N,
        virial_per_atom.data() + 3 * N,
        virial_per_atom.data() + 4 * N,
        virial_per_atom.data() + 6 * N,
        virial_per_atom.data() + 1 * N,
        virial_per_atom.data() + 5 * N,
        virial_per_atom.data() + 7 * N,
        virial_per_atom.data() + 8 * N,
        virial_per_atom.data() + 2 * N,
        virial_tot.data());
      GPU_CHECK_KERNEL

      double virial_cpu[9];
      virial_tot.copy_to_host(virial_cpu);
      virial_cpu[1] = virial_cpu[3] = 0.5 * (virial_cpu[1] + virial_cpu[3]);
      virial_cpu[2] = virial_cpu[6] = 0.5 * (virial_cpu[2] + virial_cpu[6]);
      virial_cpu[5] = virial_cpu[7] = 0.5 * (virial_cpu[5] + virial_cpu[7]);
      double volume = box.get_volume();
      double pv = scalar_pressure_ * volume;
      virial_cpu[0] += -pv;
      virial_cpu[4] += -pv;
      virial_cpu[8] += -pv;

      // Compute current deformation gradient
      double cur_deform_grad[9];
      solve_linear_equation_3x3(orig_box, box.cpu_h, cur_deform_grad);
      transpose9(cur_deform_grad);

      // Transform atomic forces
      double* d_deform_grad;
      cudaMalloc(&d_deform_grad, 9 * sizeof(double));
      cudaMemcpy(d_deform_grad, cur_deform_grad, 9 * sizeof(double), cudaMemcpyHostToDevice);

      int threads = 256;
      int blocks = (N + threads - 1) / threads;
      transform_forces_kernel<<<blocks, threads>>>(
        force_per_atom.data(), f_transformed.data(), d_deform_grad, N);
      GPU_CHECK_KERNEL

      // Transform virial
      double virial_t[9];
      memcpy(virial_t, virial_cpu, 9 * sizeof(double));
      transpose9(virial_t);
      double temp[9];
      solve_linear_equation_3x3(cur_deform_grad, virial_t, temp);
      double virial_transformed[9];
      memcpy(virial_transformed, temp, 9 * sizeof(double));
      transpose9(virial_transformed);

      // Apply hydrostatic strain if needed
      if (hydrostatic_strain_) {
        double trace = virial_transformed[0] + virial_transformed[4] + virial_transformed[8];
        for (int i = 0; i < 9; i++)
          virial_transformed[i] = 0.0;
        virial_transformed[0] = trace / 3.0;
        virial_transformed[4] = trace / 3.0;
        virial_transformed[8] = trace / 3.0;
      }

      // Apply const volume if needed
      if (const_volume_) {
        double trace = virial_transformed[0] + virial_transformed[4] + virial_transformed[8];
        virial_transformed[0] -= trace / 3.0;
        virial_transformed[4] -= trace / 3.0;
        virial_transformed[8] -= trace / 3.0;
      }

      // Scale by cell factor
      for (int i = 0; i < 9; i++) {
        virial_transformed[i] /= cell_factor_;
      }

      // Copy to extended force array
      cudaMemcpy(
        f_extended.data(), f_transformed.data(), size * sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(
        f_extended.data() + size, virial_transformed, 9 * sizeof(double), cudaMemcpyHostToDevice);

      cudaFree(d_deform_grad);
    } else {
      // Just copy atomic forces
      cudaMemcpy(
        f_extended.data(), force_per_atom.data(), size * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    // Calculate maximum force
    gpu_max_force_kernel<<<1, 1024>>>(
      f_extended.data(), d_temp.data(), optimize_cell_ ? (N + 3) : N);
    GPU_CHECK_KERNEL
    double fmax_sq;
    d_temp.copy_to_host(&fmax_sq);
    double fmax = sqrt(fmax_sq);

    calculate_total_potential(potential_per_atom);
    double energy = cpu_total_potential_[0];
    if (optimize_cell_) {
      energy += scalar_pressure_ * box.get_volume();
    }

    if (step == 0) {
      cur_pe = energy;
    } else {

        double delta = energy - cur_pe;
        
        if ((delta > 0.1)) {
          break;
        } else {
          cur_pe = energy;
        }
      
    }

    // Print progress
    if (step == 0 || (step + 1) % base == 0 || fmax < force_tolerance_) {
      double pressure = 0.0;
      if (optimize_cell_) {
        double virial_cpu[9];
        virial_tot.copy_to_host(virial_cpu);
        pressure =
          (virial_cpu[0] + virial_cpu[4] + virial_cpu[8]) / 3.0 / box.get_volume() * 160.2176621;
      }
      printf(
        "    step %d: E = %.10f eV, fmax = %.10f eV/A", step == 0 ? 0 : (step + 1), energy, fmax);
      if (optimize_cell_) {
        printf(", P = %.6f GPa", pressure);
      }
      if (const_volume_) {
        printf(", volume = %.6f A^3", box.get_volume());
      }
      printf("\n");

      if (fmax < force_tolerance_) {
        printf("  Converged! fmax < %.2e\n", force_tolerance_);
        break;
      }
    }

    // FIRE2 algorithm
    // Compute P = v · f
    gpu_dot_product_kernel<<<1, 1024>>>(v.data(), f_extended.data(), d_temp.data(), ndof);
    GPU_CHECK_KERNEL
    double P;
    d_temp.copy_to_host(&P);

    if (P > 0.0) {
      Nsteps++;
      if (Nsteps > Nmin_) {
        dt = std::min(dt * finc_, dtmax_);
        alpha *= fa_;
      }
    } else {
      Nsteps = 0;
      dt = std::max(dt * fdec_, dtmin_);
      alpha = astart_;

      // Step back: pos -= 0.5 * dt * v
      compute_dr_kernel<<<(ndof + 255) / 256, 256>>>(v.data(), dr.data(), -0.5 * dt, ndof);
      GPU_CHECK_KERNEL

      int threads = 256;
      int blocks = (N + threads - 1) / threads;

      if (optimize_cell_) {
        // Update unstrained positions
        update_positions_unstrained_kernel<<<blocks, threads>>>(
          pos_unstrained.data(), dr.data(), N);
        GPU_CHECK_KERNEL

        // Update deformation gradient
        double dr_cell[9];
        cudaMemcpy(dr_cell, dr.data() + size, 9 * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 9; i++) {
          dr_cell[i] /= cell_factor_;
        }

        double cur_deform_grad[9];
        solve_linear_equation_3x3(orig_box, box.cpu_h, cur_deform_grad);
        transpose9(cur_deform_grad);

        for (int i = 0; i < 9; i++) {
          cur_deform_grad[i] += dr_cell[i];
        }

        // Update box: new_box = orig_box @ (I + deform)
        double deform[9];
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            deform[i * 3 + j] = cur_deform_grad[j * 3 + i] - (i == j ? 1.0 : 0.0);
          }
        }

        double result[9];
        matrix_multiply(orig_box, deform, result);
        for (int i = 0; i < 9; i++) {
          box.cpu_h[i] = orig_box[i] + result[i];
        }
        box.get_inverse();

        // Update actual positions: pos = pos_unstrained @ (I + deform)
        double* d_deform;
        cudaMalloc(&d_deform, 9 * sizeof(double));
        cudaMemcpy(d_deform, deform, 9 * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(
          position_per_atom.data(),
          pos_unstrained.data(),
          size * sizeof(double),
          cudaMemcpyDeviceToDevice);
        transform_positions_kernel<<<blocks, threads>>>(position_per_atom.data(), d_deform, N);
        GPU_CHECK_KERNEL

        cudaFree(d_deform);
      } else {
        // Simple position update
        update_positions_unstrained_kernel<<<blocks, threads>>>(
          position_per_atom.data(), dr.data(), N);
        GPU_CHECK_KERNEL
      }

      // Recompute forces after step back
      force.compute(
        box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);

      if (optimize_cell_) {
        gpu_sum_virial<<<9, 1024>>>(
          N,
          virial_per_atom.data() + 0 * N,
          virial_per_atom.data() + 3 * N,
          virial_per_atom.data() + 4 * N,
          virial_per_atom.data() + 6 * N,
          virial_per_atom.data() + 1 * N,
          virial_per_atom.data() + 5 * N,
          virial_per_atom.data() + 7 * N,
          virial_per_atom.data() + 8 * N,
          virial_per_atom.data() + 2 * N,
          virial_tot.data());
        GPU_CHECK_KERNEL

        double virial_cpu[9];
        virial_tot.copy_to_host(virial_cpu);
        virial_cpu[1] = virial_cpu[3] = 0.5 * (virial_cpu[1] + virial_cpu[3]);
        virial_cpu[2] = virial_cpu[6] = 0.5 * (virial_cpu[2] + virial_cpu[6]);
        virial_cpu[5] = virial_cpu[7] = 0.5 * (virial_cpu[5] + virial_cpu[7]);
        double volume = box.get_volume();
        double pv = scalar_pressure_ * volume;
        virial_cpu[0] += -pv;
        virial_cpu[4] += -pv;
        virial_cpu[8] += -pv;

        double cur_deform_grad[9];
        solve_linear_equation_3x3(orig_box, box.cpu_h, cur_deform_grad);
        transpose9(cur_deform_grad);

        double* d_deform_grad;
        cudaMalloc(&d_deform_grad, 9 * sizeof(double));
        cudaMemcpy(d_deform_grad, cur_deform_grad, 9 * sizeof(double), cudaMemcpyHostToDevice);

        transform_forces_kernel<<<blocks, threads>>>(
          force_per_atom.data(), f_transformed.data(), d_deform_grad, N);
        GPU_CHECK_KERNEL

        double virial_t[9];
        memcpy(virial_t, virial_cpu, 9 * sizeof(double));
        transpose9(virial_t);
        double temp[9];
        solve_linear_equation_3x3(cur_deform_grad, virial_t, temp);
        double virial_transformed[9];
        memcpy(virial_transformed, temp, 9 * sizeof(double));
        transpose9(virial_transformed);

        if (hydrostatic_strain_) {
          double trace = virial_transformed[0] + virial_transformed[4] + virial_transformed[8];
          for (int i = 0; i < 9; i++)
            virial_transformed[i] = 0.0;
          virial_transformed[0] = trace / 3.0;
          virial_transformed[4] = trace / 3.0;
          virial_transformed[8] = trace / 3.0;
        }

        for (int i = 0; i < 9; i++) {
          virial_transformed[i] /= cell_factor_;
        }

        cudaMemcpy(
          f_extended.data(), f_transformed.data(), size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(
          f_extended.data() + size, virial_transformed, 9 * sizeof(double), cudaMemcpyHostToDevice);

        cudaFree(d_deform_grad);
      } else {
        cudaMemcpy(
          f_extended.data(),
          force_per_atom.data(),
          size * sizeof(double),
          cudaMemcpyDeviceToDevice);
      }

      v.fill(0.0);
      // NO continue here
    }
    // v += dt * f
    accelerate_velocity_kernel<<<(ndof + 255) / 256, 256>>>(v.data(), f_extended.data(), dt, ndof);
    GPU_CHECK_KERNEL

    if (!use_abc_) {
      // Mix velocity: v = (1 - alpha) * v + alpha * |v|/|f| * f
      gpu_norm_squared_kernel<<<1, 1024>>>(v.data(), d_temp.data(), ndof);
      GPU_CHECK_KERNEL
      double v_norm_sq;
      d_temp.copy_to_host(&v_norm_sq);
      double v_norm = sqrt(v_norm_sq);

      gpu_norm_squared_kernel<<<1, 1024>>>(f_extended.data(), d_temp.data(), ndof);
      GPU_CHECK_KERNEL
      double f_norm_sq;
      d_temp.copy_to_host(&f_norm_sq);
      double f_norm = sqrt(f_norm_sq);

      if (f_norm > 1e-10) {
        update_velocity_kernel<<<(ndof + 255) / 256, 256>>>(
          v.data(), f_extended.data(), alpha, v_norm, f_norm, ndof);
        GPU_CHECK_KERNEL
      }
    } else {
      alpha = std::max(alpha, 1e-10);
      double abc_multiplier = 1.0 / (1.0 - std::pow(1.0 - alpha, (Nsteps + 1)));
      // Mix velocity: v = (1 - alpha) * v + alpha * |v|/|f| * f
      gpu_norm_squared_kernel<<<1, 1024>>>(v.data(), d_temp.data(), ndof);
      GPU_CHECK_KERNEL
      double v_norm_sq;
      d_temp.copy_to_host(&v_norm_sq);
      double v_norm = sqrt(v_norm_sq);

      gpu_norm_squared_kernel<<<1, 1024>>>(f_extended.data(), d_temp.data(), ndof);
      GPU_CHECK_KERNEL
      double f_norm_sq;
      d_temp.copy_to_host(&f_norm_sq);
      double f_norm = sqrt(f_norm_sq);

      if (f_norm > 1e-10) {
        update_velocity_kernel<<<(ndof + 255) / 256, 256>>>(
          v.data(), f_extended.data(), alpha, v_norm, f_norm, ndof);
        GPU_CHECK_KERNEL
      }
      int threads = 256;
      int blocks = (ndof + threads - 1) / threads;
      gpu_multiply<<<blocks, threads>>>(ndof, abc_multiplier, v.data(), v.data());
      GPU_CHECK_KERNEL
      filter_v<<<blocks, threads>>>(ndof, dt, maxstep_, v.data());
      GPU_CHECK_KERNEL
    }

    // Compute displacement: dr = dt * v
    compute_dr_kernel<<<(ndof + 255) / 256, 256>>>(v.data(), dr.data(), dt, ndof);
    GPU_CHECK_KERNEL
    if (!use_abc_) {
      // Cap displacement if needed
      gpu_norm_squared_kernel<<<1, 1024>>>(dr.data(), d_temp.data(), ndof);
      GPU_CHECK_KERNEL
      double dr_norm_sq;
      d_temp.copy_to_host(&dr_norm_sq);
      double dr_norm = sqrt(dr_norm_sq);

      if (dr_norm > maxstep_) {
        double scale = maxstep_ / dr_norm;
        int threads = 256;
        int blocks = (ndof + threads - 1) / threads;
        gpu_multiply<<<blocks, threads>>>(ndof, scale, dr.data(), dr.data());
        GPU_CHECK_KERNEL
      }
    }

    // Update positions and box
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (optimize_cell_) {
      // Update unstrained positions
      update_positions_unstrained_kernel<<<blocks, threads>>>(pos_unstrained.data(), dr.data(), N);
      GPU_CHECK_KERNEL

      // Update deformation gradient
      double dr_cell[9];
      cudaMemcpy(dr_cell, dr.data() + size, 9 * sizeof(double), cudaMemcpyDeviceToHost);
      for (int i = 0; i < 9; i++) {
        dr_cell[i] /= cell_factor_;
      }

      double cur_deform_grad[9];
      solve_linear_equation_3x3(orig_box, box.cpu_h, cur_deform_grad);
      transpose9(cur_deform_grad);

      for (int i = 0; i < 9; i++) {
        cur_deform_grad[i] += dr_cell[i];
      }

      // Update box: new_box = orig_box @ (I + deform)
      double deform[9];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          deform[i * 3 + j] = cur_deform_grad[j * 3 + i] - (i == j ? 1.0 : 0.0);
        }
      }

      double result[9];
      matrix_multiply(orig_box, deform, result);
      for (int i = 0; i < 9; i++) {
        box.cpu_h[i] = orig_box[i] + result[i];
      }
      box.get_inverse();

      // Update actual positions: pos = pos_unstrained @ (I + deform)
      double* d_deform;
      cudaMalloc(&d_deform, 9 * sizeof(double));
      cudaMemcpy(d_deform, deform, 9 * sizeof(double), cudaMemcpyHostToDevice);

      cudaMemcpy(
        position_per_atom.data(),
        pos_unstrained.data(),
        size * sizeof(double),
        cudaMemcpyDeviceToDevice);
      transform_positions_kernel<<<blocks, threads>>>(position_per_atom.data(), d_deform, N);
      GPU_CHECK_KERNEL

      cudaFree(d_deform);
    } else {
      // Simple position update
      update_positions_unstrained_kernel<<<blocks, threads>>>(
        position_per_atom.data(), dr.data(), N);
      GPU_CHECK_KERNEL
    }
  }

  printf("FIRE2 energy minimization finished.\n\n");
}