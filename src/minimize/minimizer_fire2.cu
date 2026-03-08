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

// ---------------------------------------------------------------------------
// Small GPU kernels
// ---------------------------------------------------------------------------

__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

// Filter atomic velocities: limit displacement per step to maxstep (Angstrom).
// Only operates on the first `atom_size` elements (atomic DOF).
__global__ void
filter_v_atoms(const int atom_size, const double dt, const double maxstep, double* v)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < atom_size) {
    double disp = v[n] * dt;
    if (fabs(disp) > maxstep) {
      v[n] = copysign(maxstep / dt, v[n]);
    }
  }
}

// Filter cell velocities: limit strain increment per step to max_strain_step (dimensionless).
// Operates on the 9 cell DOF starting at offset `atom_size`.
// This is the key stability guard – analogous to LAMMPS vmax in box/relax.
__global__ void
filter_v_cell(const int atom_size, const double dt, const double max_strain_step, double* v)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < 9) {
    int idx = atom_size + n;
    double disp = v[idx] * dt;
    if (fabs(disp) > max_strain_step) {
      v[idx] = copysign(max_strain_step / dt, v[idx]);
    }
  }
}

// ---------------------------------------------------------------------------
// CPU helper: 3x3 matrix utilities (row-major storage)
// ---------------------------------------------------------------------------

static void transpose9(double* m)
{
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      std::swap(m[i * 3 + j], m[j * 3 + i]);
}

static void matrix_multiply(const double* A, const double* B, double* C)
{
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      C[i * 3 + j] = 0.0;
      for (int k = 0; k < 3; ++k)
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    }
}

// Solve A * X = B for 3x3 matrices using Gaussian elimination with partial pivoting.
static void solve_linear_equation_3x3(const double* A, const double* B, double* X)
{
  double a[3][3], b[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a[i][j] = A[i * 3 + j];
      b[i][j] = B[i * 3 + j];
    }

  for (int col = 0; col < 3; ++col) {
    int pivot_row = col;
    for (int i = col + 1; i < 3; ++i)
      if (fabs(a[i][col]) > fabs(a[pivot_row][col]))
        pivot_row = i;

    if (fabs(a[pivot_row][col]) < 1e-9) {
      printf("Matrix is singular or nearly singular!\n");
      return;
    }

    if (pivot_row != col)
      for (int j = 0; j < 3; ++j) {
        std::swap(a[col][j], a[pivot_row][j]);
        std::swap(b[col][j], b[pivot_row][j]);
      }

    double diag = a[col][col];
    for (int j = 0; j < 3; ++j) {
      a[col][j] /= diag;
      b[col][j] /= diag;
    }

    for (int row = 0; row < 3; ++row)
      if (row != col) {
        double factor = a[row][col];
        for (int j = 0; j < 3; ++j) {
          a[row][j] -= factor * a[col][j];
          b[row][j] -= factor * b[col][j];
        }
      }
  }

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      X[i * 3 + j] = b[i][j];
}

// ---------------------------------------------------------------------------
// GPU reduction kernels
// ---------------------------------------------------------------------------

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
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_sxx[n];
      }
      break;
    case 1:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_sxy[n];
      }
      break;
    case 2:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_sxz[n];
      }
      break;
    case 3:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_syx[n];
      }
      break;
    case 4:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_syy[n];
      }
      break;
    case 5:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_syz[n];
      }
      break;
    case 6:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_szx[n];
      }
      break;
    case 7:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_szy[n];
      }
      break;
    case 8:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_szz[n];
      }
      break;
  }
  s_s[tid] = s;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_s[tid] += s_s[tid + offset];
    __syncthreads();
  }

  if (tid == 0)
    g_s[bid] = s_s[0];
}

__global__ void gpu_dot_product_kernel(const double* a, const double* b, double* result, int size)
{
  int tid = threadIdx.x;
  int number_of_patches = (size - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  double sum = 0.0;
  for (int p = 0; p < number_of_patches; ++p) {
    int n = tid + p * 1024;
    if (n < size)
      sum += a[n] * b[n];
  }
  s_data[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    *result = s_data[0];
}

// Dot product of only the 9 cell DOF (offset = atom_size).
// Used to check if cell velocity and cell force are aligned independently.
__global__ void
gpu_dot_product_cell_kernel(const double* v, const double* f, double* result, int atom_size)
{
  int tid = threadIdx.x;
  __shared__ double s_data[1024];
  double sum = (tid < 9) ? v[atom_size + tid] * f[atom_size + tid] : 0.0;
  s_data[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    *result = s_data[0];
}

__global__ void gpu_norm_squared_kernel(const double* a, double* result, int size)
{
  int tid = threadIdx.x;
  int number_of_patches = (size - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  double sum = 0.0;
  for (int p = 0; p < number_of_patches; ++p) {
    int n = tid + p * 1024;
    if (n < size)
      sum += a[n] * a[n];
  }
  s_data[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    *result = s_data[0];
}

__global__ void gpu_max_force_kernel(const double* f, double* result, int N)
{
  int tid = threadIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  double max_val = 0.0;
  for (int p = 0; p < number_of_patches; ++p) {
    int n = tid + p * 1024;
    if (n < N) {
      double fx = f[n], fy = f[n + N], fz = f[n + 2 * N];
      double f2 = fx * fx + fy * fy + fz * fz;
      if (f2 > max_val)
        max_val = f2;
    }
  }
  s_data[tid] = max_val;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset && s_data[tid + offset] > s_data[tid])
      s_data[tid] = s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    *result = s_data[0];
}

// ---------------------------------------------------------------------------
// Position / velocity kernels
// ---------------------------------------------------------------------------

__global__ void update_positions_unstrained_kernel(double* pos, const double* dr, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    pos[idx] += dr[idx];
    pos[N + idx] += dr[N + idx];
    pos[2 * N + idx] += dr[2 * N + idx];
  }
}

// pos_strained = pos_unstrained @ (I + deform)
__global__ void transform_positions_kernel(double* pos, const double* d_deform, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    double x = pos[idx], y = pos[N + idx], z = pos[2 * N + idx];
    double d00 = 1.0 + d_deform[0], d01 = d_deform[1], d02 = d_deform[2];
    double d10 = d_deform[3], d11 = 1.0 + d_deform[4], d12 = d_deform[5];
    double d20 = d_deform[6], d21 = d_deform[7], d22 = 1.0 + d_deform[8];
    pos[idx] = x * d00 + y * d01 + z * d02;
    pos[N + idx] = x * d10 + y * d11 + z * d12;
    pos[2 * N + idx] = x * d20 + y * d21 + z * d22;
  }
}

// f_transformed = force @ deform_grad
__global__ void transform_forces_kernel(
  const double* force_per_atom, double* force_transformed, const double* d_deform_grad, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    double fx = force_per_atom[idx];
    double fy = force_per_atom[idx + N];
    double fz = force_per_atom[idx + 2 * N];
    double d00 = d_deform_grad[0], d01 = d_deform_grad[1], d02 = d_deform_grad[2];
    double d10 = d_deform_grad[3], d11 = d_deform_grad[4], d12 = d_deform_grad[5];
    double d20 = d_deform_grad[6], d21 = d_deform_grad[7], d22 = d_deform_grad[8];
    force_transformed[idx] = fx * d00 + fy * d01 + fz * d02;
    force_transformed[idx + N] = fx * d10 + fy * d11 + fz * d12;
    force_transformed[idx + 2 * N] = fx * d20 + fy * d21 + fz * d22;
  }
}

// v_new = (1 - a) * v + a * |v|/|f| * f
__global__ void
update_velocity_kernel(double* v, const double* f, double a, double v_norm, double f_norm, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    v[idx] = (1.0 - a) * v[idx] + a * (v_norm / f_norm) * f[idx];
}

// v += dt * f
__global__ void accelerate_velocity_kernel(double* v, const double* f, double dt, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    v[idx] += dt * f[idx];
}

// dr = dt * v
__global__ void compute_dr_kernel(const double* v, double* dr, double dt, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    dr[idx] = dt * v[idx];
}

// ---------------------------------------------------------------------------
// CPU helper: update box and positions from dr_cell and dr_atoms
// ---------------------------------------------------------------------------
//
// Given the current box, the cell displacement dr_cell (9 components, raw from
// the velocity vector, already divided by cell_factor outside), and the
// unstrained positions, this function:
//   1. Computes the new box
//   2. Updates the strained (actual) positions
//   3. Uploads the deform tensor to the GPU pointer d_deform (caller allocates)
//
static void apply_cell_update(
  Box& box,
  const double* orig_box,
  const double* dr_cell, // 9-element deform_grad increment (CPU)
  GPU_Vector<double>& position_per_atom,
  const GPU_Vector<double>& pos_unstrained,
  int N,
  double* d_deform_gpu) // pre-allocated device buffer (9 doubles)
{
  // Current deformation gradient: box = orig_box @ F^T  =>  F = solve(orig_box, box)^T
  double cur_deform_grad[9];
  solve_linear_equation_3x3(orig_box, box.cpu_h, cur_deform_grad);
  transpose9(cur_deform_grad);

  // Accumulate increment
  for (int i = 0; i < 9; i++)
    cur_deform_grad[i] += dr_cell[i];

  // deform = F - I  (row-major, with transposition to match box convention)
  double deform[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      deform[i * 3 + j] = cur_deform_grad[j * 3 + i] - (i == j ? 1.0 : 0.0);

  // New box: box = orig_box + orig_box @ deform
  double result[9];
  matrix_multiply(orig_box, deform, result);
  for (int i = 0; i < 9; i++)
    box.cpu_h[i] = orig_box[i] + result[i];
  box.get_inverse();

  // Upload deform to GPU and transform positions
  cudaMemcpy(d_deform_gpu, deform, 9 * sizeof(double), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  cudaMemcpy(
    position_per_atom.data(),
    pos_unstrained.data(),
    N * 3 * sizeof(double),
    cudaMemcpyDeviceToDevice);
  transform_positions_kernel<<<blocks, threads>>>(position_per_atom.data(), d_deform_gpu, N);
  GPU_CHECK_KERNEL
}

// ---------------------------------------------------------------------------
// CPU helper: compute extended force vector (atomic forces + virial-based cell
// forces) and store into f_extended.
// Also returns the current pressure for printing.
// ---------------------------------------------------------------------------
static double compute_extended_forces(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& virial_tot,
  GPU_Vector<double>& f_transformed,
  GPU_Vector<double>& f_extended,
  const double* orig_box,
  int N,
  int size,
  bool optimize_cell,
  bool hydrostatic_strain,
  double cell_factor)
{
  // Compute forces
  force.compute(
    box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);

  if (!optimize_cell) {
    cudaMemcpy(
      f_extended.data(), force_per_atom.data(), size * sizeof(double), cudaMemcpyDeviceToDevice);
    return 0.0;
  }

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
  // Symmetrise off-diagonal pairs
  virial_cpu[1] = virial_cpu[3] = 0.5 * (virial_cpu[1] + virial_cpu[3]);
  virial_cpu[2] = virial_cpu[6] = 0.5 * (virial_cpu[2] + virial_cpu[6]);
  virial_cpu[5] = virial_cpu[7] = 0.5 * (virial_cpu[5] + virial_cpu[7]);

  double pressure =
    (virial_cpu[0] + virial_cpu[4] + virial_cpu[8]) / 3.0 / box.get_volume() * 160.2176621;

  // Compute current deformation gradient
  double cur_deform_grad[9];
  solve_linear_equation_3x3(orig_box, box.cpu_h, cur_deform_grad);
  transpose9(cur_deform_grad);

  // Transform atomic forces: f_transformed = f @ F
  double* d_deform_grad;
  cudaMalloc(&d_deform_grad, 9 * sizeof(double));
  cudaMemcpy(d_deform_grad, cur_deform_grad, 9 * sizeof(double), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  transform_forces_kernel<<<blocks, threads>>>(
    force_per_atom.data(), f_transformed.data(), d_deform_grad, N);
  GPU_CHECK_KERNEL

  // Transform virial: virial_transformed = solve(F, virial^T)^T / cell_factor
  double virial_t[9];
  memcpy(virial_t, virial_cpu, 9 * sizeof(double));
  transpose9(virial_t);
  double temp[9];
  solve_linear_equation_3x3(cur_deform_grad, virial_t, temp);
  double virial_transformed[9];
  memcpy(virial_transformed, temp, 9 * sizeof(double));
  transpose9(virial_transformed);

  if (hydrostatic_strain) {
    double trace = virial_transformed[0] + virial_transformed[4] + virial_transformed[8];
    for (int i = 0; i < 9; i++)
      virial_transformed[i] = 0.0;
    virial_transformed[0] = virial_transformed[4] = virial_transformed[8] = trace / 3.0;
  } else {
    // Noise filter for off-diagonal (shear) cell force components.
    //
    // For high-symmetry structures (e.g. BCC, FCC with small unit cells),
    // the true shear stress is zero by symmetry, but floating-point summation
    // produces tiny non-zero off-diagonal virial values. Once divided by the
    // small cell_factor (= N = 2 for a 2-atom cell), these noise values become
    // comparable to or larger than the residual diagonal forces. FIRE2 then
    // drives velocity in these flat/noisy shear directions indefinitely,
    // accumulating kinetic energy that eventually causes explosion.
    //
    // Strategy: compute the RMS of diagonal cell forces. If an off-diagonal
    // component is smaller than a relative threshold (off_diag_tol) times the
    // diagonal RMS, treat it as noise and zero it. This is safe because:
    //   - Far from convergence the diagonal forces dominate; the filter changes
    //     nothing meaningful.
    //   - Near convergence on a symmetric structure, off-diagonal forces are
    //     pure noise; zeroing them stabilises the final approach.
    //   - For genuinely asymmetric structures the off-diagonal forces are large
    //     relative to diagonal; the filter leaves them untouched.
    //
    // The threshold 0.01 (1%) is conservative: it only suppresses components
    // that are negligible compared to the dominant driving force.
    const double off_diag_tol = 0.01;
    double diag_rms = std::sqrt(
      (virial_transformed[0] * virial_transformed[0] +
       virial_transformed[4] * virial_transformed[4] +
       virial_transformed[8] * virial_transformed[8]) /
      3.0);
    // Off-diagonal indices in row-major 3x3: (0,1)=1, (0,2)=2, (1,0)=3,
    //                                         (1,2)=5, (2,0)=6, (2,1)=7
    const int off_diag[6] = {1, 2, 3, 5, 6, 7};
    for (int k = 0; k < 6; k++) {
      int idx = off_diag[k];
      if (std::fabs(virial_transformed[idx]) < off_diag_tol * diag_rms) {
        virial_transformed[idx] = 0.0;
      }
    }
  }

  for (int i = 0; i < 9; i++)
    virial_transformed[i] /= cell_factor;

  // Assemble extended force vector
  cudaMemcpy(
    f_extended.data(), f_transformed.data(), size * sizeof(double), cudaMemcpyDeviceToDevice);
  cudaMemcpy(
    f_extended.data() + size, virial_transformed, 9 * sizeof(double), cudaMemcpyHostToDevice);

  cudaFree(d_deform_grad);
  return pressure;
}

} // namespace

// ---------------------------------------------------------------------------
// Main FIRE2 compute function
// ---------------------------------------------------------------------------

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
  const int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;

  // Working arrays
  GPU_Vector<double> v(ndof, 0.0);
  GPU_Vector<double> dr(ndof);
  GPU_Vector<double> f_extended(ndof);
  GPU_Vector<double> f_transformed(size);
  GPU_Vector<double> virial_tot(9);
  GPU_Vector<double> d_temp(1);
  GPU_Vector<double> pos_unstrained(size);

  // Best-state checkpoint: save the configuration with the lowest fmax seen
  // so far. When an explosion is detected (fmax >> fmax_best), we restore
  // from here and restart with a smaller dt, rather than doing a local
  // half-step retrace from an already-bad position.
  GPU_Vector<double> pos_best(size);            // best strained positions
  GPU_Vector<double> pos_unstrained_best(size); // best unstrained positions (cell opt only)
  double box_best[9] = {};                      // best box matrix
  double fmax_best = 1e30;                      // fmax at checkpoint
  double dtmax_at_best_save = 1e30;             // dtmax when checkpoint was saved
  // Explosion is declared when fmax > explosion_factor * fmax_best.
  // Factor 20: large enough to ignore normal fluctuations, small enough to
  // catch a real structural blowup before it propagates too far.
  const double explosion_factor = 20.0;

  // Consecutive explosion counter: if we keep exploding from the same
  // checkpoint, that checkpoint may itself be an unstable saddle point.
  // After 3 explosions, enter "crawl mode": force alpha high and Nmin large
  // so FIRE2 can only take very small, well-damped steps.
  int explosion_count = 0;
  bool crawl_mode = false;

  // Pre-allocate device buffer for deform tensor (reused throughout)
  double* d_deform_buf = nullptr;
  if (optimize_cell_) {
    cudaMalloc(&d_deform_buf, 9 * sizeof(double));
  }

  double orig_box[9] = {};
  if (optimize_cell_) {
    for (int i = 0; i < 9; i++)
      orig_box[i] = box.cpu_h[i];
    cudaMemcpy(
      pos_unstrained.data(),
      position_per_atom.data(),
      size * sizeof(double),
      cudaMemcpyDeviceToDevice);
  }

  // FIRE2 running state

  double dt = dt_0_;
  double alpha = astart_;
  int Nsteps = 0;
  double fmax_prev = 1e30; // tracks previous fmax for growth detection

  // For very small systems (N <= 9 atoms with cell opt), the 9 cell DOF equal
  // or outnumber the atomic DOF. The energy surface is nearly flat in many
  // cell directions and the integrator becomes extremely sensitive to step size.
  // Cap dtmax tighter so the system cannot escape its basin even when FIRE2
  // wants to keep increasing dt.
  if (optimize_cell_ && N <= 9) {
    dtmax_ = std::min(dtmax_, 0.05);
    printf("    [FIRE2] Small system (N=%d) with cell opt: dtmax capped at %.4f\n", N, dtmax_);
  }

  printf("\nEnergy minimization with changed box started.\n");

  const int atom_blocks = (size + 255) / 256;
  const int ndof_blocks = (ndof + 255) / 256;

  // Energy window for plateau-convergence detection (see criterion (b) below)
  const int win_size = 10;
  std::vector<double> energy_window(win_size, 0.0);

  for (int step = 0; step < number_of_steps_; ++step) {

    // ------------------------------------------------------------------
    // 1. Compute forces (and virial-based cell forces when optimize_cell_)
    // ------------------------------------------------------------------
    double pressure = compute_extended_forces(
      force,
      box,
      position_per_atom,
      type,
      group,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      virial_tot,
      f_transformed,
      f_extended,
      orig_box,
      N,
      size,
      optimize_cell_,
      hydrostatic_strain_,
      cell_factor_);

    gpu_max_force_kernel<<<1, 1024>>>(
      f_extended.data(), d_temp.data(), optimize_cell_ ? (N + 3) : N);
    GPU_CHECK_KERNEL
    double fmax_sq;
    d_temp.copy_to_host(&fmax_sq);
    double fmax = sqrt(fmax_sq);

    // When optimising the cell, also include the 9 cell-force components in
    // the fmax criterion. The gpu_max_force_kernel only scans atomic DOF
    // (triplets f[n], f[n+N], f[n+2N]), so cell forces at f_extended[size..]
    // are completely invisible to it. This means convergence can be declared
    // while residual cell forces are still large, OR the checkpoint fmax_best
    // can be saved at a state where cell forces are actually noisy/large.
    // Fix: copy the 9 cell-force values to CPU and take their absolute maximum,
    // then fold it into fmax via hypot-style max(fmax, max_cell_force).
    if (optimize_cell_) {
      double cell_forces[9];
      cudaMemcpy(
        cell_forces, f_extended.data() + size, 9 * sizeof(double), cudaMemcpyDeviceToHost);
      double max_cell_f = 0.0;
      for (int i = 0; i < 9; i++)
        max_cell_f = std::max(max_cell_f, std::fabs(cell_forces[i]));
      fmax = std::max(fmax, max_cell_f);
    }

    calculate_total_potential(potential_per_atom);
    double energy = cpu_total_potential_[0];
    if (step == 0 || (step + 1) % base == 0 || fmax < force_tolerance_) {
      printf(
        "    step %d: E = %.10f eV, fmax = %.10f eV/A", step == 0 ? 0 : (step + 1), energy, fmax);
      if (optimize_cell_)
        printf(", P = %.6f GPa", pressure);
      printf("\n");
    }

    if (fmax < force_tolerance_) {
      printf("  Converged! fmax = %.2e < ftol = %.2e\n", fmax, force_tolerance_);
      break;
    }

    // ------------------------------------------------------------------
    // 1b. Checkpoint: save best-seen configuration for explosion recovery.
    //
    //     We save whenever fmax improves by more than 1% over the stored
    //     best. We also record dtmax at save time: a checkpoint found under
    //     a large dtmax may be an unstable saddle that can't be re-approached
    //     with the same step size. Recording it lets the explosion handler
    //     decide whether to trust the checkpoint or reduce dt further.
    //     Finding a new best always resets the explosion counter.
    // ------------------------------------------------------------------
    if (fmax < 0.99 * fmax_best) {
      fmax_best = fmax;
      dtmax_at_best_save = dtmax_;
      cudaMemcpy(
        pos_best.data(), position_per_atom.data(), size * sizeof(double),
        cudaMemcpyDeviceToDevice);
      if (optimize_cell_) {
        cudaMemcpy(
          pos_unstrained_best.data(), pos_unstrained.data(), size * sizeof(double),
          cudaMemcpyDeviceToDevice);
        for (int i = 0; i < 9; i++)
          box_best[i] = box.cpu_h[i];
      }
      // Progress made: reset explosion counter and exit crawl mode if we
      // have descended far enough that the new best is clearly stable.
      explosion_count = 0;
      if (crawl_mode && fmax < 0.5 * fmax_best) {
        crawl_mode = false;
      }
    }
    //
    //    Three reset triggers:
    //    (a) P_total <= 0: velocity not aligned with force
    //    (b) P_cell <= 0: cell velocity diverging from cell force
    //    (c) fmax_current > fmax_grow_factor * fmax_prev: forces growing
    //        (indicates integrator instability even if P > 0 momentarily)
    // ------------------------------------------------------------------
    gpu_dot_product_kernel<<<1, 1024>>>(v.data(), f_extended.data(), d_temp.data(), ndof);
    GPU_CHECK_KERNEL
    double P;
    d_temp.copy_to_host(&P);

    double P_cell = 1.0;
    if (optimize_cell_) {
      gpu_dot_product_cell_kernel<<<1, 1024>>>(v.data(), f_extended.data(), d_temp.data(), size);
      GPU_CHECK_KERNEL
      d_temp.copy_to_host(&P_cell);
    }

    // Detect force growth: reset if fmax increased significantly vs last step.
    //
    // Near convergence, fmax fluctuates randomly due to numerical noise; a tight
    // threshold triggers reset-death-loops where dt halves repeatedly and the
    // minimizer can never escape the plateau. We therefore relax the threshold
    // progressively as fmax approaches force_tolerance_:
    //   - Far from convergence (fmax > 100x ftol): strict 1.2x guard
    //   - Mid-range (fmax <= 100x ftol):            relaxed 2.0x guard
    //   - Near convergence (fmax <= 10x ftol):       very relaxed 5.0x guard
    double fmax_grow_factor;
    if (fmax > 100.0 * force_tolerance_) {
      fmax_grow_factor = 1.2;
    } else if (fmax > 10.0 * force_tolerance_) {
      fmax_grow_factor = 2.0;
    } else {
      fmax_grow_factor = 5.0;
    }
    bool fmax_growing = (step > 0) && (fmax > fmax_grow_factor * fmax_prev);
    fmax_prev = fmax;

    const bool going_right = (P > 0.0) && (P_cell > 0.0) && !fmax_growing;

    if (going_right) {
      // ---- All checks pass: increase dt ----
      Nsteps++;
      if (Nsteps > Nmin_) {
        // Near convergence, cap dtmax more aggressively to avoid overshooting
        // the shallow minimum. When forces are tiny, a large dt produces large
        // displacements that skip over the basin floor entirely.
        double dtmax_eff = dtmax_;
        if (fmax < 10.0 * force_tolerance_) {
          dtmax_eff = dtmax_ * 0.2;  // 5x tighter near convergence
        } else if (fmax < 100.0 * force_tolerance_) {
          dtmax_eff = dtmax_ * 0.5;  // 2x tighter in mid-range
        }
        // In crawl mode, dt grows only at 1/4 the normal rate to stay damped.
        double finc_eff = crawl_mode ? (1.0 + (finc_ - 1.0) * 0.25) : finc_;
        dt = std::min(dt * finc_eff, dtmax_eff);
        alpha *= fa_;
      }
    } else {
      // ---- Something going wrong: reset ----
      Nsteps = 0;
      dt = std::max(dt * fdec_, dtmin_);
      alpha = astart_;

      // Explosion check: if fmax has grown far beyond the best-ever value,
      // the structure has left its basin entirely. A half-step retrace from
      // the current (bad) position is futile; restore the best checkpoint
      // instead. We track consecutive explosions: after 3, enter crawl mode
      // (slow dt ramp, high damping) so FIRE2 cannot build up enough velocity
      // to escape the basin again.
      const bool exploded = (fmax_best < 1e29) && (fmax > explosion_factor * fmax_best);

      if (exploded) {
        explosion_count++;

        // Progressive dtmax reduction each time we explode.
        // Never let dtmax fall below dtmin_ * 10 (below that FIRE2 can't move).
        dtmax_ = std::max(dtmax_ * 0.5, dtmin_ * 10.0);
        // Restart dt from min(dt_0_, dtmax_) so it re-ramps from a safe value.
        dt = std::min(dt_0_, dtmax_);

        printf(
          "    [FIRE2] Explosion #%d at step %d (fmax=%.3e >> best=%.3e). "
          "dtmax -> %.5f.\n",
          explosion_count, step, fmax, fmax_best, dtmax_);

        // After 3 explosions from the same checkpoint, enter crawl mode:
        // force high damping (alpha = astart_) and reset Nsteps so dt can
        // only grow very slowly. This keeps FIRE2 well-damped near the
        // unstable region until it finds a better basin.
        if (explosion_count >= 3 && !crawl_mode) {
          crawl_mode = true;
          printf(
            "    [FIRE2] Entering crawl mode: forcing high damping, small steps.\n");
        }

        // Restore best positions
        cudaMemcpy(
          position_per_atom.data(), pos_best.data(), size * sizeof(double),
          cudaMemcpyDeviceToDevice);
        if (optimize_cell_) {
          cudaMemcpy(
            pos_unstrained.data(), pos_unstrained_best.data(), size * sizeof(double),
            cudaMemcpyDeviceToDevice);
          for (int i = 0; i < 9; i++)
            box.cpu_h[i] = box_best[i];
          box.get_inverse();
        }

        // Reset FIRE2 state
        alpha = astart_;
        Nsteps = 0;
        v.fill(0.0);

        // Recompute forces at the restored configuration
        pressure = compute_extended_forces(
          force, box, position_per_atom, type, group,
          potential_per_atom, force_per_atom, virial_per_atom,
          virial_tot, f_transformed, f_extended,
          orig_box, N, size, optimize_cell_, hydrostatic_strain_, cell_factor_);

      } else {
        // Normal reset: step back half a step and recompute

        // Step back: pos -= 0.5 * dt * v  (half-step retrace)
        compute_dr_kernel<<<ndof_blocks, 256>>>(v.data(), dr.data(), -0.5 * dt, ndof);
        GPU_CHECK_KERNEL

        if (optimize_cell_) {
          // Retrace atomic positions (unstrained)
          update_positions_unstrained_kernel<<<atom_blocks, 256>>>(
            pos_unstrained.data(), dr.data(), N);
          GPU_CHECK_KERNEL

          // Retrace cell DOF
          double dr_cell[9];
          cudaMemcpy(dr_cell, dr.data() + size, 9 * sizeof(double), cudaMemcpyDeviceToHost);
          for (int i = 0; i < 9; i++)
            dr_cell[i] /= cell_factor_;

          apply_cell_update(
            box, orig_box, dr_cell, position_per_atom, pos_unstrained, N, d_deform_buf);
        } else {
          update_positions_unstrained_kernel<<<atom_blocks, 256>>>(
            position_per_atom.data(), dr.data(), N);
          GPU_CHECK_KERNEL
        }

        // Recompute forces after step-back
        pressure = compute_extended_forces(
          force, box, position_per_atom, type, group,
          potential_per_atom, force_per_atom, virial_per_atom,
          virial_tot, f_transformed, f_extended,
          orig_box, N, size, optimize_cell_, hydrostatic_strain_, cell_factor_);

        // Zero all velocities after bad step
        v.fill(0.0);
      } // end exploded / normal retrace

    } // end going_right / reset branch

    // ------------------------------------------------------------------
    // 3. Velocity update: v += dt * f
    // ------------------------------------------------------------------
    accelerate_velocity_kernel<<<ndof_blocks, 256>>>(v.data(), f_extended.data(), dt, ndof);
    GPU_CHECK_KERNEL

    // ------------------------------------------------------------------
    // 4. FIRE2 velocity mixing + Nesterov correction
    //
    //    v = (1-alpha)*v + alpha*(|v|/|f|)*f
    //
    //    Guard against |f|->0 near convergence:
    //    An adaptive force floor (f_floor) is used to stabilize the |v|/|f|
    //    ratio. Far from convergence, a moderate floor prevents ratio explosion
    //    if forces are transiently small. Near convergence (fmax < 10x ftol),
    //    the floor is removed entirely so velocity mixing remains accurate and
    //    does not stall on a plateau.
    //
    //    Nesterov abc_multiplier (applied to ATOMIC DOF only) is disabled when
    //    fmax < 50x ftol to prevent repeated velocity amplification when forces
    //    are already tiny, which was the primary cause of convergence plateaus.
    // ------------------------------------------------------------------
    alpha = std::max(alpha, 1e-10);

    // Compute norms with a safety floor
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

    // Only mix if forces are meaningful; skip if essentially zero
    // (convergence already checked above, so we're still running)
    //
    // Adaptive f_floor strategy:
    //   - When far from convergence (fmax > 100x ftol): use a moderate floor to
    //     prevent |v|/|f| explosion if forces are transiently small mid-trajectory.
    //   - When close to convergence (fmax <= 100x ftol): shrink the floor
    //     proportionally to fmax so velocity mixing keeps working accurately.
    //     A frozen floor here would clamp f_norm_eff above the true |f| and
    //     distort the mixing ratio, causing the plateau stall.
    //   - When essentially at convergence (fmax <= 10x ftol): remove the floor
    //     entirely and let the natural force magnitude drive mixing.
    if (f_norm > 1e-30) {
      double f_norm_eff;
      if (fmax > 10.0 * force_tolerance_) {
        // Far from convergence: moderate floor proportional to sqrt(N)
        const double f_floor = 1e-6 * std::sqrt(static_cast<double>(N));
        f_norm_eff = std::max(f_norm, f_floor);
      } else {
        // Near convergence: no floor, trust the actual force norm
        f_norm_eff = f_norm;
      }
      update_velocity_kernel<<<ndof_blocks, 256>>>(
        v.data(), f_extended.data(), alpha, v_norm, f_norm_eff, ndof);
      GPU_CHECK_KERNEL
    } else {
      v.fill(0.0);
    }

    // Nesterov abc_multiplier on ATOMIC DOF only, capped for stability.
    //
    // Nesterov acceleration helps early convergence but is counter-productive
    // near the minimum: when fmax is already small, amplifying velocity by up
    // to abc_mult_max_ repeatedly inflates |v| far above |f|, which then feeds
    // back into a distorted velocity mixing ratio and produces the plateau stall.
    // We therefore disable Nesterov when fmax is within 50x of force_tolerance_.
    if (fmax > 50.0 * force_tolerance_) {
      double abc_raw = 1.0 / (1.0 - std::pow(std::max(1.0 - alpha, 0.0), (Nsteps + 1)));
      double abc_multiplier = std::min(abc_raw, abc_mult_max_);
      if (abc_multiplier > 1.0 + 1e-9) {
        gpu_multiply<<<atom_blocks, 256>>>(size, abc_multiplier, v.data(), v.data());
        GPU_CHECK_KERNEL
      }
    }
    // (Near convergence: skip Nesterov entirely, let FIRE2 mixing handle it)

    // ------------------------------------------------------------------
    // 5. Velocity limiters (second key fix)
    //    - Atomic DOF: max displacement = maxstep_ (Angstrom)
    //    - Cell DOF:   max strain step  = max_strain_step_ (dimensionless)
    //      This is the vmax equivalent from LAMMPS box/relax.
    // ------------------------------------------------------------------
    filter_v_atoms<<<atom_blocks, 256>>>(size, dt, maxstep_, v.data());
    GPU_CHECK_KERNEL

    if (optimize_cell_) {
      filter_v_cell<<<1, 256>>>(size, dt, max_strain_step_, v.data());
      GPU_CHECK_KERNEL
    }

    // ------------------------------------------------------------------
    // 6. Compute displacement dr = dt * v
    // ------------------------------------------------------------------
    compute_dr_kernel<<<ndof_blocks, 256>>>(v.data(), dr.data(), dt, ndof);
    GPU_CHECK_KERNEL

    // ------------------------------------------------------------------
    // 7. Apply displacements
    // ------------------------------------------------------------------
    if (optimize_cell_) {
      update_positions_unstrained_kernel<<<atom_blocks, 256>>>(pos_unstrained.data(), dr.data(), N);
      GPU_CHECK_KERNEL

      double dr_cell[9];
      cudaMemcpy(dr_cell, dr.data() + size, 9 * sizeof(double), cudaMemcpyDeviceToHost);
      for (int i = 0; i < 9; i++)
        dr_cell[i] /= cell_factor_;

      apply_cell_update(box, orig_box, dr_cell, position_per_atom, pos_unstrained, N, d_deform_buf);
    } else {
      update_positions_unstrained_kernel<<<atom_blocks, 256>>>(
        position_per_atom.data(), dr.data(), N);
      GPU_CHECK_KERNEL
    }
  }

  if (optimize_cell_ && d_deform_buf)
    cudaFree(d_deform_buf);

  printf("Energy minimization finished.\n\n");
}
