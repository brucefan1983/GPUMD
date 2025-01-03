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
The FIRE (fast inertial relaxation engine) minimizer
Reference: PhysRevLett 97, 170201 (2006)
           Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire_box_change.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

namespace
{

// get virial in v
__global__ void process_matrix(const double* v, double* processed_v, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < 3) {
    processed_v[idx] = v[N + idx];
    processed_v[idx + 3] = v[N * 2 + 3 + idx];
    processed_v[idx + 6] = v[N * 3 + 6 + idx];
  }
}

// transpose 9-element array
void transpose9(double* matrix)
{
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      std::swap(matrix[i * 3 + j], matrix[j * 3 + i]);
    }
  }
}

// C = A * B, only for 9-element array
void matrix_multiply(const double* A, const double* B, double* C)
{
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i * 3 + j] = 0.0f;
      for (int k = 0; k < 3; ++k) {
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
      }
    }
  }
}

void update_box(double* box, const double* d_v, int N)
{

  double processed_v[9];
  double* d_processed_v;

  cudaMalloc(&d_processed_v, 9 * sizeof(double));
  process_matrix<<<1, 9>>>(d_v, d_processed_v, N);
  cudaMemcpy(processed_v, d_processed_v, 9 * sizeof(double), cudaMemcpyDeviceToHost);
  transpose9(processed_v);

  for (int i = 0; i < 9; i++) {
    processed_v[i] = processed_v[i] / N;
  }

  double result[9];
  matrix_multiply(box, processed_v, result);

  for (int i = 0; i < 9; ++i) {
    box[i] += result[i];
  }

  cudaFree(d_processed_v);
}

__global__ void get_force_temp_kernel(
  const double* force_per_atom, double* force_temp, const double* d_deform_grad, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    // force_per_atom index
    int fx_idx = idx;
    int fy_idx = idx + N;
    int fz_idx = idx + 2 * N;

    // d_deform_grad flatten
    double d00 = d_deform_grad[0], d01 = d_deform_grad[1], d02 = d_deform_grad[2];
    double d10 = d_deform_grad[3], d11 = d_deform_grad[4], d12 = d_deform_grad[5];
    double d20 = d_deform_grad[6], d21 = d_deform_grad[7], d22 = d_deform_grad[8];

    // force_temp index
    int out_fx_idx = fx_idx;
    int out_fy_idx = fy_idx + 3;
    int out_fz_idx = fz_idx + 6;

    // force_per_atom @ deform_grad to force_temp
    force_temp[out_fx_idx] =
      force_per_atom[fx_idx] * d00 + force_per_atom[fy_idx] * d01 + force_per_atom[fz_idx] * d02;

    force_temp[out_fy_idx] =
      force_per_atom[fx_idx] * d10 + force_per_atom[fy_idx] * d11 + force_per_atom[fz_idx] * d12;

    force_temp[out_fz_idx] =
      force_per_atom[fx_idx] * d20 + force_per_atom[fy_idx] * d21 + force_per_atom[fz_idx] * d22;
  }
}

void get_force_temp(
  const double* force_per_atom,
  const double* deform,
  double* virial_cpu_deform,
  double* force_temp,
  int N)
{

  double* d_deform;
  cudaMalloc(&d_deform, 9 * sizeof(double));
  cudaMemcpy(d_deform, deform, 9 * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  get_force_temp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    force_per_atom, force_temp, d_deform, N);

  for (int m = 0; m < 9; m++) {
    virial_cpu_deform[m] = virial_cpu_deform[m] / N;
  }

  cudaMemcpy(force_temp + N, virial_cpu_deform, 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(
    force_temp + 2 * N + 3, virial_cpu_deform + 3, 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(
    force_temp + 3 * N + 6, virial_cpu_deform + 6, 3 * sizeof(double), cudaMemcpyHostToDevice);

  cudaFree(d_deform);
}

template <int N>
void solveLinearEquation(const double* A, const double* B, double* X)
{

  double a[N][N], b[N][N];
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      a[i][j] = A[j * N + i];
      b[i][j] = B[j * N + i];
    }
  }

  for (int col = 0; col < N; ++col) {
    for (int i = 0; i < N; ++i) {
      if (i == col) {
        double diag = a[i][col];
        if (fabs(diag) < 1e-9) {
          printf("Matrix is singular or nearly singular!\n");
          return;
        }
        for (int j = 0; j < N; ++j) {
          a[i][j] /= diag;
          b[i][j] /= diag;
        }
      } else {
        double factor = a[i][col];
        for (int j = 0; j < N; ++j) {
          a[i][j] -= factor * a[col][j];
          b[i][j] -= factor * b[col][j];
        }
      }
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      X[i * N + j] = b[i][j];
    }
  }
}

// get the total virial
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
  //<<<9, 1024>>>
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

__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

__global__ void gpu_vector_sum(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] + b[n];
}

// sum temp and pos_per_atom
__global__ void sum_v_pos(const int N, double* pos, const double* v)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    pos[idx] += v[idx];
    pos[N + idx] += v[N + 3 + idx];
    pos[2 * N + idx] += v[2 * N + 6 + idx];
  }
}

__global__ void gpu_pairwise_product(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] * b[n];
}

void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

__global__ void gpu_sum(const int size, double* a, double* result)
{
  int number_of_patches = (size - 1) / 1024 + 1;
  int tid = threadIdx.x;
  int n, patch;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * 1024;
    if (n < size)
      data[tid] += a[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      data[tid] += data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = data[0];
}

double sum(GPU_Vector<double>& a)
{
  double ret;
  GPU_Vector<double> result(1);
  gpu_sum<<<1, 1024>>>(a.size(), a.data(), result.data());
  result.copy_to_host(&ret);
  return ret;
}

double dot(GPU_Vector<double>& a, GPU_Vector<double>& b)
{
  GPU_Vector<double> temp(a.size());
  pairwise_product(a, b, temp);
  return sum(temp);
}

void scalar_multiply(const double& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = b.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(size, a, b.data(), c.data());
}

void vector_sum(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_vector_sum<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}
} // namespace

void Minimizer_FIRE_Box_Change::compute(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  double next_dt;
  const int size = number_of_atoms_ * 3;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;

  // minimize with changed box
  // create a velocity vector in GPU
  GPU_Vector<double> v(size + 9, 0);
  GPU_Vector<double> temp1(size + 9);
  GPU_Vector<double> temp2(size + 9);
  GPU_Vector<double> force_temp(size + 9);
  GPU_Vector<double> virialtot(9); // total virial vector of the system

  if (box.triclinic == 0) { // orthogonal box to triclinic box
    double a = box.cpu_h[0];
    double b = box.cpu_h[1];
    double c = box.cpu_h[2];
    box.triclinic = 1;
    for (int i = 0; i < 18; i++) {
      box.cpu_h[i] = 0.0;
    }
    box.cpu_h[0] = a;
    box.cpu_h[4] = b;
    box.cpu_h[8] = c;
    box.get_inverse();
  }

  double initial_box[9] = {0.0};
  for (int i = 0; i < 9; i++) {
    initial_box[i] = box.cpu_h[i];
  }

  printf("\nEnergy minimization with changed box started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    force.compute(
      box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_sum_virial<<<9, 1024>>>(
      number_of_atoms_,
      virial_per_atom.data() + 0 * number_of_atoms_,
      virial_per_atom.data() + 3 * number_of_atoms_,
      virial_per_atom.data() + 4 * number_of_atoms_,
      virial_per_atom.data() + 6 * number_of_atoms_,
      virial_per_atom.data() + 1 * number_of_atoms_,
      virial_per_atom.data() + 5 * number_of_atoms_,
      virial_per_atom.data() + 7 * number_of_atoms_,
      virial_per_atom.data() + 8 * number_of_atoms_,
      virial_per_atom.data() + 2 * number_of_atoms_,
      virialtot.data());
    GPU_CHECK_KERNEL
    double deform[9];
    // deform = np.linalg.solve(initial_box, current_box)
    solveLinearEquation<3>(initial_box, box.cpu_h, deform);
    transpose9(deform);
    double virial_cpu[9];
    virialtot.copy_to_host(virial_cpu);
    transpose9(virial_cpu);
    // virial_cpu_deform = np.linalg.solve(deform, virial_cpu).T
    double virial_cpu_deform[9];
    solveLinearEquation<3>(deform, virial_cpu, virial_cpu_deform);
    transpose9(virial_cpu_deform);

    if (hydrostatic_strain == 1) {
      double trace = virial_cpu_deform[0] + virial_cpu_deform[4] + virial_cpu_deform[8];
      for (int i = 0; i < 9; i++)
        virial_cpu_deform[i] = 0.0;
      virial_cpu_deform[0] = trace / 3.0;
      virial_cpu_deform[4] = trace / 3.0;
      virial_cpu_deform[8] = trace / 3.0;
    }

    get_force_temp(
      force_per_atom.data(), deform, virial_cpu_deform, force_temp.data(), number_of_atoms_);
    calculate_force_square_max(force_temp);
    const double force_max = sqrt(cpu_force_square_max_[0]);
    calculate_total_potential(potential_per_atom);

    if (step % base == 0 || force_max < force_tolerance_) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A, pressure = %.10f GPa.\n",
        step,
        cpu_total_potential_[0],
        force_max,
        (virial_cpu[0] + virial_cpu[4] + virial_cpu[8]) / 3. / box.get_volume() * 160.2176621);
      if (force_max < force_tolerance_)
        break;
    }

    P = dot(v, force_temp);

    if (P > 0) {
      if (N_neg > N_min) {
        next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha *= f_alpha;
      }
      N_neg++;
    } else {
      next_dt = dt * f_dec;
      if (next_dt > dt_min)
        dt = next_dt;
      alpha = alpha_start;
      // move position back
      scalar_multiply(-0.5 * dt, v, temp1);
      sum_v_pos<<<(number_of_atoms_ - 1) / 128 + 1, 128>>>(
        number_of_atoms_, position_per_atom.data(), temp1.data());
      v.fill(0);
      N_neg = 0;
    }

    // md step
    // implicit Euler integration
    double F_modulus = sqrt(dot(force_temp, force_temp));
    double v_modulus = sqrt(dot(v, v));
    // dv = F/m*dt
    scalar_multiply(dt / m, force_temp, temp2);
    vector_sum(v, temp2, v);
    scalar_multiply(1 - alpha, v, temp1);
    scalar_multiply(alpha * v_modulus / F_modulus, force_temp, temp2);
    vector_sum(temp1, temp2, v);
    // dx = v*dt
    scalar_multiply(dt, v, temp1);
    sum_v_pos<<<(number_of_atoms_ - 1) / 128 + 1, 128>>>(
      number_of_atoms_, position_per_atom.data(), temp1.data());
    update_box(box.cpu_h, v.data(), number_of_atoms_);
    box.get_inverse();
  }

  int triclinic = 0;
  for (int i = 0; i < 9; i++) {
    if ((i != 0) & (i != 4) & (i != 8)) {
      if (abs(box.cpu_h[i]) > 1e-9) {
        triclinic = 1;
        break;
      }
    }
  }
  if (triclinic == 0) {
    box.triclinic = 0;
    double a = box.cpu_h[0];
    double b = box.cpu_h[4];
    double c = box.cpu_h[8];
    for (int i = 0; i < 18; i++)
      box.cpu_h[i] = 0.0;
    box.cpu_h[0] = a;
    box.cpu_h[1] = b;
    box.cpu_h[2] = c;
    box.cpu_h[3] = a * 0.5;
    box.cpu_h[4] = b * 0.5;
    box.cpu_h[5] = c * 0.5;
    box.get_inverse();
    // printf("box triclinic is: %d\n", box.triclinic);
    // printf("current box is: \n");
    // for (int i = 0; i < 18; i++)
    //   printf("%.5f ", box.cpu_h[i]);
  }

  printf("Energy minimization finished.\n");
}