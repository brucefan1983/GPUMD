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

#include "force/neighbor.cuh"
#include "lsqt.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"

/*----------------------------------------------------------------------------80
    This file implements the linear-scaling quantum transport (LSQT) method
    similar to our GPUQT code (https://github.com/brucefan1983/gpuqt)

    In this file, we use the unit system with
        length:      Angstrom
        charge:      e
        energy:      eV
        time:        hbar/eV
------------------------------------------------------------------------------*/

namespace
{
#define BLOCK_SIZE_EC 512 // do not change this

// copy state: so = si
__global__ void gpu_copy_state(int N, double* sir, double* sii, double* sor, double* soi)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    sor[n] = sir[n];
    soi[n] = sii[n];
  }
}

// will be used for U(t)
__global__ void gpu_chebyshev_01(
  int N,
  double* s0r,
  double* s0i,
  double* s1r,
  double* s1i,
  double* sr,
  double* si,
  double b0,
  double b1,
  int direction)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double bessel_0 = b0;
    double bessel_1 = b1 * direction;
    sr[n] = bessel_0 * s0r[n] + bessel_1 * s1i[n];
    si[n] = bessel_0 * s0i[n] - bessel_1 * s1r[n];
  }
}

// will be used for U(t)
__global__ void gpu_chebyshev_2(
  int N,
  double Em_inv,
  int* NN,
  int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* s0r,
  double* s0i,
  double* s1r,
  double* s1i,
  double* s2r,
  double* s2i,
  double* sr,
  double* si,
  double bessel_m,
  int label)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double temp_real = U[n] * s1r[n]; // on-site
    double temp_imag = U[n] * s1i[n]; // on-site
    int neighbor_number = NN[n];
#pragma unroll
    for (int m = 0; m < neighbor_number; ++m) {
      int index_1 = m * N + n;
      int index_2 = NL[index_1];
      double a = Hr[index_1];
      double b = Hi[index_1];
      double c = s1r[index_2];
      double d = s1i[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real *= Em_inv; // scale
    temp_imag *= Em_inv; // scale

    temp_real = 2.0 * temp_real - s0r[n];
    temp_imag = 2.0 * temp_imag - s0i[n];
    switch (label) {
      case 1: {
        sr[n] += bessel_m * temp_real;
        si[n] += bessel_m * temp_imag;
        break;
      }
      case 2: {
        sr[n] -= bessel_m * temp_real;
        si[n] -= bessel_m * temp_imag;
        break;
      }
      case 3: {
        sr[n] += bessel_m * temp_imag;
        si[n] -= bessel_m * temp_real;
        break;
      }
      case 4: {
        sr[n] -= bessel_m * temp_imag;
        si[n] += bessel_m * temp_real;
        break;
      }
    }
    s2r[n] = temp_real;
    s2i[n] = temp_imag;
  }
}

// for KPM
__global__ void gpu_kernel_polynomial(
  int N,
  double Em_inv,
  int* NN,
  int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* s0r,
  double* s0i,
  double* s1r,
  double* s1i,
  double* s2r,
  double* s2i)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {

    double temp_real = U[n] * s1r[n]; // on-site
    double temp_imag = U[n] * s1i[n]; // on-site
    int neighbor_number = NN[n];
#pragma unroll
    for (int m = 0; m < neighbor_number; ++m) {
      int index_1 = m * N + n;
      int index_2 = NL[index_1];
      double a = Hr[index_1];
      double b = Hi[index_1];
      double c = s1r[index_2];
      double d = s1i[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }

    temp_real *= Em_inv; // scale
    temp_imag *= Em_inv; // scale

    temp_real = 2.0 * temp_real - s0r[n];
    temp_imag = 2.0 * temp_imag - s0i[n];
    s2r[n] = temp_real;
    s2i[n] = temp_imag;
  }
}

// apply the Hamiltonian: H * si = so
__global__ void gpu_apply_hamiltonian(
  int N,
  double Em_inv,
  int* NN,
  int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* sir,
  double* sii,
  double* sor,
  double* soi)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double temp_real = U[n] * sir[n]; // on-site
    double temp_imag = U[n] * sii[n]; // on-site
    int neighbor_number = NN[n];
#pragma unroll
    for (int m = 0; m < neighbor_number; ++m) {
      int index_1 = m * N + n;
      int index_2 = NL[index_1];
      double a = Hr[index_1];
      double b = Hi[index_1];
      double c = sir[index_2];
      double d = sii[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real *= Em_inv; // scale
    temp_imag *= Em_inv; // scale
    sor[n] = temp_real;
    soi[n] = temp_imag;
  }
}

// so = V * si (no scaling; no on-site)
__global__ void gpu_apply_current(
  int N,
  int* NN,
  int* NL,
  double* Hr,
  double* Hi,
  double* g_xx,
  double* sir,
  double* sii,
  double* sor,
  double* soi)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double temp_real = 0.0;
    double temp_imag = 0.0;
    int neighbor_number = NN[n];
#pragma unroll
    for (int m = 0; m < neighbor_number; ++m) {
      int index_1 = m * N + n;
      int index_2 = NL[index_1];
      double a = Hr[index_1];
      double b = Hi[index_1];
      double c = sir[index_2];
      double d = sii[index_2];
      double xx = g_xx[index_1];
      temp_real += (a * c - b * d) * xx;
      temp_imag += (a * d + b * c) * xx;
    }
    sor[n] = +temp_imag;
    soi[n] = -temp_real;
  }
}

// 1st step of <sl|sr>
static __global__ void gpu_find_inner_product_1(
  int N, double* srr, double* sri, double* slr, double* sli, double* moments, int offset)
{
  int tid = threadIdx.x;
  int n = blockIdx.x * blockDim.x + tid;
  __shared__ double s_data[BLOCK_SIZE_EC];
  s_data[tid] = 0.0;
  if (n < N) {
    s_data[tid] = (srr[n] * slr[n] + sri[n] * sli[n]);
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    moments[blockIdx.x + offset] = s_data[0];
  }
}

// 2nd step of <sl|sr>
__global__ void gpu_find_inner_product_2(
  int number_of_blocks, int number_of_patches, double* moments_tmp, double* moments)
{
  int tid = threadIdx.x;
  __shared__ double s_data[BLOCK_SIZE_EC];
  s_data[tid] = 0.0;
#pragma unroll
  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * BLOCK_SIZE_EC;
    if (n < number_of_blocks) {
      s_data[tid] += moments_tmp[blockIdx.x * number_of_blocks + n];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    moments[blockIdx.x] = s_data[0];
}

// get the Chebyshev moments: <sl|T_m(H)|sr>
void find_moments_chebyshev(
  int N,
  int Nm,
  double Em,
  int* NN,
  int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* slr,
  double* sli,
  double* srr,
  double* sri,
  double* moments)
{
  int grid_size = (N - 1) / BLOCK_SIZE_EC + 1;
  int number_of_blocks = grid_size;
  int number_of_patches = (number_of_blocks - 1) / BLOCK_SIZE_EC + 1;

  int memory_moments = sizeof(double) * Nm;
  int memory_moments_tmp = memory_moments * grid_size;
  double Em_inv = 1.0 / Em;

  double *s0r, *s1r, *s2r, *s0i, *s1i, *s2i, *moments_tmp;
  cudaMalloc((void**)&s0r, sizeof(double) * N);
  cudaMalloc((void**)&s1r, sizeof(double) * N);
  cudaMalloc((void**)&s2r, sizeof(double) * N);
  cudaMalloc((void**)&s0i, sizeof(double) * N);
  cudaMalloc((void**)&s1i, sizeof(double) * N);
  cudaMalloc((void**)&s2i, sizeof(double) * N);
  cudaMalloc((void**)&moments_tmp, memory_moments_tmp);

  // T_0(H)
  gpu_copy_state<<<grid_size, BLOCK_SIZE_EC>>>(N, srr, sri, s0r, s0i);
  gpu_find_inner_product_1<<<grid_size, BLOCK_SIZE_EC>>>(
    N, s0r, s0i, slr, sli, moments_tmp, 0 * grid_size);

  // T_1(H)
  gpu_apply_hamiltonian<<<grid_size, BLOCK_SIZE_EC>>>(
    N, Em_inv, NN, NL, U, Hr, Hi, s0r, s0i, s1r, s1i);
  gpu_find_inner_product_1<<<grid_size, BLOCK_SIZE_EC>>>(
    N, s1r, s1i, slr, sli, moments_tmp, 1 * grid_size);

  // T_m(H) (m >= 2)
  for (int m = 2; m < Nm; ++m) {
    gpu_kernel_polynomial<<<grid_size, BLOCK_SIZE_EC>>>(
      N, Em_inv, NN, NL, U, Hr, Hi, s0r, s0i, s1r, s1i, s2r, s2i);
    gpu_find_inner_product_1<<<grid_size, BLOCK_SIZE_EC>>>(
      N, s2r, s2i, slr, sli, moments_tmp, m * grid_size);
    // permute the pointers; do not need to copy the data
    double* temp_real;
    double* temp_imag;
    temp_real = s0r;
    temp_imag = s0i;
    s0r = s1r;
    s0i = s1i;
    s1r = s2r;
    s1i = s2i;
    s2r = temp_real;
    s2i = temp_imag;
  }

  gpu_find_inner_product_2<<<Nm, BLOCK_SIZE_EC>>>(
    number_of_blocks, number_of_patches, moments_tmp, moments);

  cudaFree(s0r);
  cudaFree(s0i);
  cudaFree(s1r);
  cudaFree(s1i);
  cudaFree(s2r);
  cudaFree(s2i);
  cudaFree(moments_tmp);
}

// Jackson damping
void apply_damping(int Nm, double* moments)
{
  for (int k = 0; k < Nm; ++k) {
    double a = 1.0 / (Nm + 1.0);
    double damping = (1.0 - k * a) * cos(k * PI * a) + sin(k * PI * a) * a / tan(PI * a);
    moments[k] *= damping;
  }
}

// kernel polynomial summation
void perform_chebyshev_summation(
  int Nm, int Ne, double Em, double* E, double* moments, double* correlation)
{
  for (int step1 = 0; step1 < Ne; ++step1) {
    double energy_scaled = E[step1] / Em;
    double chebyshev_0 = 1.0;
    double chebyshev_1 = energy_scaled;
    double chebyshev_2;
    double temp = moments[1] * chebyshev_1;
    for (int step2 = 2; step2 < Nm; ++step2) {
      chebyshev_2 = 2.0 * energy_scaled * chebyshev_1 - chebyshev_0;
      chebyshev_0 = chebyshev_1;
      chebyshev_1 = chebyshev_2;
      temp += moments[step2] * chebyshev_2;
    }
    temp *= 2.0;
    temp += moments[0];
    temp *= 2.0 / (PI * sqrt(1.0 - energy_scaled * energy_scaled));
    correlation[step1] = temp / Em;
  }
}

// direction = +1: U(+t) |state>
// direction = -1: U(-t) |state>
void evolve(
  int N,
  double Em,
  int direction,
  double time_step_scaled,
  int* NN,
  int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* sr,
  double* si)
{
  int grid_size = (N - 1) / BLOCK_SIZE_EC + 1;
  double Em_inv = 1.0 / Em;
  double* s0r;
  double* s1r;
  double* s2r;
  double* s0i;
  double* s1i;
  double* s2i;
  cudaMalloc((void**)&s0r, sizeof(double) * N);
  cudaMalloc((void**)&s0i, sizeof(double) * N);
  cudaMalloc((void**)&s1r, sizeof(double) * N);
  cudaMalloc((void**)&s1i, sizeof(double) * N);
  cudaMalloc((void**)&s2r, sizeof(double) * N);
  cudaMalloc((void**)&s2i, sizeof(double) * N);

  // T_0(H) |psi> = |psi>
  gpu_copy_state<<<grid_size, BLOCK_SIZE_EC>>>(N, sr, si, s0r, s0i);

  // T_1(H) |psi> = H |psi>
  gpu_apply_hamiltonian<<<grid_size, BLOCK_SIZE_EC>>>(
    N, Em_inv, NN, NL, U, Hr, Hi, sr, si, s1r, s1i);

  // |final_state> = c_0 * T_0(H) |psi> + c_1 * T_1(H) |psi>
  double bessel_0 = j0(time_step_scaled);
  double bessel_1 = 2.0 * j1(time_step_scaled);
  gpu_chebyshev_01<<<grid_size, BLOCK_SIZE_EC>>>(
    N, s0r, s0i, s1r, s1i, sr, si, bessel_0, bessel_1, direction);

  for (int m = 2; m < 1000000; ++m) {
    double bessel_m = jn(m, time_step_scaled);
    if (bessel_m < 1.0e-15 && bessel_m > -1.0e-15) {
      break;
    }
    bessel_m *= 2.0;
    int label;
    int m_mod_4 = m % 4;
    if (m_mod_4 == 0) {
      label = 1;
    } else if (m_mod_4 == 2) {
      label = 2;
    } else if ((m_mod_4 == 1 && direction == 1) || (m_mod_4 == 3 && direction == -1)) {
      label = 3;
    } else {
      label = 4;
    }
    gpu_chebyshev_2<<<grid_size, BLOCK_SIZE_EC>>>(
      N, Em_inv, NN, NL, U, Hr, Hi, s0r, s0i, s1r, s1i, s2r, s2i, sr, si, bessel_m, label);

    // permute the pointers; do not need to copy the data
    double *temp_real, *temp_imag;
    temp_real = s0r;
    temp_imag = s0i;
    s0r = s1r;
    s0i = s1i;
    s1r = s2r;
    s1i = s2i;
    s2r = temp_real;
    s2i = temp_imag;
  }
  cudaFree(s0r);
  cudaFree(s0i);
  cudaFree(s1r);
  cudaFree(s1i);
  cudaFree(s2r);
  cudaFree(s2i);
}

// set up Hamiltonian and related quantities
__global__ void gpu_initialize_model(
  const Box box,
  const int N,
  const int direction,
  const double* x,
  const double* y,
  const double* z,
  const int* NN,
  const int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* xx)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int neighbor_number = NN[n1];
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = NL[index];
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double z12 = z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

      if (direction == 1) {
        xx[index] = x12;
      }
      if (direction == 2) {
        xx[index] = y12;
      }
      if (direction == 3) {
        xx[index] = z12;
      }
      // Hr[index] = -2.5; // test
      Hr[index] = -2.7 * 1.44 * 1.44 / (d12 * d12); // a CNT model
      Hi[index] = 0.0;                              // may be used in the future
    }
    U[n1] = 0.0; // may be used in the future
  }
}

void find_dos_or_others(
  int N,
  int Nm,
  int Ne,
  double Em,
  double* E,
  int* NN,
  int* NL,
  double* U,
  double* Hr,
  double* Hi,
  double* slr,
  double* sli,
  double* srr,
  double* sri,
  double* dos_or_others)
{
  std::vector<double> moments_cpu(Nm);
  GPU_Vector<double> moments_gpu(Nm);
  find_moments_chebyshev(N, Nm, Em, NN, NL, U, Hr, Hi, slr, sli, srr, sri, moments_gpu.data());
  moments_gpu.copy_to_host(moments_cpu.data());
  apply_damping(Nm, moments_cpu.data());
  perform_chebyshev_summation(Nm, Ne, Em, E, moments_cpu.data(), dos_or_others);
}

void initialize_state(int N, GPU_Vector<double>& sr, GPU_Vector<double>& si)
{
  std::vector<double> sr_cpu(N);
  std::vector<double> si_cpu(N);
  for (int n = 0; n < N; ++n) {
    double random_phase = rand() / double(RAND_MAX) * 2.0 * PI;
    sr_cpu[n] = cos(random_phase);
    si_cpu[n] = sin(random_phase);
  }
  sr.copy_from_host(sr_cpu.data());
  si.copy_from_host(si_cpu.data());
}
} // namespace

void LSQT::preprocess(Atom& atom, int number_of_steps)
{
  number_of_atoms = atom.number_of_atoms;
  this->number_of_steps = number_of_steps;
  int M = number_of_atoms * 10; // number of pairs
  transport_direction = 1;
  number_of_moments = 1000;
  number_of_energy_points = 1001;
  maximum_energy = 10.1;
  dt = 1.6; // TODO (this is 1.6 * hbar/eV, which is about 1 fs)
  E.resize(number_of_energy_points);
  for (int n = 0; n < number_of_energy_points; ++n) {
    E[n] = (n - (number_of_energy_points - 1) / 2) * 0.02;
  }

  cell_count.resize(number_of_atoms);
  cell_count_sum.resize(number_of_atoms);
  cell_contents.resize(number_of_atoms);
  NN.resize(number_of_atoms);
  NL.resize(M);

  xx.resize(M);
  Hr.resize(M);
  Hi.resize(M);
  U.resize(number_of_atoms);

  sigma.resize(number_of_energy_points);

  slr.resize(number_of_atoms);
  sli.resize(number_of_atoms);
  srr.resize(number_of_atoms);
  sri.resize(number_of_atoms);
  scr.resize(number_of_atoms);
  sci.resize(number_of_atoms);
}

void LSQT::process(Atom& atom, Box& box, const int step)
{
  find_neighbor(
    0,
    number_of_atoms,
    2.1,
    box,
    atom.type,
    atom.position_per_atom,
    cell_count,
    cell_count_sum,
    cell_contents,
    NN,
    NL);

  gpu_initialize_model<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    box,
    number_of_atoms,
    transport_direction,
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2,
    NN.data(),
    NL.data(),
    U.data(),
    Hr.data(),
    Hi.data(),
    xx.data());

  find_dos_and_velocity(atom, box);
  find_sigma(atom, box, step);
}

void LSQT::find_dos_and_velocity(Atom& atom, Box& box)
{
  std::vector<double> dos(number_of_energy_points);
  std::vector<double> velocity(number_of_energy_points);

  GPU_Vector<double> sr(number_of_atoms);
  GPU_Vector<double> si(number_of_atoms);
  GPU_Vector<double> sxr(number_of_atoms);
  GPU_Vector<double> sxi(number_of_atoms);

  initialize_state(number_of_atoms, sr, si);

  // dos
  find_dos_or_others(
    number_of_atoms,
    number_of_moments,
    number_of_energy_points,
    maximum_energy,
    E.data(),
    NN.data(),
    NL.data(),
    U.data(),
    Hr.data(),
    Hi.data(),
    sr.data(),
    si.data(),
    sr.data(),
    si.data(),
    dos.data());

  FILE* os_dos = my_fopen("lsqt_dos.out", "a");
  for (int n = 0; n < number_of_energy_points; ++n)
    fprintf(os_dos, "%25.15e", dos[n] / number_of_atoms);
  fprintf(os_dos, "\n");
  fclose(os_dos);

  // velocity
  gpu_apply_current<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms,
    NN.data(),
    NL.data(),
    Hr.data(),
    Hi.data(),
    xx.data(),
    sr.data(),
    si.data(),
    sxr.data(),
    sxi.data());

  find_dos_or_others(
    number_of_atoms,
    number_of_moments,
    number_of_energy_points,
    maximum_energy,
    E.data(),
    NN.data(),
    NL.data(),
    U.data(),
    Hr.data(),
    Hi.data(),
    sxr.data(),
    sxi.data(),
    sxr.data(),
    sxi.data(),
    velocity.data());

  FILE* os_vel = my_fopen("lsqt_velocity.out", "a");
  for (int n = 0; n < number_of_energy_points; ++n)
    fprintf(os_vel, "%25.15e", sqrt(velocity[n] / dos[n]));
  fprintf(os_vel, "\n");
  fclose(os_vel);
}

void LSQT::find_sigma(Atom& atom, Box& box, const int step)
{
  double dt_scaled = dt * maximum_energy;
  double V = box.get_volume();

  if (step == 0) {
    initialize_state(number_of_atoms, slr, sli);
    gpu_apply_current<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
      number_of_atoms,
      NN.data(),
      NL.data(),
      Hr.data(),
      Hi.data(),
      xx.data(),
      slr.data(),
      sli.data(),
      srr.data(),
      sri.data());
  } else {
    evolve(
      number_of_atoms,
      maximum_energy,
      -1,
      dt_scaled,
      NN.data(),
      NL.data(),
      U.data(),
      Hr.data(),
      Hi.data(),
      slr.data(),
      sli.data());

    evolve(
      number_of_atoms,
      maximum_energy,
      -1,
      dt_scaled,
      NN.data(),
      NL.data(),
      U.data(),
      Hr.data(),
      Hi.data(),
      srr.data(),
      sri.data());
  }

  gpu_apply_current<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    number_of_atoms,
    NN.data(),
    NL.data(),
    Hr.data(),
    Hi.data(),
    xx.data(),
    slr.data(),
    sli.data(),
    scr.data(),
    sci.data());

  std::vector<double> vac(number_of_energy_points);

  find_dos_or_others(
    number_of_atoms,
    number_of_moments,
    number_of_energy_points,
    maximum_energy,
    E.data(),
    NN.data(),
    NL.data(),
    U.data(),
    Hr.data(),
    Hi.data(),
    scr.data(),
    sci.data(),
    srr.data(),
    sri.data(),
    vac.data());

  FILE* os_sigma = my_fopen("lsqt_sigma.out", "a");
  for (int n = 0; n < number_of_energy_points; ++n) {
    sigma[n] += vac[n] * dt / V;
    fprintf(os_sigma, "%25.15e", sigma[n]);
  }
  fprintf(os_sigma, "\n");
  fclose(os_sigma);
}
