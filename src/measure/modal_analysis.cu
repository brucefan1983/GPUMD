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

/*----------------------------------------------------------------------------80
Green-Kubo Modal Analysis (GKMA) and
Homogenous Nonequilibrium Modal Analysis (HNEMA) implementations.

Original GMKA method is detailed in:
H.R. Seyf, K. Gordiz, F. DeAngelis, and A. Henry, "Using Green-Kubo modal
analysis (GKMA) and interface conductance modal analysis (ICMA) to study
phonon transport with molecular dynamics," J. Appl. Phys., 125, 081101 (2019).

The code here is inspired by the LAMMPS implementation provided by the Henry
group at MIT. This code can be found:
https://drive.google.com/open?id=1IHJ7x-bLZISX3I090dW_Y_y-Mqkn07zg

GPUMD Contributing author: Alexander Gabourie (Stanford University)
------------------------------------------------------------------------------*/

#include "modal_analysis.cuh"
#include "utilities/error.cuh"

#define NUM_OF_HEAT_COMPONENTS 5
#define BLOCK_SIZE 128
#define ACCUM_BLOCK 1024
#define BIN_BLOCK 128
#define BLOCK_SIZE_FORCE 64
#define BLOCK_SIZE_GK 16

#define ACCUMULATE 0
#define SET 1

static __global__ void gpu_reset_data(int num_elements, float* data)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num_elements) {
    data[n] = 0.0f;
  }
}

static __global__ void gpu_scale_jm(int num_elements, float factor, float* jm)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < num_elements) {
    jm[n] *= factor;
  }
}

static __device__ void gpu_bin_reduce(
  int num_modes,
  int bin_size,
  int shift,
  int num_bins,
  int tid,
  int bid,
  int number_of_patches,
  const float* __restrict__ jm,
  float* bin_out)
{
  __shared__ float s_data_xin[BIN_BLOCK];
  __shared__ float s_data_xout[BIN_BLOCK];
  __shared__ float s_data_yin[BIN_BLOCK];
  __shared__ float s_data_yout[BIN_BLOCK];
  __shared__ float s_data_z[BIN_BLOCK];
  s_data_xin[tid] = 0.0f;
  s_data_xout[tid] = 0.0f;
  s_data_yin[tid] = 0.0f;
  s_data_yout[tid] = 0.0f;
  s_data_z[tid] = 0.0f;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * BIN_BLOCK;
    if (n < bin_size) {
      s_data_xin[tid] += jm[n + shift];
      s_data_xout[tid] += jm[n + shift + num_modes];
      s_data_yin[tid] += jm[n + shift + 2 * num_modes];
      s_data_yout[tid] += jm[n + shift + 3 * num_modes];
      s_data_z[tid] += jm[n + shift + 4 * num_modes];
    }
  }

  __syncthreads();
#pragma unroll
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data_xin[tid] += s_data_xin[tid + offset];
      s_data_xout[tid] += s_data_xout[tid + offset];
      s_data_yin[tid] += s_data_yin[tid + offset];
      s_data_yout[tid] += s_data_yout[tid + offset];
      s_data_z[tid] += s_data_z[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    bin_out[bid] = s_data_xin[0];
    bin_out[bid + num_bins] = s_data_xout[0];
    bin_out[bid + 2 * num_bins] = s_data_yin[0];
    bin_out[bid + 3 * num_bins] = s_data_yout[0];
    bin_out[bid + 4 * num_bins] = s_data_z[0];
  }
}

static __global__ void gpu_bin_modes(
  int num_modes,
  const int* __restrict__ bin_count,
  const int* __restrict__ bin_sum,
  int num_bins,
  const float* __restrict__ jm,
  float* bin_out)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bin_size = bin_count[bid];
  int shift = bin_sum[bid];
  int number_of_patches = (bin_size - 1) / BIN_BLOCK + 1;

  gpu_bin_reduce(num_modes, bin_size, shift, num_bins, tid, bid, number_of_patches, jm, bin_out);
}

static __global__ void elemwise_mass_scale(
  int num_participating,
  int N1,
  const float* __restrict__ g_sqrtmass,
  const double* __restrict__ g_vx,
  const double* __restrict__ g_vy,
  const double* __restrict__ g_vz,
  float* mvx,
  float* mvy,
  float* mvz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nglobal = i + N1;
  if (i < num_participating) {
    float sqrtmass = g_sqrtmass[i];
    float vx, vy, vz;
    vx = __double2float_rn(g_vx[nglobal]);
    vy = __double2float_rn(g_vy[nglobal]);
    vz = __double2float_rn(g_vz[nglobal]);
    mvx[i] = sqrtmass * vx;
    mvy[i] = sqrtmass * vy;
    mvz[i] = sqrtmass * vz;
  }
}

static __global__ void gpu_set_mass_terms(
  int num_participating,
  int N1,
  const double* __restrict__ g_mass,
  float* sqrtmass,
  float* rsqrtmass)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nglobal = i + N1;
  if (i < num_participating) {
    float mass = __double2float_rn(g_mass[nglobal]);
    sqrtmass[i] = sqrt(mass);
    rsqrtmass[i] = rsqrt(mass);
  }
}

static __global__ void prepare_sm(
  int num_participating,
  int N1,
  const double* __restrict__ sxx,
  const double* __restrict__ sxy,
  const double* __restrict__ sxz,
  const double* __restrict__ syx,
  const double* __restrict__ syy,
  const double* __restrict__ syz,
  const double* __restrict__ szx,
  const double* __restrict__ szy,
  const double* __restrict__ szz,
  const float* __restrict__ rsqrtmass,
  float* smx,
  float* smy,
  float* smz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int nglobal = i + N1;
  if (i < num_participating) {
    float invmass = rsqrtmass[i];
    // x's
    smx[i] = __double2float_rn(sxx[nglobal]) * invmass;
    smx[i + num_participating] = __double2float_rn(syx[nglobal]) * invmass;
    smx[i + 2 * num_participating] = __double2float_rn(szx[nglobal]) * invmass;

    // y's
    smy[i] = __double2float_rn(sxy[nglobal]) * invmass;
    smy[i + num_participating] = __double2float_rn(syy[nglobal]) * invmass;
    smy[i + 2 * num_participating] = __double2float_rn(szy[nglobal]) * invmass;

    // z's
    smz[i] = __double2float_rn(sxz[nglobal]) * invmass;
    smz[i + num_participating] = __double2float_rn(syz[nglobal]) * invmass;
    smz[i + 2 * num_participating] = __double2float_rn(szz[nglobal]) * invmass;
  }
}

template <int operate>
static __global__ void gpu_update_jm(
  int num_modes,
  const float* __restrict__ jmx,
  const float* __restrict__ jmy,
  const float* __restrict__ jmz,
  float* jm)
{
  int mode = blockIdx.x * blockDim.x + threadIdx.x;
  if (mode < num_modes) {
    int yidx = mode + num_modes;
    int zidx = mode + 2 * num_modes;

    if (operate == SET) {
      jm[mode] = jmx[mode] + jmy[mode];                             // jxi
      jm[mode + num_modes] = jmz[mode];                             // jxo
      jm[mode + 2 * num_modes] = jmx[yidx] + jmy[yidx];             // jyi
      jm[mode + 3 * num_modes] = jmz[yidx];                         // jyo
      jm[mode + 4 * num_modes] = jmx[zidx] + jmy[zidx] + jmz[zidx]; // jz
    }
    if (operate == ACCUMULATE) {
      jm[mode] += jmx[mode] + jmy[mode];                             // jxi
      jm[mode + num_modes] += jmz[mode];                             // jxo
      jm[mode + 2 * num_modes] += jmx[yidx] + jmy[yidx];             // jyi
      jm[mode + 3 * num_modes] += jmz[yidx];                         // jyo
      jm[mode + 4 * num_modes] += jmx[zidx] + jmy[zidx] + jmz[zidx]; // jz
    }
  }
}

void MODAL_ANALYSIS::compute_heat(
  const GPU_Vector<double>& velocity_per_atom, const GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = velocity_per_atom.size() / 3;

  int grid_size = (num_participating - 1) / BLOCK_SIZE + 1;
  // precalculate velocity*sqrt(mass)
  elemwise_mass_scale<<<grid_size, BLOCK_SIZE>>>(
    num_participating, N1, sqrtmass.data(), velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms,
    mvx.data(), mvy.data(), mvz.data());
  CUDA_CHECK_KERNEL

  // Scale stress tensor by inv(sqrt(mass))
  prepare_sm<<<grid_size, BLOCK_SIZE>>>(
    num_participating, N1, virial_per_atom.data(), virial_per_atom.data() + number_of_atoms * 3,
    virial_per_atom.data() + number_of_atoms * 4, virial_per_atom.data() + number_of_atoms * 6,
    virial_per_atom.data() + number_of_atoms * 1, virial_per_atom.data() + number_of_atoms * 5,
    virial_per_atom.data() + number_of_atoms * 7, virial_per_atom.data() + number_of_atoms * 8,
    virial_per_atom.data() + number_of_atoms * 2, rsqrtmass.data(), smx.data(), smy.data(),
    smz.data());
  CUDA_CHECK_KERNEL

  const float alpha = 1.0;
  const float beta = 0.0;
  int stride = 1;

  // Calculate modal velocities
  cublasSgemv(
    ma_handle, CUBLAS_OP_N, num_modes, num_participating, &alpha, eigx.data(), num_modes,
    mvx.data(), stride, &beta, xdotx.data(), stride);
  cublasSgemv(
    ma_handle, CUBLAS_OP_N, num_modes, num_participating, &alpha, eigy.data(), num_modes,
    mvy.data(), stride, &beta, xdoty.data(), stride);
  cublasSgemv(
    ma_handle, CUBLAS_OP_N, num_modes, num_participating, &alpha, eigz.data(), num_modes,
    mvz.data(), stride, &beta, xdotz.data(), stride);

  // Calculate intermediate value
  // (i.e. heat current without modal velocities)
  cublasSgemm(
    ma_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_modes, 3, num_participating, &alpha, eigx.data(),
    num_modes, smx.data(), num_participating, &beta, jmx.data(), num_modes);
  cublasSgemm(
    ma_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_modes, 3, num_participating, &alpha, eigy.data(),
    num_modes, smy.data(), num_participating, &beta, jmy.data(), num_modes);
  cublasSgemm(
    ma_handle, CUBLAS_OP_N, CUBLAS_OP_N, num_modes, 3, num_participating, &alpha, eigz.data(),
    num_modes, smz.data(), num_participating, &beta, jmz.data(), num_modes);

  // calculate modal heat current
  cublasSdgmm(
    ma_handle, CUBLAS_SIDE_LEFT, num_modes, 3, jmx.data(), num_modes, xdotx.data(), stride,
    jmx.data(), num_modes);
  cublasSdgmm(
    ma_handle, CUBLAS_SIDE_LEFT, num_modes, 3, jmy.data(), num_modes, xdoty.data(), stride,
    jmy.data(), num_modes);
  cublasSdgmm(
    ma_handle, CUBLAS_SIDE_LEFT, num_modes, 3, jmz.data(), num_modes, xdotz.data(), stride,
    jmz.data(), num_modes);

  // Prepare modal heat current for jxi, jxo, jyi, jyo, jz format
  grid_size = (num_modes - 1) / BLOCK_SIZE + 1;
  if (method == GKMA_METHOD) {
    gpu_update_jm<SET>
      <<<grid_size, BLOCK_SIZE>>>(num_modes, jmx.data(), jmy.data(), jmz.data(), jm.data());
  } else if (method == HNEMA_METHOD) {
    gpu_update_jm<ACCUMULATE>
      <<<grid_size, BLOCK_SIZE>>>(num_modes, jmx.data(), jmy.data(), jmz.data(), jm.data());
  }
  CUDA_CHECK_KERNEL
}

void MODAL_ANALYSIS::setN(const std::vector<int>& cpu_type_size)
{
  N1 = 0;
  N2 = 0;
  for (int n = 0; n < atom_begin; ++n) {
    N1 += cpu_type_size[n];
  }
  for (int n = atom_begin; n <= atom_end; ++n) {
    N2 += cpu_type_size[n];
  }

  num_participating = N2 - N1;
}

void MODAL_ANALYSIS::set_eigmode(int mode, std::ifstream& eigfile, GPU_Vector<float>& eig)
{
  float floatval;
  for (int i = 0; i < num_participating; i++) {
    eigfile >> floatval;
    // column major ordering for cuBLAS
    eig[mode + i * num_modes] = floatval;
  }
}

void MODAL_ANALYSIS::preprocess(
  char* input_dir, const std::vector<int>& cpu_type_size, const GPU_Vector<double>& mass)
{
  if (!compute)
    return;
  num_modes = last_mode - first_mode + 1;
  samples_per_output = output_interval / sample_interval;
  setN(cpu_type_size);

  strcpy(output_file_position, input_dir);
  if (method == GKMA_METHOD) {
    strcat(output_file_position, "/heatmode.out");
  } else if (method == HNEMA_METHOD) {
    strcat(output_file_position, "/kappamode.out");
  }

  size_t eig_size = num_participating * num_modes;
  eigx.resize(eig_size, Memory_Type::managed);
  eigy.resize(eig_size, Memory_Type::managed);
  eigz.resize(eig_size, Memory_Type::managed);

  // initialize eigenvector data structures
  strcpy(eig_file_position, input_dir);
  strcat(eig_file_position, "/eigenvector.in");
  std::ifstream eigfile;
  eigfile.open(eig_file_position);
  if (!eigfile) {
    PRINT_INPUT_ERROR("Cannot open eigenvector.in file.");
  }

  // GPU phonon code output format
  std::string val;

  // Setup binning
  if (f_flag) {
    GPU_Vector<double> f(num_modes, Memory_Type::managed);
    getline(eigfile, val);
    std::stringstream ss(val);
    for (int i = 0; i < first_mode - 1; i++) {
      ss >> f[0];
    }
    double temp;
    for (int i = 0; i < num_modes; i++) {
      ss >> temp;
      f[i] = copysign(sqrt(abs(temp)) / (2.0 * PI), temp);
    }
    double fmax, fmin; // freq are in ascending order in file
    int shift;
    const double epsilon = 1.e-6;
    fmax = (floor(abs(f[num_modes - 1]) / f_bin_size) + 1) * f_bin_size;
    fmin = floor(abs(f[0]) / f_bin_size) * f_bin_size;
    shift = floor(abs(fmin) / f_bin_size + epsilon);
    num_bins = floor((fmax - fmin) / f_bin_size + epsilon);

    bin_count.resize(num_bins, 0, Memory_Type::managed);
    for (int i = 0; i < num_modes; i++)
      bin_count[int(abs(f[i] / f_bin_size)) - shift]++;

  } else {
    num_bins = (int)ceil((double)num_modes / (double)bin_size);
    bin_count.resize(num_bins, Memory_Type::managed);
    for (int i_ = 0; i_ < num_bins; i_++) {
      bin_count[i_] = (int)bin_size;
    }
    if (num_modes % bin_size != 0) {
      bin_count[num_bins - 1] = num_modes % bin_size;
    }

    getline(eigfile, val);
  }

  bin_sum.resize(num_bins, 0, Memory_Type::managed);
  for (int i = 1; i < num_bins; i++)
    bin_sum[i] = bin_sum[i - 1] + bin_count[i - 1];

  // skips modes up to first_mode
  for (int i = 1; i < first_mode; i++) {
    getline(eigfile, val);
  }
  for (int j = 0; j < num_modes; j++) // modes
  {
    set_eigmode(j, eigfile, eigx);
    set_eigmode(j, eigfile, eigy);
    set_eigmode(j, eigfile, eigz);
  }
  eigfile.close();

  // Allocate intermediate vector
  mvx.resize(num_participating, Memory_Type::managed);
  mvy.resize(num_participating, Memory_Type::managed);
  mvz.resize(num_participating, Memory_Type::managed);

  // Allocate modal velocities
  xdotx.resize(num_modes, Memory_Type::managed);
  xdoty.resize(num_modes, Memory_Type::managed);
  xdotz.resize(num_modes, Memory_Type::managed);

  // Allocate modal measured quantities
  size_t jmxyz_size = num_modes * 3;
  jmx.resize(jmxyz_size, Memory_Type::managed);
  jmy.resize(jmxyz_size, Memory_Type::managed);
  jmz.resize(jmxyz_size, Memory_Type::managed);

  num_heat_stored = num_modes * NUM_OF_HEAT_COMPONENTS;
  jm.resize(num_heat_stored, 0.0f, Memory_Type::managed);

  size_t bin_out_size = num_bins * NUM_OF_HEAT_COMPONENTS;
  bin_out.resize(bin_out_size, Memory_Type::managed);

  size_t sm_size = num_modes * 3;
  smx.resize(sm_size, Memory_Type::managed);
  smy.resize(sm_size, Memory_Type::managed);
  smz.resize(sm_size, Memory_Type::managed);

  // prepare masses
  sqrtmass.resize(num_participating, Memory_Type::managed);
  rsqrtmass.resize(num_participating, Memory_Type::managed);
  gpu_set_mass_terms<<<(num_participating - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
    num_participating, N1, mass.data(), sqrtmass.data(), rsqrtmass.data());
  CUDA_CHECK_KERNEL

  cublasCreate(&ma_handle);
}

void MODAL_ANALYSIS::process(
  const int step,
  const double temperature,
  const double volume,
  const double fe,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& virial_per_atom)
{
  if (!compute)
    return;
  if (!((step + 1) % sample_interval == 0))
    return;

  compute_heat(velocity_per_atom, virial_per_atom);

  if (method == HNEMA_METHOD && !((step + 1) % output_interval == 0))
    return;

  gpu_bin_modes<<<num_bins, BIN_BLOCK>>>(
    num_modes, bin_count.data(), bin_sum.data(), num_bins, jm.data(), bin_out.data());
  CUDA_CHECK_KERNEL

  if (method == HNEMA_METHOD) {
    float factor = KAPPA_UNIT_CONVERSION / (volume * temperature * fe * (float)samples_per_output);
    int num_bins_stored = num_bins * NUM_OF_HEAT_COMPONENTS;
    gpu_scale_jm<<<(num_bins_stored - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(
      num_bins_stored, factor, bin_out.data());
    CUDA_CHECK_KERNEL
  }

  // Compute thermal conductivity and output
  cudaDeviceSynchronize(); // ensure GPU ready to move data to CPU
  FILE* fid = fopen(output_file_position, "a");
  for (int i = 0; i < num_bins; i++) {
    fprintf(
      fid, "%g %g %g %g %g\n", bin_out[i], bin_out[i + num_bins], bin_out[i + 2 * num_bins],
      bin_out[i + 3 * num_bins], bin_out[i + 4 * num_bins]);
  }
  fflush(fid);
  fclose(fid);

  if (method == HNEMA_METHOD) {
    int grid_size = (num_heat_stored - 1) / BLOCK_SIZE + 1;
    gpu_reset_data<<<grid_size, BLOCK_SIZE>>>(num_heat_stored, jm.data());
    CUDA_CHECK_KERNEL
  }
}

void MODAL_ANALYSIS::postprocess()
{
  if (!compute)
    return;
  cublasDestroy(ma_handle);
}
