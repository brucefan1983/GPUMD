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

#include "dump_shock_nemd.cuh"
#include <cstring>

namespace
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

static __global__ void gpu_com(
  int N,
  int avg_window,
  double* g_mass,
  double* g_x,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  double* com_vx_data,
  double* com_vy_data,
  double* com_vz_data,
  double* density_data)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double mass, vx, vy, vz;
  if (i < N) {
    int l = (int)(g_x[i] / avg_window);
    mass = g_mass[i];
    vx = g_vx[i];
    vy = g_vy[i];
    vz = g_vz[i];
    atomicAdd(&com_vx_data[l], vx * mass);
    atomicAdd(&com_vy_data[l], vy * mass);
    atomicAdd(&com_vz_data[l], vz * mass);
    atomicAdd(&density_data[l], mass);
  }
}

static __global__ void gpu_calc1(
  int bins, double* com_vx_data, double* com_vy_data, double* com_vz_data, double* density_data)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i < bins) && (density_data[i] > 1e-5)) {
    com_vx_data[i] /= density_data[i];
    com_vy_data[i] /= density_data[i];
    com_vz_data[i] /= density_data[i];
  }
}

static __global__ void gpu_thermo(
  int N,
  double avg_window,
  double* g_x,
  double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  double* g_pxx,
  double* g_pyy,
  double* g_pzz,
  double* temp_data,
  double* com_vx_data,
  double* com_vy_data,
  double* com_vz_data,
  double* pxx_data,
  double* pyy_data,
  double* pzz_data,
  double* number_data)
{
  double mass, vx, vy, vz;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int l = (int)(g_x[i] / avg_window);
    mass = g_mass[i];
    vx = g_vx[i] - com_vx_data[l];
    vy = g_vy[i] - com_vy_data[l];
    vz = g_vz[i] - com_vz_data[l];
    atomicAdd(&temp_data[l], (vx * vx + vy * vy + vz * vz) * mass);
    atomicAdd(&pxx_data[l], g_pxx[i] + vx * vx * mass);
    atomicAdd(&pyy_data[l], g_pyy[i] + vy * vy * mass);
    atomicAdd(&pzz_data[l], g_pzz[i] + vz * vz * mass);
    atomicAdd(&number_data[l], 1);
  }
}

static __global__ void gpu_calc2(
  int bins,
  double slice_volume,
  double* temp_data,
  double* pxx_data,
  double* pyy_data,
  double* pzz_data,
  double* density_data,
  double* number_data)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < bins) {
    // if there are too few atoms, stop calculating temperature
    if (number_data[i] >= 20)
      temp_data[i] /= 3 * number_data[i] * K_B;
    pxx_data[i] = pxx_data[i] / slice_volume * 1.602177e+2;
    pyy_data[i] = pyy_data[i] / slice_volume * 1.602177e+2;
    pzz_data[i] = pzz_data[i] / slice_volume * 1.602177e+2;
    density_data[i] /= slice_volume;
  }
}

void write_to_file(FILE* file, double* array, int n)
{
  for (int i = 0; i < n; i++)
    fprintf(file, "%f ", array[i]);
  fprintf(file, "\n");
}

} // namespace

void Dump_Shock_NEMD::parse(const char** param, int num_param)
{
  dump_ = true;

  printf("Dump spatial histogram thermo information for piston shock wave simulation, ");
  int i = 1;
  while (i < num_param) {
    if (strcmp(param[i], "interval") == 0) {
      if (!is_valid_int(param[i + 1], &dump_interval_))
        PRINT_INPUT_ERROR("Dump interval should be an integer.");
      i += 2;
    } else if (strcmp(param[i], "bin_size") == 0) {
      if (!is_valid_real(param[i + 1], &avg_window))
        PRINT_INPUT_ERROR("Wrong inputs for bin_size.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }
}

void Dump_Shock_NEMD::preprocess(Atom& atom, Box& box)
{
  if (!dump_)
    return;

  n = atom.number_of_atoms;
  bins = (int)box.cpu_h[direction] / avg_window + 1;
  if (n < bins)
    PRINT_INPUT_ERROR("Too few atoms!");
  for (int i = 0; i < 3; i++)
    if (i != direction)
      slice_vol *= box.cpu_h[i]; // create vectors to store hist
  slice_vol *= avg_window;

  temp_file = my_fopen("temperature_hist.txt", "w");
  pxx_file = my_fopen("pxx_hist.txt", "w");
  pyy_file = my_fopen("pyy_hist.txt", "w");
  pzz_file = my_fopen("pzz_hist.txt", "w");
  density_file = my_fopen("density_hist.txt", "w");
  com_vx_file = my_fopen("vp_hist.txt", "w");
}

void Dump_Shock_NEMD::process(Atom& atom, Box& box, const int step)
{
  if (!dump_ || step % dump_interval_ != 0)
    return;

  gpu_temp.resize(bins, 0);
  gpu_pxx.resize(bins, 0);
  gpu_pyy.resize(bins, 0);
  gpu_pzz.resize(bins, 0);
  gpu_density.resize(bins, 0);
  gpu_number.resize(bins, 0);
  gpu_com_vx.resize(bins, 0);
  gpu_com_vy.resize(bins, 0);
  gpu_com_vz.resize(bins, 0);
  cpu_temp.resize(bins, 0);
  cpu_pxx.resize(bins, 0);
  cpu_pyy.resize(bins, 0);
  cpu_pzz.resize(bins, 0);
  cpu_density.resize(bins, 0);
  cpu_com_vx.resize(bins, 0);
  cpu_com_vy.resize(bins, 0);
  cpu_com_vz.resize(bins, 0);
  // calculate COM velocity first
  gpu_com<<<(n - 1) / 128 + 1, 128>>>(
    n,
    avg_window,
    atom.mass.data(),
    atom.position_per_atom.data() + direction * n,
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + n,
    atom.velocity_per_atom.data() + 2 * n,
    gpu_com_vx.data(),
    gpu_com_vy.data(),
    gpu_com_vz.data(),
    gpu_density.data());
  gpu_calc1<<<(bins - 1) / 128 + 1, 128>>>(
    bins, gpu_com_vx.data(), gpu_com_vy.data(), gpu_com_vz.data(), gpu_density.data());
  // get spatial thermo info
  gpu_thermo<<<(n - 1) / 128 + 1, 128>>>(
    n,
    avg_window,
    atom.position_per_atom.data() + direction * n,
    atom.mass.data(),
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + n,
    atom.velocity_per_atom.data() + 2 * n,
    atom.virial_per_atom.data(),
    atom.virial_per_atom.data() + 1 * n,
    atom.virial_per_atom.data() + 2 * n,
    gpu_temp.data(),
    gpu_com_vx.data(),
    gpu_com_vy.data(),
    gpu_com_vz.data(),
    gpu_pxx.data(),
    gpu_pyy.data(),
    gpu_pzz.data(),
    gpu_number.data());
  gpu_calc2<<<(bins - 1) / 128 + 1, 128>>>(
    bins,
    slice_vol,
    gpu_temp.data(),
    gpu_pxx.data(),
    gpu_pyy.data(),
    gpu_pzz.data(),
    gpu_density.data(),
    gpu_number.data());
  // copy from gpu to cpu
  gpu_temp.copy_to_host(cpu_temp.data());
  gpu_pxx.copy_to_host(cpu_pxx.data());
  gpu_pyy.copy_to_host(cpu_pyy.data());
  gpu_pzz.copy_to_host(cpu_pzz.data());
  gpu_density.copy_to_host(cpu_density.data());
  gpu_com_vx.copy_to_host(cpu_com_vx.data());
  gpu_com_vy.copy_to_host(cpu_com_vy.data());
  gpu_com_vz.copy_to_host(cpu_com_vz.data());
  gpu_com_vx.copy_to_host(cpu_com_vx.data());
  gpu_com_vy.copy_to_host(cpu_com_vy.data());
  gpu_com_vz.copy_to_host(cpu_com_vz.data());
  // write to file
  for (int i = 0; i < bins; i++) {
    cpu_com_vx[i] /= 0.01 * TIME_UNIT_CONVERSION; // to km/s
    cpu_com_vy[i] /= 0.01 * TIME_UNIT_CONVERSION; // to km/s
    cpu_com_vz[i] /= 0.01 * TIME_UNIT_CONVERSION; // to km/s
    cpu_density[i] *= 1.660538921;                // to g/cm3
  }
  write_to_file(temp_file, cpu_temp.data(), bins);
  write_to_file(pxx_file, cpu_pxx.data(), bins);
  write_to_file(pyy_file, cpu_pyy.data(), bins);
  write_to_file(pzz_file, cpu_pzz.data(), bins);
  write_to_file(density_file, cpu_density.data(), bins);
  write_to_file(com_vx_file, cpu_com_vx.data(), bins);
}

void Dump_Shock_NEMD::postprocess()
{
  if (!dump_)
    return;
  printf("Closing files ...\n");
  fclose(temp_file);
  fclose(pxx_file);
  fclose(pyy_file);
  fclose(pzz_file);
  fclose(density_file);
  fclose(com_vx_file);
  dump_ = false;
}