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

#include "dump_piston.cuh"

namespace
{

static __global__ void gpu_com(
  int N,
  int bins,
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
    int l = (int)g_x[i];
    mass = g_mass[i];
    vx = g_vx[i];
    vy = g_vy[i];
    vz = g_vz[i];
    atomicAdd(&com_vx_data[l], vx * mass);
    atomicAdd(&com_vy_data[l], vy * mass);
    atomicAdd(&com_vz_data[l], vz * mass);
    atomicAdd(&density_data[l], mass);
  }
  __syncthreads();
  if (i < bins) {
    com_vx_data[i] /= density_data[i];
    com_vy_data[i] /= density_data[i];
    com_vz_data[i] /= density_data[i];
  }
}

static __global__ void gpu_thermo(
  int N,
  int bins,
  double slice_volume,
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
  double* density_data,
  int* number_data)
{
  double mass, vx, vy, vz;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int l = (int)g_x[i];
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
  __syncthreads();
  if (i < bins) {
    temp_data[i] /= 3 * number_data[i] * K_B;
    pxx_data[i] /= slice_volume;
    pyy_data[i] /= slice_volume;
    pzz_data[i] /= slice_volume;
    density_data[i] /= slice_volume;
  }
}
} // namespace

void Dump_Piston::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump spatial histogram thermo information for piston shock wave simulation.\n");

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }
  printf("    every %d steps.\n", dump_interval_);

  if (strcmp(param[2], "x") == 0) {
    direction = 0;
  } else if (strcmp(param[2], "y") == 0) {
    direction = 1;
  } else if (strcmp(param[2], "z") == 0) {
    direction = 2;
  } else
    PRINT_INPUT_ERROR("Direction should be x or y or z.");
  printf("    in %s direction.\n", direction);
}

void Dump_Piston::preprocess()
{
  if (dump_) {
    temp_file = my_fopen("temperature_hist.txt", "w");
    pxx_file = my_fopen("pxx_hist.txt", "w");
    pyy_file = my_fopen("pyy_hist.txt", "w");
    pzz_file = my_fopen("pzz_hist.txt", "w");
    density_file = my_fopen("density_hist.txt", "w");
  }
}
void Dump_Piston::process(Atom& atom, Box& box, const int step)
{
  int n = atom.number_of_atoms;
  bins = (int)box.cpu_h[direction] + 1;
  // create vectors to store hist
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
  // calculate COM velocity first
  gpu_com<<<(n - 1) / 128 + 1, 128>>>(
    n,
    bins,
    atom.mass.data(),
    atom.position_per_atom.data() + direction * n,
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + n,
    atom.velocity_per_atom.data() + 2 * n,
    gpu_com_vx.data(),
    gpu_com_vy.data(),
    gpu_com_vz.data(),
    gpu_density.data());
  // get spatial thermo info
  gpu_thermo<<<(n - 1) / 128 + 1, 128>>>(
    n,
    bins,
    1,
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
    gpu_density.data(),
    gpu_number.data());
  // copy from gpu to cpu
  // write to file
}

void Dump_Piston::postprocess()
{
  printf("Closing files ...\n");
  if (dump_) {
    fclose(temp_file);
    fclose(pxx_file);
    fclose(pyy_file);
    fclose(pzz_file);
    fclose(density_file);
    dump_ = false;
  }
}