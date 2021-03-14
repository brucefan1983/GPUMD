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
Get the fitness
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "fitness.cuh"
#include "gpu_vector.cuh"
#include "neighbor.cuh"
#include "nn2b.cuh"
#include <vector>

Fitness::Fitness(char* input_dir)
{
  read_potential(input_dir);
  read_train_in(input_dir);
  neighbor.compute(Nc, N, max_Na, Na.data(), Na_sum.data(), r.data(), h.data());
  potential->initialize(N, max_Na);
  error_cpu.resize(Nc);
  error_gpu.resize(Nc);
}

static void get_inverse(float* cpu_h)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  float volume = cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
                 cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
                 cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= volume;
  }
}

void Fitness::read_train_in(char* input_dir)
{
  print_line_1();
  printf("Started reading train.in.\n");
  print_line_2();

  char file_train[200];
  strcpy(file_train, input_dir);
  strcat(file_train, "/train.in");
  FILE* fid = my_fopen(file_train, "r");

  // get Nc and Nc_force
  read_Nc(fid);
  h.resize(Nc * 18, Memory_Type::managed);
  pe_ref.resize(Nc, Memory_Type::managed);
  virial_ref.resize(Nc * 6, Memory_Type::managed);
  Na.resize(Nc, Memory_Type::managed);
  Na_sum.resize(Nc, Memory_Type::managed);

  read_Na(fid);
  type.resize(N, Memory_Type::managed);
  r.resize(N * 3, Memory_Type::managed);
  force.resize(N * 3, Memory_Type::managed);
  force_ref.resize(N * 3, Memory_Type::managed);
  pe.resize(N, Memory_Type::managed);
  virial.resize(N * 6, Memory_Type::managed);

  float energy_minimum = 0.0;
  potential_square_sum = 0.0;
  virial_square_sum = 0.0;
  force_square_sum = 0.0;

  for (int n = 0; n < Nc; ++n) {
    int count;

    // energy, virial

    count = fscanf(
      fid, "%f%f%f%f%f%f%f", &pe_ref[n], &virial_ref[n + Nc * 0], &virial_ref[n + Nc * 1],
      &virial_ref[n + Nc * 2], &virial_ref[n + Nc * 3], &virial_ref[n + Nc * 4],
      &virial_ref[n + Nc * 5]);
    if (count != 7) {
      print_error("reading error for train.in.\n");
    }
    if (pe_ref[n] < energy_minimum)
      energy_minimum = pe_ref[n];

    // box (transpose of VASP input matrix)
    float h_tmp[9];
    for (int k = 0; k < 9; ++k) {
      count = fscanf(fid, "%f", &h_tmp[k]);
      if (count != 1) {
        print_error("reading error for train.in.\n");
      }
    }
    h[0 + 18 * n] = h_tmp[0];
    h[3 + 18 * n] = h_tmp[1];
    h[6 + 18 * n] = h_tmp[2];
    h[1 + 18 * n] = h_tmp[3];
    h[4 + 18 * n] = h_tmp[4];
    h[7 + 18 * n] = h_tmp[5];
    h[2 + 18 * n] = h_tmp[6];
    h[5 + 18 * n] = h_tmp[7];
    h[8 + 18 * n] = h_tmp[8];

    get_inverse(h.data() + 18 * n);

    // type, position, force
    for (int k = 0; k < Na[n]; ++k) {
      count = fscanf(
        fid, "%d%f%f%f%f%f%f", &type[Na_sum[n] + k], &r[Na_sum[n] + k], &r[Na_sum[n] + k + N],
        &r[Na_sum[n] + k + N * 2], &force_ref[Na_sum[n] + k], &force_ref[Na_sum[n] + k + N],
        &force_ref[Na_sum[n] + k + N * 2]);
      if (count != 7) {
        print_error("reading error for train.in.\n");
      }
      force_square_sum += force_ref[Na_sum[n] + k] * force_ref[Na_sum[n] + k] +
                          force_ref[Na_sum[n] + k + N] * force_ref[Na_sum[n] + k + N] +
                          force_ref[Na_sum[n] + k + N * 2] * force_ref[Na_sum[n] + k + N * 2];
    }
  }

  fclose(fid);

  for (int n = 0; n < Nc; ++n) {
    float energy = pe_ref[n] - energy_minimum;
    potential_square_sum += energy * energy;
    for (int k = 0; k < 6; ++k) {
      virial_square_sum += virial_ref[n + Nc * k] * virial_ref[n + Nc * k];
    }
  }
}

void Fitness::read_Nc(FILE* fid)
{
  int count = fscanf(fid, "%d", &Nc);
  if (count != 1)
    print_error("Reading error for xyz.in.\n");

  if (Nc < 2) {
    print_error("Number of configurations should >= 2\n");
  }
  printf("Number of configurations is %d:\n", Nc);
}

void Fitness::read_Na(FILE* fid)
{
  N = 0;
  max_Na = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    Na_sum[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    int count = fscanf(fid, "%d", &Na[nc]);
    if (count != 1) {
      print_error("Reading error for train.in.\n");
    }
    N += Na[nc];
    if (Na[nc] > max_Na) {
      max_Na = Na[nc];
    }
    if (Na[nc] < 2) {
      print_error("Number of atoms %d should >= 2\n");
    }
  }

  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum[nc] = Na_sum[nc - 1] + Na[nc - 1];
  }

  printf("Total number of atoms is %d:\n", N);
  printf("    %d atoms in the largest configuration;\n", max_Na);
}

void Fitness::read_potential(char* input_dir)
{
  print_line_1();
  printf("Started reading potential.in.\n");
  print_line_2();

  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/potential.in");
  FILE* fid = my_fopen(file, "r");

  char name[20];

  int count = fscanf(fid, "%s%d", name, &potential_type);
  if (count != 2) {
    print_error("reading error for potential.in.");
  }
  if (potential_type == 0) {
    int num_neurons_per_layer = 0;
    count = fscanf(fid, "%s%d", name, &num_neurons_per_layer);
    if (count != 2) {
      print_error("reading error for potential.in.");
    }
    printf("num_neurons_per_layer is %d.\n", num_neurons_per_layer);
    number_of_variables = num_neurons_per_layer * (num_neurons_per_layer + 4) + 1;
    printf("Use the NN2B potential with %d parameters.\n", number_of_variables);
    potential = std::make_unique<NN2B>(num_neurons_per_layer);
  } else {
    print_error("unsupported potential type.\n");
  }

  count = fscanf(fid, "%s%f", name, &neighbor.cutoff);
  if (count != 2) {
    print_error("reading error for potential.in.");
  }
  printf("cutoff for neighbor list is %g A.\n", neighbor.cutoff);

  count = fscanf(fid, "%s%f", name, &cost.weight_force);
  if (count != 2) {
    print_error("reading error for potential.in.");
  }
  if (cost.weight_force < 0) {
    print_error("weight for force should >= 0\n");
  }
  printf("weight for force is %g.\n", cost.weight_force);

  count = fscanf(fid, "%s%f", name, &cost.weight_energy);
  if (count != 2) {
    print_error("reading error for potential.in.");
  }
  if (cost.weight_energy < 0) {
    print_error("weight for energy should >= 0\n");
  }
  printf("weight for energy is %g.\n", cost.weight_energy);

  count = fscanf(fid, "%s%f", name, &cost.weight_stress);
  if (count != 2) {
    print_error("reading error for potential.in.");
  }
  if (cost.weight_stress < 0) {
    print_error("weight for stress should >= 0\n");
  }
  printf("weight for stress is %g.\n", cost.weight_stress);

  fclose(fid);
}

void Fitness::compute(const int population_size, const float* population, float* fitness)
{
  for (int n = 0; n < population_size; ++n) {
    const float* individual = population + n * number_of_variables;
    potential->update_potential(individual);
    potential->find_force(
      Nc, N, Na.data(), Na_sum.data(), max_Na, type.data(), h.data(), &neighbor, r.data(), force,
      virial, pe);
    fitness[n] = cost.weight_energy * get_fitness_energy();
    fitness[n] += cost.weight_stress * get_fitness_stress();
    fitness[n] += cost.weight_force * get_fitness_force();
  }
}

void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref)
{
  for (int nc = 0; nc < Nc; ++nc) {
    int offset = Na_sum[nc];
    float data_nc = 0.0;
    for (int m = 0; m < Na[nc]; ++m) {
      data_nc += data[offset + m];
    }
    fprintf(fid, "%g %g\n", data_nc, ref[nc]);
  }
}

void Fitness::predict(char* input_dir, const float* elite)
{
  potential->update_potential(elite);
  potential->find_force(
    Nc, N, Na.data(), Na_sum.data(), max_Na, type.data(), h.data(), &neighbor, r.data(), force,
    virial, pe);

  CHECK(cudaDeviceSynchronize());

  char file_force[200];
  strcpy(file_force, input_dir);
  strcat(file_force, "/force.out");
  FILE* fid_force = my_fopen(file_force, "w");
  for (int n = 0; n < N; ++n) {
    fprintf(
      fid_force, "%g %g %g %g %g %g\n", force[n], force[n + N], force[n + N * 2], force_ref[n],
      force_ref[n + N], force_ref[n + N * 2]);
  }
  fclose(fid_force);

  char file_energy[200];
  strcpy(file_energy, input_dir);
  strcat(file_energy, "/energy.out");
  FILE* fid_energy = my_fopen(file_energy, "w");
  predict_energy_or_stress(fid_energy, pe.data(), pe_ref.data());
  fclose(fid_energy);

  char file_virial[200];
  strcpy(file_virial, input_dir);
  strcat(file_virial, "/virial.out");
  FILE* fid_virial = my_fopen(file_virial, "w");
  predict_energy_or_stress(fid_virial, virial.data(), virial_ref.data());
  predict_energy_or_stress(fid_virial, virial.data() + N, virial_ref.data() + Nc);
  predict_energy_or_stress(fid_virial, virial.data() + N * 2, virial_ref.data() + Nc * 2);
  predict_energy_or_stress(fid_virial, virial.data() + N * 3, virial_ref.data() + Nc * 3);
  predict_energy_or_stress(fid_virial, virial.data() + N * 4, virial_ref.data() + Nc * 4);
  predict_energy_or_stress(fid_virial, virial.data() + N * 5, virial_ref.data() + Nc * 5);
  fclose(fid_virial);
}

static __global__ void gpu_sum_force_error(
  int N,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_fx_ref,
  float* g_fy_ref,
  float* g_fz_ref,
  float* g_error)
{
  int tid = threadIdx.x;
  int number_of_rounds = (N - 1) / blockDim.x + 1;
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * blockDim.x;
    if (n < N) {
      float dx = g_fx[n] - g_fx_ref[n];
      float dy = g_fy[n] - g_fy_ref[n];
      float dz = g_fz[n] - g_fz_ref[n];
      s_error[tid] += dx * dx + dy * dy + dz * dz;
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    g_error[0] = s_error[0];
  }
}

float Fitness::get_fitness_force(void)
{
  gpu_sum_force_error<<<1, 512, sizeof(float) * 512>>>(
    N, force.data(), force.data() + N, force.data() + N * 2, force_ref.data(), force_ref.data() + N,
    force_ref.data() + N * 2, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), sizeof(float), cudaMemcpyDeviceToHost));
  return sqrt(error_cpu[0] / force_square_sum);
}

static __global__ void
gpu_sum_pe_error(int* g_Na, int* g_Na_sum, float* g_pe, float* g_pe_ref, float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int offset = g_Na_sum[bid];
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;
  if (tid < Na) {
    int n = offset + tid; // particle index
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    float diff = s_pe[0] - g_pe_ref[bid];
    error_gpu[bid] = diff * diff;
  }
}

static int get_block_size(int max_num_atom)
{
  int block_size = 64;
  for (int n = 64; n < 1024; n <<= 1) {
    if (max_num_atom > n) {
      block_size = n << 1;
    }
  }
  return block_size;
}

float Fitness::get_fitness_energy(void)
{
  int block_size = get_block_size(max_Na);
  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), pe.data(), pe_ref.data(), error_gpu.data());
  int mem = sizeof(float) * Nc;
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  float error_ave = 0.0;
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }
  return sqrt(error_ave / potential_square_sum);
}

float Fitness::get_fitness_stress(void)
{
  float error_ave = 0.0;
  int mem = sizeof(float) * Nc;
  int block_size = get_block_size(max_Na);

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data(), virial_ref.data(), error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N, virial_ref.data() + Nc, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 2, virial_ref.data() + Nc * 2, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 3, virial_ref.data() + Nc * 3, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 4, virial_ref.data() + Nc * 4, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 5, virial_ref.data() + Nc * 5, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }

  return sqrt(error_ave / virial_square_sum);
}
