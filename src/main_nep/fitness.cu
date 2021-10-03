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

#include "fitness.cuh"
#include "nep.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

Fitness::Fitness(char* input_dir, Parameters& para)
{
  print_line_1();
  printf("Started reading train.in.\n");
  print_line_2();
  std::vector<Structure> structures_train;
  read_structures(true, input_dir, para, structures_train);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  printf("Number of batches = %d\n", num_batches);
  int batch_size_old = para.batch_size;
  para.batch_size = (structures_train.size() - 1) / num_batches + 1;
  if (batch_size_old != para.batch_size) {
    printf("Hello, I changed the batch_size from %d to %d.\n", batch_size_old, para.batch_size);
  }
  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    printf("\nBatch %d:\n", batch_id);
    int n1 = batch_id * para.batch_size;
    int n2 = std::min(int(structures_train.size()), n1 + para.batch_size);
    printf("Number of configurations = %d.\n", n2 - n1);
    train_set[batch_id].construct(input_dir, para, structures_train, n1, n2);
  }

  print_line_1();
  printf("Started reading test.in.\n");
  print_line_2();
  std::vector<Structure> structures_test;
  read_structures(false, input_dir, para, structures_test);
  test_set.construct(input_dir, para, structures_test, 0, structures_test.size());

  int N = test_set.N;
  int N_times_max_NN_radial = test_set.N * test_set.max_NN_radial;
  int N_times_max_NN_angular = test_set.N * test_set.max_NN_angular;
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n].N > N) {
      N = train_set[n].N;
    };
    if (train_set[n].N * train_set[n].max_NN_radial > N_times_max_NN_radial) {
      N_times_max_NN_radial = train_set[n].N * train_set[n].max_NN_radial;
    };
    if (train_set[n].N * train_set[n].max_NN_angular > N_times_max_NN_angular) {
      N_times_max_NN_angular = train_set[n].N * train_set[n].max_NN_angular;
    };
  }
  potential.reset(new NEP2(input_dir, para, N, N_times_max_NN_radial, N_times_max_NN_angular));

  char file_loss_out[200];
  strcpy(file_loss_out, input_dir);
  strcat(file_loss_out, "/loss.out");
  fid_loss_out = my_fopen(file_loss_out, "w");
}

Fitness::~Fitness() { fclose(fid_loss_out); }

void Fitness::compute(
  const int generation, Parameters& para, const float* population, float* fitness)
{
  int batch_id = generation % num_batches;
  for (int n = 0; n < para.population_size; ++n) {
    const float* individual = population + n * para.number_of_variables;
    potential->find_force(para, individual, train_set[batch_id]);
    fitness[n + 0 * para.population_size] = train_set[batch_id].get_rmse_energy();
    fitness[n + 1 * para.population_size] = train_set[batch_id].get_rmse_force();
    fitness[n + 2 * para.population_size] = train_set[batch_id].get_rmse_virial();
  }
}

void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref)
{
  for (int nc = 0; nc < test_set.Nc; ++nc) {
    int offset = test_set.Na_sum[nc];
    float data_nc = 0.0f;
    for (int m = 0; m < test_set.Na[nc]; ++m) {
      data_nc += data[offset + m];
    }
    fprintf(fid, "%g %g\n", data_nc / test_set.Na[nc], ref[nc]);
  }
}

void Fitness::report_error(
  char* input_dir,
  Parameters& para,
  const int generation,
  const float loss_total,
  const float loss_L1,
  const float loss_L2,
  const float loss_energy,
  const float loss_force,
  const float loss_virial,
  const float* elite)
{
  if (0 == (generation + 1) % 100) {

    potential->find_force(para, elite, test_set);
    float rmse_energy_test = test_set.get_rmse_energy();
    float rmse_force_test = test_set.get_rmse_force();
    float rmse_virial_test = test_set.get_rmse_virial();

    char file_nep[200];
    strcpy(file_nep, input_dir);
    strcat(file_nep, "/nep.txt");
    FILE* fid_nep = my_fopen(file_nep, "w");

    fprintf(fid_nep, "nep %d\n", para.num_types);
    fprintf(fid_nep, "cutoff %g %g\n", para.rc_radial, para.rc_angular);
    fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
    fprintf(fid_nep, "l_max %d\n", para.L_max);
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid_nep, "%15.7e ", elite[m]);
    }
    fprintf(fid_nep, "\n");
    para.q_scaler_gpu.copy_to_host(para.q_scaler_cpu.data());
    para.q_min_gpu.copy_to_host(para.q_min_cpu.data());
    for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
      fprintf(fid_nep, "%15.7e %15.7e\n", para.q_scaler_cpu[d], para.q_min_cpu[d]);
    }
    fclose(fid_nep);

    printf(
      "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n", generation + 1,
      loss_total, loss_L1, loss_L2, loss_energy, loss_force, loss_virial, rmse_energy_test,
      rmse_force_test, rmse_virial_test);
    fflush(stdout);
    fprintf(
      fid_loss_out, "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
      generation + 1, loss_total, loss_L1, loss_L2, loss_energy, loss_force, loss_virial,
      rmse_energy_test, rmse_force_test, rmse_virial_test);
    fflush(fid_loss_out);

    update_energy_force_virial(input_dir);
  }
}

void Fitness::update_energy_force_virial(char* input_dir)
{
  test_set.energy.copy_to_host(test_set.energy_cpu.data());
  test_set.virial.copy_to_host(test_set.virial_cpu.data());
  test_set.force.copy_to_host(test_set.force_cpu.data());

  // update force.out
  char file_force[200];
  strcpy(file_force, input_dir);
  strcat(file_force, "/force.out");
  FILE* fid_force = my_fopen(file_force, "w");
  for (int nc = 0; nc < test_set.Nc; ++nc) {
    int offset = test_set.Na_sum[nc];
    for (int m = 0; m < test_set.structures[nc].num_atom_original; ++m) {
      int n = offset + m;
      fprintf(
        fid_force, "%g %g %g %g %g %g\n", test_set.force_cpu[n], test_set.force_cpu[n + test_set.N],
        test_set.force_cpu[n + test_set.N * 2], test_set.force_ref_cpu[n],
        test_set.force_ref_cpu[n + test_set.N], test_set.force_ref_cpu[n + test_set.N * 2]);
    }
  }
  fclose(fid_force);

  // update energy.out
  char file_energy[200];
  strcpy(file_energy, input_dir);
  strcat(file_energy, "/energy.out");
  FILE* fid_energy = my_fopen(file_energy, "w");
  predict_energy_or_stress(fid_energy, test_set.energy_cpu.data(), test_set.energy_ref_cpu.data());
  fclose(fid_energy);

  // update virial.out
  char file_virial[200];
  strcpy(file_virial, input_dir);
  strcat(file_virial, "/virial.out");
  FILE* fid_virial = my_fopen(file_virial, "w");
  predict_energy_or_stress(fid_virial, test_set.virial_cpu.data(), test_set.virial_ref_cpu.data());

  predict_energy_or_stress(
    fid_virial, test_set.virial_cpu.data() + test_set.N,
    test_set.virial_ref_cpu.data() + test_set.Nc);

  predict_energy_or_stress(
    fid_virial, test_set.virial_cpu.data() + test_set.N * 2,
    test_set.virial_ref_cpu.data() + test_set.Nc * 2);

  predict_energy_or_stress(
    fid_virial, test_set.virial_cpu.data() + test_set.N * 3,
    test_set.virial_ref_cpu.data() + test_set.Nc * 3);

  predict_energy_or_stress(
    fid_virial, test_set.virial_cpu.data() + test_set.N * 4,
    test_set.virial_ref_cpu.data() + test_set.Nc * 4);

  predict_energy_or_stress(
    fid_virial, test_set.virial_cpu.data() + test_set.N * 5,
    test_set.virial_ref_cpu.data() + test_set.Nc * 5);

  fclose(fid_virial);
}

void Fitness::test(char* input_dir, Parameters& para, const float* elite)
{
  potential->find_force(para, elite, test_set);
  float rmse_energy_test = test_set.get_rmse_energy();
  float rmse_force_tes = test_set.get_rmse_force();
  float rmse_virial_tes = test_set.get_rmse_virial();
  printf("Energy RMSE = %g\n", rmse_energy_test);
  printf("Force RMSE = %g\n", rmse_force_tes);
  printf("Virial RMSE = %g\n", rmse_virial_tes);
  update_energy_force_virial(input_dir);
}
