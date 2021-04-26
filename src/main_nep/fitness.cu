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
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

static void
find_permuted_indices(char* input_dir, GPU_Vector<int>& Na, std::vector<int>& permuted_indices)
{
  std::mt19937 rng;
#ifdef DEBUG
  rng = std::mt19937(54321);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
  for (int i = 0; i < permuted_indices.size(); ++i) {
    permuted_indices[i] = i;
  }
  std::uniform_int_distribution<int> rand_int(0, INT_MAX);
  for (int i = 0; i < permuted_indices.size(); ++i) {
    int j = rand_int(rng) % (permuted_indices.size() - i) + i;
    int temp = permuted_indices[i];
    permuted_indices[i] = permuted_indices[j];
    permuted_indices[j] = temp;
  }

  char filename[200];
  strcpy(filename, input_dir);
  strcat(filename, "/configuration_id.out");
  FILE* fid = my_fopen(filename, "w");
  for (int i = 0; i < permuted_indices.size(); ++i) {
    fprintf(fid, "%d %d\n", permuted_indices[i], Na[permuted_indices[i]]);
  }
  fclose(fid);
}

Fitness::Fitness(char* input_dir, Parameters& para)
{
  data_set.read_train_in(input_dir, para);
  std::vector<int> configuration_id(data_set.Nc);
  find_permuted_indices(input_dir, data_set.Na, configuration_id);

  print_line_1();
  printf("Started building the training set:\n");
  print_line_2();

  data_set.make_train_or_test_set(
    para, data_set.Nc - para.test_set_size, 0, configuration_id, train_set);

  print_line_1();
  printf("Started building the testing set:\n");
  print_line_2();
  data_set.make_train_or_test_set(
    para, para.test_set_size, data_set.Nc - para.test_set_size, configuration_id, test_set);

  potential_train.reset(new NEP2(para, train_set));
  potential_test.reset(new NEP2(para, test_set));

  char file_train_out[200];
  strcpy(file_train_out, input_dir);
  strcat(file_train_out, "/loss.out");
  fid_train_out = my_fopen(file_train_out, "w");

  char file_potential_out[200];
  strcpy(file_potential_out, input_dir);
  strcat(file_potential_out, "/ann.out");
  fid_potential_out = my_fopen(file_potential_out, "w");
}

Fitness::~Fitness()
{
  fclose(fid_train_out);
  fclose(fid_potential_out);
}

void Fitness::compute(
  const int generation, Parameters& para, const float* population, float* fitness)
{
  const int num_of_batches = (train_set.Nc - 1) / para.batch_size + 1;
  const int batch_id = generation % num_of_batches;
  const int configuration_start = batch_id * para.batch_size;
  const int configuration_end = std::min(train_set.Nc, configuration_start + para.batch_size);
  for (int n = 0; n < para.population_size; ++n) {
    const float* individual = population + n * para.number_of_variables;
    potential_train->find_force(configuration_start, configuration_end, individual, train_set);
    fitness[n + 0 * para.population_size] =
      train_set.get_rmse_energy(configuration_start, configuration_end);
    fitness[n + 1 * para.population_size] = train_set.get_rmse_force(
      train_set.Na_sum[configuration_start],
      train_set.Na_sum[configuration_end - 1] + train_set.Na[configuration_end - 1]);
    fitness[n + 2 * para.population_size] =
      train_set.get_rmse_virial(configuration_start, configuration_end);
  }
}

static void
predict_energy_or_stress(FILE* fid, int Nc, int* Na, int* Na_sum, float* data, float* ref)
{
  for (int nc = 0; nc < Nc; ++nc) {
    int offset = Na_sum[nc];
    float data_nc = 0.0f;
    for (int m = 0; m < Na[nc]; ++m) {
      data_nc += data[offset + m];
    }
    fprintf(fid, "%g %g\n", data_nc / Na[nc], ref[nc]);
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
  if (0 == (generation + 1) % 1000) {
    char file_nep[200];
    strcpy(file_nep, input_dir);
    strcat(file_nep, "/nep.out");
    FILE* fid_nep = my_fopen(file_nep, "w");

    fprintf(fid_nep, "nep 1\n"); // TODO: output the correct # types
    fprintf(fid_nep, "cutoff %g\n", para.rc);
    fprintf(fid_nep, "n_max %d\n", para.n_max);
    fprintf(fid_nep, "l_max %d\n", para.L_max);
    fprintf(fid_nep, "num_neurons1 %d\n", para.num_neurons1);
    fprintf(fid_nep, "num_neurons2 %d\n", para.num_neurons1);
    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid_nep, "%15.7e ", elite[m]);
    }
    fprintf(fid_nep, "\n");

    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid_potential_out, "%15.7e ", elite[m]);
    }
    fprintf(fid_potential_out, "\n");
    fflush(fid_potential_out);

    potential_train->find_force(0, train_set.Nc, elite, train_set);
    potential_test->find_force(0, test_set.Nc, elite, test_set);
    float rmse_energy_test = test_set.get_rmse_energy(0, test_set.Nc);
    float rmse_force_test = test_set.get_rmse_force(0, test_set.N);
    float rmse_virial_test = test_set.get_rmse_virial(0, test_set.Nc);

    printf(
      "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-12.5f%-12.5f%-12.5f\n", generation + 1,
      loss_total, loss_L1, loss_L2, loss_energy, loss_force, loss_virial, rmse_energy_test,
      rmse_force_test, rmse_virial_test);
    fflush(stdout);
    fprintf(
      fid_train_out, "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-12.5f%-12.5f%-12.5f\n",
      generation + 1, loss_total, loss_L1, loss_L2, loss_energy, loss_force, loss_virial,
      rmse_energy_test, rmse_force_test, rmse_virial_test);
    fflush(fid_train_out);

    // Synchronize
    CHECK(cudaDeviceSynchronize());

    // update force.out
    char file_force[200];
    strcpy(file_force, input_dir);
    strcat(file_force, "/force.out");
    FILE* fid_force = my_fopen(file_force, "w");
    for (int n = 0; n < train_set.N; ++n) {
      fprintf(
        fid_force, "%g %g %g %g %g %g\n", train_set.force[n], train_set.force[n + train_set.N],
        train_set.force[n + train_set.N * 2], train_set.force_ref[n],
        train_set.force_ref[n + train_set.N], train_set.force_ref[n + train_set.N * 2]);
    }
    for (int n = 0; n < test_set.N; ++n) {
      fprintf(
        fid_force, "%g %g %g %g %g %g\n", test_set.force[n], test_set.force[n + test_set.N],
        test_set.force[n + test_set.N * 2], test_set.force_ref[n],
        test_set.force_ref[n + test_set.N], test_set.force_ref[n + test_set.N * 2]);
    }
    fclose(fid_force);

    // update energy.out
    char file_energy[200];
    strcpy(file_energy, input_dir);
    strcat(file_energy, "/energy.out");
    FILE* fid_energy = my_fopen(file_energy, "w");
    predict_energy_or_stress(
      fid_energy, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(), train_set.pe.data(),
      train_set.pe_ref.data());
    predict_energy_or_stress(
      fid_energy, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(), test_set.pe.data(),
      test_set.pe_ref.data());
    fclose(fid_energy);

    // update virial.out
    char file_virial[200];
    strcpy(file_virial, input_dir);
    strcat(file_virial, "/virial.out");
    FILE* fid_virial = my_fopen(file_virial, "w");
    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data(), train_set.virial_ref.data());
    predict_energy_or_stress(
      fid_virial, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(), test_set.virial.data(),
      test_set.virial_ref.data());

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N, train_set.virial_ref.data() + train_set.Nc);
    predict_energy_or_stress(
      fid_virial, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(),
      test_set.virial.data() + test_set.N, test_set.virial_ref.data() + test_set.Nc);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 2, train_set.virial_ref.data() + train_set.Nc * 2);
    predict_energy_or_stress(
      fid_virial, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(),
      test_set.virial.data() + test_set.N * 2, test_set.virial_ref.data() + test_set.Nc * 2);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 3, train_set.virial_ref.data() + train_set.Nc * 3);

    predict_energy_or_stress(
      fid_virial, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(),
      test_set.virial.data() + test_set.N * 3, test_set.virial_ref.data() + test_set.Nc * 3);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 4, train_set.virial_ref.data() + train_set.Nc * 4);

    predict_energy_or_stress(
      fid_virial, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(),
      test_set.virial.data() + test_set.N * 4, test_set.virial_ref.data() + test_set.Nc * 4);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 5, train_set.virial_ref.data() + train_set.Nc * 5);

    predict_energy_or_stress(
      fid_virial, test_set.Nc, test_set.Na.data(), test_set.Na_sum.data(),
      test_set.virial.data() + test_set.N * 5, test_set.virial_ref.data() + test_set.Nc * 5);

    fclose(fid_virial);
  }
}
