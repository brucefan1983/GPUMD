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

Fitness::Fitness(char* input_dir, Parameters& para)
{
  train_set.read_train_in(input_dir, para);
  potential.reset(new NEP2(para, train_set));

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
    potential->find_force(para, configuration_start, configuration_end, individual, train_set);
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
  if (0 == (generation + 1) % 100) {
    // Synchronize
    CHECK(cudaDeviceSynchronize());

    char file_nep[200];
    strcpy(file_nep, input_dir);
    strcat(file_nep, "/nep.out");
    FILE* fid_nep = my_fopen(file_nep, "w");

    fprintf(fid_nep, "nep 1\n"); // TODO: output the correct # types
    fprintf(fid_nep, "cutoff %g %g\n", para.rc_radial, para.rc_angular);
    fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
    fprintf(fid_nep, "l_max %d\n", para.L_max);
    fprintf(fid_nep, "num_neurons1 %d\n", para.num_neurons1);
    fprintf(fid_nep, "num_neurons2 %d\n", para.num_neurons2);
    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid_nep, "%15.7e ", elite[m]);
    }
    fprintf(fid_nep, "\n");
    for (int d = 0; d < para.q_scaler.size(); ++d) {
      fprintf(fid_nep, "%15.7e %15.7e\n", para.q_scaler[d], para.q_min[d]);
    }
    fclose(fid_nep);

    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid_potential_out, "%15.7e ", elite[m]);
    }
    fprintf(fid_potential_out, "\n");
    fflush(fid_potential_out);

    potential->find_force(para, 0, train_set.Nc, elite, train_set);
    float rmse_energy_train = train_set.get_rmse_energy(0, train_set.Nc);
    float rmse_force_train = train_set.get_rmse_force(0, train_set.N);
    float rmse_virial_train = train_set.get_rmse_virial(0, train_set.Nc);

    printf(
      "%-8d%-11.5f%-11.5f%-11.5f%-12.5f%-12.5f%-12.5f\n", generation + 1, loss_total, loss_L1,
      loss_L2, rmse_energy_train, rmse_force_train, rmse_virial_train);
    fflush(stdout);
    fprintf(
      fid_train_out, "%-8d%-11.5f%-11.5f%-11.5f%-12.5f%-12.5f%-12.5f\n", generation + 1, loss_total,
      loss_L1, loss_L2, rmse_energy_train, rmse_force_train, rmse_virial_train);
    fflush(fid_train_out);

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
    fclose(fid_force);

    // update energy.out
    char file_energy[200];
    strcpy(file_energy, input_dir);
    strcat(file_energy, "/energy.out");
    FILE* fid_energy = my_fopen(file_energy, "w");
    predict_energy_or_stress(
      fid_energy, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(), train_set.pe.data(),
      train_set.pe_ref.data());
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
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N, train_set.virial_ref.data() + train_set.Nc);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 2, train_set.virial_ref.data() + train_set.Nc * 2);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 3, train_set.virial_ref.data() + train_set.Nc * 3);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 4, train_set.virial_ref.data() + train_set.Nc * 4);

    predict_energy_or_stress(
      fid_virial, train_set.Nc, train_set.Na.data(), train_set.Na_sum.data(),
      train_set.virial.data() + train_set.N * 5, train_set.virial_ref.data() + train_set.Nc * 5);

    fclose(fid_virial);
  }
}
