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
  train_set.construct(input_dir, para);
  potential.reset(new NEP2(input_dir, para, train_set));

  char file_loss_out[200];
  strcpy(file_loss_out, input_dir);
  strcat(file_loss_out, "/loss.out");
  fid_loss_out = my_fopen(file_loss_out, "w");
}

Fitness::~Fitness() { fclose(fid_loss_out); }

void Fitness::compute(
  const int generation, Parameters& para, const float* population, float* fitness)
{
  for (int n = 0; n < para.population_size; ++n) {
    const float* individual = population + n * para.number_of_variables;
    potential->find_force(para, individual, train_set);
    fitness[n + 0 * para.population_size] = train_set.get_rmse_energy();
    fitness[n + 1 * para.population_size] = train_set.get_rmse_force();
    fitness[n + 2 * para.population_size] = train_set.get_rmse_virial();
  }
}

void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref)
{
  for (int nc = 0; nc < train_set.Nc; ++nc) {
    int offset = train_set.Na_sum[nc];
    float data_nc = 0.0f;
    for (int m = 0; m < train_set.Na[nc]; ++m) {
      data_nc += data[offset + m];
    }
    fprintf(fid, "%g %g\n", data_nc / train_set.Na[nc], ref[nc]);
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
    strcat(file_nep, "/nep.txt");
    FILE* fid_nep = my_fopen(file_nep, "w");
    fprintf(fid_nep, "nep %d\n", train_set.num_types);
    fprintf(fid_nep, "cutoff %g %g\n", para.rc_radial, para.rc_angular);
    fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
    fprintf(fid_nep, "l_max %d\n", para.L_max);
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid_nep, "%15.7e ", elite[m]);
    }
    fprintf(fid_nep, "\n");
    for (int d = 0; d < para.q_scaler.size(); ++d) {
      fprintf(fid_nep, "%15.7e %15.7e\n", para.q_scaler[d], para.q_min[d]);
    }
    fclose(fid_nep);

    potential->find_force(para, elite, train_set);
    float rmse_energy_train = train_set.get_rmse_energy();
    float rmse_force_train = train_set.get_rmse_force();
    float rmse_virial_train = train_set.get_rmse_virial();

    printf(
      "%-8d%-11.5f%-11.5f%-11.5f%-12.5f%-12.5f%-12.5f\n", generation + 1, loss_total, loss_L1,
      loss_L2, rmse_energy_train, rmse_force_train, rmse_virial_train);
    fflush(stdout);
    fprintf(
      fid_loss_out, "%-8d%-11.5f%-11.5f%-11.5f%-12.5f%-12.5f%-12.5f\n", generation + 1, loss_total,
      loss_L1, loss_L2, rmse_energy_train, rmse_force_train, rmse_virial_train);
    fflush(fid_loss_out);

    update_energy_force_virial(input_dir);
  }
}

void Fitness::update_energy_force_virial(char* input_dir)
{
  // update force.out
  char file_force[200];
  strcpy(file_force, input_dir);
  strcat(file_force, "/force.out");
  FILE* fid_force = my_fopen(file_force, "w");
  for (int nc = 0; nc < train_set.Nc; ++nc) {
    int offset = train_set.Na_sum[nc];
    for (int m = 0; m < train_set.Na_original[nc]; ++m) {
      int n = offset + m;
      fprintf(
        fid_force, "%g %g %g %g %g %g\n", train_set.force[n], train_set.force[n + train_set.N],
        train_set.force[n + train_set.N * 2], train_set.force_ref[n],
        train_set.force_ref[n + train_set.N], train_set.force_ref[n + train_set.N * 2]);
    }
  }
  fclose(fid_force);

  // update energy.out
  char file_energy[200];
  strcpy(file_energy, input_dir);
  strcat(file_energy, "/energy.out");
  FILE* fid_energy = my_fopen(file_energy, "w");
  predict_energy_or_stress(fid_energy, train_set.pe.data(), train_set.pe_ref.data());
  fclose(fid_energy);

  // update virial.out
  char file_virial[200];
  strcpy(file_virial, input_dir);
  strcat(file_virial, "/virial.out");
  FILE* fid_virial = my_fopen(file_virial, "w");
  predict_energy_or_stress(fid_virial, train_set.virial.data(), train_set.virial_ref.data());

  predict_energy_or_stress(
    fid_virial, train_set.virial.data() + train_set.N, train_set.virial_ref.data() + train_set.Nc);

  predict_energy_or_stress(
    fid_virial, train_set.virial.data() + train_set.N * 2,
    train_set.virial_ref.data() + train_set.Nc * 2);

  predict_energy_or_stress(
    fid_virial, train_set.virial.data() + train_set.N * 3,
    train_set.virial_ref.data() + train_set.Nc * 3);

  predict_energy_or_stress(
    fid_virial, train_set.virial.data() + train_set.N * 4,
    train_set.virial_ref.data() + train_set.Nc * 4);

  predict_energy_or_stress(
    fid_virial, train_set.virial.data() + train_set.N * 5,
    train_set.virial_ref.data() + train_set.Nc * 5);

  fclose(fid_virial);
}

void Fitness::test(char* input_dir, Parameters& para, const float* elite)
{
  potential->find_force(para, elite, train_set);
  float rmse_energy_train = train_set.get_rmse_energy();
  float rmse_force_train = train_set.get_rmse_force();
  float rmse_virial_train = train_set.get_rmse_virial();
  printf("Energy RMSE = %g\n", rmse_energy_train);
  printf("Force RMSE = %g\n", rmse_force_train);
  printf("Virial RMSE = %g\n", rmse_virial_train);
  update_energy_force_virial(input_dir);
}
