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
#include "neighbor.cuh"
#include "nep.cuh"
#include "nep2.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

Fitness::Fitness(char* input_dir, Parameters& para)
{
  potential.reset(new NEP2(para.rc, para.num_neurons, para.n_max, para.L_max));
  neighbor.cutoff = para.rc;

  training_set.read_train_in(input_dir);
  neighbor.compute(
    training_set.Nc, training_set.N, training_set.max_Na, training_set.Na.data(),
    training_set.Na_sum.data(), training_set.r.data(), training_set.h.data());
  potential->initialize(training_set.N, training_set.max_Na);
  training_set.error_cpu.resize(training_set.Nc);
  training_set.error_gpu.resize(training_set.Nc);

  char file_train_out[200];
  strcpy(file_train_out, input_dir);
  strcat(file_train_out, "/train.out");
  fid_train_out = my_fopen(file_train_out, "w");
}

void Fitness::compute(Parameters& para, const float* population, float* fitness)
{
  for (int n = 0; n < para.population_size; ++n) {
    const float* individual = population + n * para.number_of_variables;
    potential->update_potential(individual);
    potential->find_force(
      training_set.Nc, training_set.N, training_set.Na.data(), training_set.Na_sum.data(),
      training_set.max_Na, training_set.atomic_number.data(), training_set.h.data(), &neighbor,
      training_set.r.data(), training_set.force, training_set.virial, training_set.pe);
    fitness[n + 0 * para.population_size] =
      para.weight_energy * training_set.get_fitness_energy() / training_set.potential_std;
    fitness[n + 1 * para.population_size] =
      para.weight_force * training_set.get_fitness_force() / training_set.force_std;
    fitness[n + 2 * para.population_size] =
      para.weight_stress * training_set.get_fitness_stress() / training_set.virial_std;
  }
}

void Fitness::report_error(
  char* input_dir,
  Parameters& para,
  const int generation,
  const float loss_total,
  const float loss_L1,
  const float loss_L2,
  const float* elite)
{
  if (0 == (generation + 1) % 100) {
    // save a potential
    char file[200];
    strcpy(file, input_dir);
    strcat(file, "/potential.out");
    FILE* fid = my_fopen(file, "w");
    fprintf(fid, "nep 1\n");
    fprintf(fid, "cutoff %g\n", para.rc);
    fprintf(fid, "num_neurons %d\n", para.num_neurons);
    fprintf(fid, "n_max %d\n", para.n_max);
    fprintf(fid, "l_max %d\n", para.L_max);
    for (int m = 0; m < para.number_of_variables; ++m) {
      fprintf(fid, "%g ", elite[m]);
    }
    fprintf(fid, "\n");
    fclose(fid);

    // calculate force, energy, and virial
    potential->update_potential(elite);
    potential->find_force(
      training_set.Nc, training_set.N, training_set.Na.data(), training_set.Na_sum.data(),
      training_set.max_Na, training_set.atomic_number.data(), training_set.h.data(), &neighbor,
      training_set.r.data(), training_set.force, training_set.virial, training_set.pe);

    // report errors
    float rmse_energy = training_set.get_fitness_energy();
    float rmse_force = training_set.get_fitness_force();
    float rmse_virial = training_set.get_fitness_stress();
    printf(
      "%-7d%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-12.2f%-10.2f%-12.2f\n", generation + 1,
      loss_total * 100.0f, loss_L1 * 100.0f, loss_L2 * 100.0f,
      rmse_energy / training_set.potential_std * 100.0f,
      rmse_force / training_set.force_std * 100.0f, rmse_virial / training_set.virial_std * 100.0f,
      rmse_energy * 1000.0f, rmse_force * 1000.0f, rmse_virial * 1000.0f);
    fprintf(
      fid_train_out, "%-7d%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-10.2f%-12.2f%-10.2f%-12.2f\n",
      generation + 1, loss_total * 100.0f, loss_L1 * 100.0f, loss_L2 * 100.0f,
      rmse_energy / training_set.potential_std * 100.0f,
      rmse_force / training_set.force_std * 100.0f, rmse_virial / training_set.virial_std * 100.0f,
      rmse_energy * 1000.0f, rmse_force * 1000.0f, rmse_virial * 1000.0f);

    // Synchronize
    CHECK(cudaDeviceSynchronize());

    // update force.out
    char file_force[200];
    strcpy(file_force, input_dir);
    strcat(file_force, "/force.out");
    FILE* fid_force = my_fopen(file_force, "w");
    for (int n = 0; n < training_set.N; ++n) {
      fprintf(
        fid_force, "%g %g %g %g %g %g\n", training_set.force[n],
        training_set.force[n + training_set.N], training_set.force[n + training_set.N * 2],
        training_set.force_ref[n], training_set.force_ref[n + training_set.N],
        training_set.force_ref[n + training_set.N * 2]);
    }
    fclose(fid_force);

    // update energy.out
    char file_energy[200];
    strcpy(file_energy, input_dir);
    strcat(file_energy, "/energy.out");
    FILE* fid_energy = my_fopen(file_energy, "w");
    predict_energy_or_stress(fid_energy, training_set.pe.data(), training_set.pe_ref.data());
    fclose(fid_energy);

    // update virial.out
    char file_virial[200];
    strcpy(file_virial, input_dir);
    strcat(file_virial, "/virial.out");
    FILE* fid_virial = my_fopen(file_virial, "w");
    predict_energy_or_stress(
      fid_virial, training_set.virial.data(), training_set.virial_ref.data());
    predict_energy_or_stress(
      fid_virial, training_set.virial.data() + training_set.N,
      training_set.virial_ref.data() + training_set.Nc);
    predict_energy_or_stress(
      fid_virial, training_set.virial.data() + training_set.N * 2,
      training_set.virial_ref.data() + training_set.Nc * 2);
    predict_energy_or_stress(
      fid_virial, training_set.virial.data() + training_set.N * 3,
      training_set.virial_ref.data() + training_set.Nc * 3);
    predict_energy_or_stress(
      fid_virial, training_set.virial.data() + training_set.N * 4,
      training_set.virial_ref.data() + training_set.Nc * 4);
    predict_energy_or_stress(
      fid_virial, training_set.virial.data() + training_set.N * 5,
      training_set.virial_ref.data() + training_set.Nc * 5);
    fclose(fid_virial);
  }
}

void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref)
{
  for (int nc = 0; nc < training_set.Nc; ++nc) {
    int offset = training_set.Na_sum[nc];
    float data_nc = 0.0;
    for (int m = 0; m < training_set.Na[nc]; ++m) {
      data_nc += data[offset + m];
    }
    fprintf(fid, "%g %g\n", data_nc / training_set.Na[nc], ref[nc]);
  }
}
