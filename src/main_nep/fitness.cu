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
#include "nep3.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

Fitness::Fitness(Parameters& para)
{
  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));

  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  num_batches = ((para.num_types + 1) * para.num_types) / 2;
  printf("Number of devices = %d\n", deviceCount);
  printf("Number of batches = %d\n", num_batches);

  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    train_set[batch_id].resize(deviceCount);
  }
  int batch_id = 0;
  for (int n1 = 0; n1 < para.num_types; ++n1) {
    for (int n2 = n1; n2 < para.num_types; ++n2) {
      printf("\nBatch %d:\n", batch_id);
      for (int device_id = 0; device_id < deviceCount; ++device_id) {
        print_line_1();
        printf("Constructing train_set in device  %d.\n", device_id);
        CHECK(cudaSetDevice(device_id));
        train_set[batch_id][device_id].construct(para, structures_train, n1, n2, device_id);
        print_line_2();
      }
      batch_id++;
    }
  }

  std::vector<Structure> structures_test;
  has_test_set = read_structures(false, para, structures_test);
  if (has_test_set) {
    test_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing test_set in device  %d.\n", device_id);
      CHECK(cudaSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id);
      print_line_2();
    }
  }

  int N = -1;
  int N_times_max_NN_radial = -1;
  int N_times_max_NN_angular = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  if (has_test_set) {
    N = test_set[0].N;
    N_times_max_NN_radial = test_set[0].N * test_set[0].max_NN_radial;
    N_times_max_NN_angular = test_set[0].N * test_set[0].max_NN_angular;
    max_NN_radial = test_set[0].max_NN_radial;
    max_NN_angular = test_set[0].max_NN_angular;
  }
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n][0].N > N) {
      N = train_set[n][0].N;
    };
    if (train_set[n][0].N * train_set[n][0].max_NN_radial > N_times_max_NN_radial) {
      N_times_max_NN_radial = train_set[n][0].N * train_set[n][0].max_NN_radial;
    };
    if (train_set[n][0].N * train_set[n][0].max_NN_angular > N_times_max_NN_angular) {
      N_times_max_NN_angular = train_set[n][0].N * train_set[n][0].max_NN_angular;
    };

    if (train_set[n][0].max_NN_radial > max_NN_radial) {
      max_NN_radial = train_set[n][0].max_NN_radial;
    }
    if (train_set[n][0].max_NN_angular > max_NN_angular) {
      max_NN_angular = train_set[n][0].max_NN_angular;
    }
  }

  potential.reset(
    new NEP3(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));

  if (para.prediction == 0) {
    fid_loss_out = my_fopen("loss.out", "a");
  }
}

Fitness::~Fitness()
{
  if (fid_loss_out != NULL) {
    fclose(fid_loss_out);
  }
}

void Fitness::compute(
  const int generation, Parameters& para, const float* population, float* fitness)
{
  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));
  int population_iter = (para.population_size - 1) / deviceCount + 1;

  if (generation == 0) {
    std::vector<float> dummy_solution(para.number_of_variables * deviceCount, 1.0f);
    for (int n = 0; n < num_batches; ++n) {
      potential->find_force(para, dummy_solution.data(), train_set[n], true, deviceCount);
    }

  } else {
    int batch_id = generation % num_batches;
    for (int n = 0; n < population_iter; ++n) {
      const float* individual = population + deviceCount * n * para.number_of_variables;
      potential->find_force(para, individual, train_set[batch_id], false, deviceCount);
      for (int m = 0; m < deviceCount; ++m) {
        float energy_shift_per_structure_not_used;
        auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
          para, energy_shift_per_structure_not_used, true, true, m);
        auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
        auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);

        for (int t = 0; t <= para.num_types; ++t) {
          fitness[deviceCount * n + m + (6 * t + 3) * para.population_size] =
            para.lambda_e * rmse_energy_array[t];
          fitness[deviceCount * n + m + (6 * t + 4) * para.population_size] =
            para.lambda_f * rmse_force_array[t];
          fitness[deviceCount * n + m + (6 * t + 5) * para.population_size] =
            para.lambda_v * rmse_virial_array[t];
        }
      }
    }
  }
}

void Fitness::predict_energy_or_stress(FILE* fid, float* data, float* ref, Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    float data_nc = 0.0f;
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      data_nc += data[offset + m];
    }
    fprintf(fid, "%g %g\n", data_nc / dataset.Na_cpu[nc], ref[nc]);
  }
}

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, float* elite)
{
  if (para.version == 2) {
    if (para.enable_zbl) {
      fprintf(fid_nep, "nep_zbl %d ", para.num_types);
    } else {
      fprintf(fid_nep, "nep %d ", para.num_types);
    }
  } else if (para.version == 3) {
    if (para.enable_zbl) {
      fprintf(fid_nep, "nep3_zbl %d ", para.num_types);
    } else {
      fprintf(fid_nep, "nep3 %d ", para.num_types);
    }
  } else if (para.version == 4) {
    if (para.enable_zbl) {
      fprintf(fid_nep, "nep4_zbl %d ", para.num_types);
    } else {
      fprintf(fid_nep, "nep4 %d ", para.num_types);
    }
  }

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_nep, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_nep, "\n");
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_nep, "zbl 0 0\n");
    } else {
      fprintf(fid_nep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }
  fprintf(
    fid_nep, "cutoff %g %g %d %d\n", para.rc_radial, para.rc_angular, max_NN_radial,
    max_NN_angular);
  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  if (para.version >= 3) {
    fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
    fprintf(fid_nep, "l_max %d %d %d\n", para.L_max, para.L_max_4body, para.L_max_5body);
  } else {
    fprintf(fid_nep, "l_max %d\n", para.L_max);
  }

  fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", elite[m]);
  }
  CHECK(cudaSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_nep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 8 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_nep, "%15.7e\n", para.zbl_para[d]);
    }
  }
}

void Fitness::report_error(
  Parameters& para,
  const int generation,
  const float loss_total,
  const float loss_L1,
  const float loss_L2,
  float* elite)
{
  if (0 == (generation + 1) % 100) {
    int batch_id = generation % num_batches;
    potential->find_force(para, elite, train_set[batch_id], false, 1);
    float energy_shift_per_structure;
    auto rmse_energy_train_array =
      train_set[batch_id][0].get_rmse_energy(para, energy_shift_per_structure, false, true, 0);
    auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(para, false, 0);
    auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(para, false, 0);

    float rmse_energy_train = rmse_energy_train_array.back();
    float rmse_force_train = rmse_force_train_array.back();
    float rmse_virial_train = rmse_virial_train_array.back();

    // correct the last bias parameter in the NN
    if (para.train_mode == 0) {
      elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
    }

    float rmse_energy_test = 0.0f;
    float rmse_force_test = 0.0f;
    float rmse_virial_test = 0.0f;
    if (has_test_set) {
      potential->find_force(para, elite, test_set, false, 1);
      float energy_shift_per_structure_not_used;
      auto rmse_energy_test_array =
        test_set[0].get_rmse_energy(para, energy_shift_per_structure_not_used, false, false, 0);
      auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
      auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
      rmse_energy_test = rmse_energy_test_array.back();
      rmse_force_test = rmse_force_test_array.back();
      rmse_virial_test = rmse_virial_test_array.back();
    }

    FILE* fid_nep = my_fopen("nep.txt", "w");
    write_nep_txt(fid_nep, para, elite);
    fclose(fid_nep);

    if (0 == (generation + 1) % 100000) {
      time_t rawtime;
      time(&rawtime);
      struct tm* timeinfo = localtime(&rawtime);
      char buffer[200];
      strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
      std::string filename(buffer + std::to_string(generation + 1) + ".txt");

      FILE* fid_nep = my_fopen(filename.c_str(), "w");
      write_nep_txt(fid_nep, para, elite);
      fclose(fid_nep);
    }

    if (para.train_mode == 0) {
      printf(
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n", generation + 1,
        loss_total, loss_L1, loss_L2, rmse_energy_train, rmse_force_train, rmse_virial_train,
        rmse_energy_test, rmse_force_test, rmse_virial_test);
      fprintf(
        fid_loss_out, "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
        generation + 1, loss_total, loss_L1, loss_L2, rmse_energy_train, rmse_force_train,
        rmse_virial_train, rmse_energy_test, rmse_force_test, rmse_virial_test);
    } else {
      printf(
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f\n", generation + 1, loss_total, loss_L1, loss_L2,
        rmse_virial_train, rmse_virial_test);
      fprintf(
        fid_loss_out, "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f\n", generation + 1, loss_total,
        loss_L1, loss_L2, rmse_virial_train, rmse_virial_test);
    }
    fflush(stdout);
    fflush(fid_loss_out);

    if (has_test_set) {
      if (para.train_mode == 0) {
        FILE* fid_force = my_fopen("force_test.out", "w");
        FILE* fid_energy = my_fopen("energy_test.out", "w");
        FILE* fid_virial = my_fopen("virial_test.out", "w");
        update_energy_force_virial(fid_energy, fid_force, fid_virial, test_set[0]);
        fclose(fid_energy);
        fclose(fid_force);
        fclose(fid_virial);
      } else if (para.train_mode == 1) {
        FILE* fid_dipole = my_fopen("dipole_test.out", "w");
        update_dipole(fid_dipole, test_set[0]);
        fclose(fid_dipole);
      } else if (para.train_mode == 2) {
        FILE* fid_polarizability = my_fopen("polarizability_test.out", "w");
        update_polarizability(fid_polarizability, test_set[0]);
        fclose(fid_polarizability);
      }
    }

    if (0 == (generation + 1) % 1000) {
      predict(para, elite);
    }
  }
}

void Fitness::update_energy_force_virial(
  FILE* fid_energy, FILE* fid_force, FILE* fid_virial, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());

  // update force.out
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
      int n = offset + m;
      fprintf(
        fid_force, "%g %g %g %g %g %g\n", dataset.force_cpu[n], dataset.force_cpu[n + dataset.N],
        dataset.force_cpu[n + dataset.N * 2], dataset.force_ref_cpu[n],
        dataset.force_ref_cpu[n + dataset.N], dataset.force_ref_cpu[n + dataset.N * 2]);
    }
  }

  // update energy.out
  predict_energy_or_stress(
    fid_energy, dataset.energy_cpu.data(), dataset.energy_ref_cpu.data(), dataset);

  // update virial.out
  for (int k = 0; k < 6; ++k) {
    predict_energy_or_stress(
      fid_virial, dataset.virial_cpu.data() + dataset.N * k,
      dataset.virial_ref_cpu.data() + dataset.Nc * k, dataset);
  }
}

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  for (int k = 0; k < 3; ++k) {
    predict_energy_or_stress(
      fid_dipole, dataset.virial_cpu.data() + dataset.N * k,
      dataset.virial_ref_cpu.data() + dataset.Nc * k, dataset);
  }
}

void Fitness::update_polarizability(FILE* fid_polarizability, Dataset& dataset)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  for (int k = 0; k < 6; ++k) {
    predict_energy_or_stress(
      fid_polarizability, dataset.virial_cpu.data() + dataset.N * k,
      dataset.virial_ref_cpu.data() + dataset.Nc * k, dataset);
  }
}

void Fitness::predict(Parameters& para, float* elite)
{
  if (para.train_mode == 0) {
    FILE* fid_force = my_fopen("force_train.out", "w");
    FILE* fid_energy = my_fopen("energy_train.out", "w");
    FILE* fid_virial = my_fopen("virial_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_energy_force_virial(fid_energy, fid_force, fid_virial, train_set[batch_id][0]);
    }
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
  } else if (para.train_mode == 1) {
    FILE* fid_dipole = my_fopen("dipole_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_dipole(fid_dipole, train_set[batch_id][0]);
    }
    fclose(fid_dipole);
  } else if (para.train_mode == 2) {
    FILE* fid_polarizability = my_fopen("polarizability_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_polarizability(fid_polarizability, train_set[batch_id][0]);
    }
    fclose(fid_polarizability);
  }
}