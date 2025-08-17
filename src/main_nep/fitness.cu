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

/*----------------------------------------------------------------------------80
Get the fitness
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "tnep.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <cstring>

Fitness::Fitness(Parameters& para)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));

  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  printf("Number of devices = %d\n", deviceCount);
  printf("Number of batches = %d\n", num_batches);
  int batch_size_old = para.batch_size;
  para.batch_size = (structures_train.size() - 1) / num_batches + 1;
  if (batch_size_old != para.batch_size) {
    printf("Hello, I changed the batch_size from %d to %d.\n", batch_size_old, para.batch_size);
  }

  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    train_set[batch_id].resize(deviceCount);
  }
  int count = 0;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    const int batch_size_minimal = structures_train.size() / num_batches;
    const bool is_larger_batch =
      batch_id + batch_size_minimal * num_batches < structures_train.size();
    const int batch_size = is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
    count += batch_size;
    printf("\nBatch %d:\n", batch_id);
    printf("Number of configurations = %d.\n", batch_size);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing train_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      train_set[batch_id][device_id].construct(
        para, structures_train, count - batch_size, count, device_id);
      print_line_2();
    }
  }

  std::vector<Structure> structures_test;
  has_test_set = read_structures(false, para, structures_test);
  if (has_test_set) {
    test_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing test_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id);
      print_line_2();
    }
  }

  int N = -1;
  int Nc = -1;
  int N_times_max_NN_radial = -1;
  int N_times_max_NN_angular = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  if (has_test_set) {
    N = test_set[0].N;
    Nc = test_set[0].Nc;
    N_times_max_NN_radial = test_set[0].N * test_set[0].max_NN_radial;
    N_times_max_NN_angular = test_set[0].N * test_set[0].max_NN_angular;
    max_NN_radial = test_set[0].max_NN_radial;
    max_NN_angular = test_set[0].max_NN_angular;
  }
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n][0].N > N) {
      N = train_set[n][0].N;
    };
    if (train_set[n][0].Nc > Nc) {
      Nc = train_set[n][0].Nc;
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

  if (para.train_mode == 1 || para.train_mode == 2) {
    potential.reset(
      new TNEP(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
  } else {
    if (para.charge_mode) {
      potential.reset(
        new NEP_Charge(para, N, Nc, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
    } else {
      potential.reset(
        new NEP(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
    }
  }

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
  CHECK(gpuGetDeviceCount(&deviceCount));
  int population_iter = (para.population_size - 1) / deviceCount + 1;

  if (generation == 0) {
    std::vector<float> dummy_solution(para.number_of_variables * deviceCount, para.initial_para);
    for (int n = 0; n < num_batches; ++n) {
      potential->find_force(
        para,
        dummy_solution.data(),
        train_set[n],
        (para.fine_tune ? false : true),
        true,
        deviceCount);
    }
  } else {
    int batch_id = generation % num_batches;
    bool calculate_neighbor = (num_batches > 1) || (generation % 100 == 0);
    for (int n = 0; n < population_iter; ++n) {
      const float* individual = population + deviceCount * n * para.number_of_variables;
      potential->find_force(
        para, individual, train_set[batch_id], false, calculate_neighbor, deviceCount);
      for (int m = 0; m < deviceCount; ++m) {
        float energy_shift_per_structure_not_used;
        auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
          para, energy_shift_per_structure_not_used, true, true, m);
        auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
        auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
        auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);

        for (int t = 0; t <= para.num_types; ++t) {
          fitness[deviceCount * n + m + (7 * t + 3) * para.population_size] =
            para.lambda_e * rmse_energy_array[t];
          fitness[deviceCount * n + m + (7 * t + 4) * para.population_size] =
            para.lambda_f * rmse_force_array[t];
          fitness[deviceCount * n + m + (7 * t + 5) * para.population_size] =
            para.lambda_v * rmse_virial_array[t];
          if (para.charge_mode) {
            fitness[deviceCount * n + m + (7 * t + 6) * para.population_size] =
              para.lambda_q * rmse_charge_array[t];
          } else {
            fitness[deviceCount * n + m + (7 * t + 6) * para.population_size] = 0.0f;
          }
        }
      }
    }

    if (para.use_full_batch) {
      int count_batch = 0;
      for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
        if (batch_id == generation % num_batches) {
          continue; // skip the batch that has already been calculated
        }
        ++count_batch;
        for (int n = 0; n < population_iter; ++n) {
          const float* individual = population + deviceCount * n * para.number_of_variables;
          potential->find_force(
            para, individual, train_set[batch_id], false, calculate_neighbor, deviceCount);
          for (int m = 0; m < deviceCount; ++m) {
            float energy_shift_per_structure_not_used;
            auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
              para, energy_shift_per_structure_not_used, true, true, m);
            auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
            auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
            auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);
            for (int t = 0; t <= para.num_types; ++t) {
              // energy
              float old_value = fitness[deviceCount * n + m + (7 * t + 3) * para.population_size];
              float new_value = para.lambda_e * rmse_energy_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness[deviceCount * n + m + (7 * t + 3) * para.population_size] = new_value;
              // force
              old_value = fitness[deviceCount * n + m + (7 * t + 4) * para.population_size];
              new_value = para.lambda_f * rmse_force_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness[deviceCount * n + m + (7 * t + 4) * para.population_size] = new_value;
              // virial
              old_value = fitness[deviceCount * n + m + (7 * t + 5) * para.population_size];
              new_value = para.lambda_v * rmse_virial_array[t];
              new_value = old_value * old_value * count_batch + new_value * new_value;
              new_value = sqrt(new_value / (count_batch + 1));
              fitness[deviceCount * n + m + (7 * t + 5) * para.population_size] = new_value;
              // charge
              if (para.charge_mode) {
                old_value = fitness[deviceCount * n + m + (7 * t + 6) * para.population_size];
                new_value = para.lambda_q * rmse_charge_array[t];
                new_value = old_value * old_value * count_batch + new_value * new_value;
                new_value = sqrt(new_value / (count_batch + 1));
                fitness[deviceCount * n + m + (7 * t + 6) * para.population_size] = new_value;
              }
            }
          }
        }
      }
    }
  }
}

void Fitness::output(
  bool is_stress,
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int n = 0; n < num_components; ++n) {
      int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
      float data_nc = 0.0f;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        data_nc += prediction[offset + m];
      }
      if (!is_stress) {
        fprintf(fid, "%g ", data_nc / dataset.Na_cpu[nc]);
      } else {
        fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
      }
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.Nc + nc];
      if (is_stress) {
        ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

void Fitness::output_atomic(
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
for (int nc = 0; nc < dataset.Nc; ++nc) {
  int offset = dataset.Na_sum_cpu[nc];
  for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
    for (int n = 0; n < num_components; ++n) {
      int index = n * dataset.N + offset + m;
      fprintf(fid, "%g ", prediction[index]);
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.N + offset + m];
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}
}

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, float* elite)
{
  if (para.train_mode == 0) { // potential model
    if (!para.charge_mode) {
      if (para.version == 3) {
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
    } else {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep4_zbl_charge%d %d ", para.charge_mode, para.num_types);
      } else {
        fprintf(fid_nep, "nep4_charge%d %d ", para.charge_mode, para.num_types);
      }
    }
  } else if (para.train_mode == 1) { // dipole model
    if (para.version == 3) {
      fprintf(fid_nep, "nep3_dipole %d ", para.num_types);
    } else if (para.version == 4) {
      fprintf(fid_nep, "nep4_dipole %d ", para.num_types);
    }
  } else if (para.train_mode == 2) { // polarizability model
    if (para.version == 3) {
      fprintf(fid_nep, "nep3_polarizability %d ", para.num_types);
    } else if (para.version == 4) {
      fprintf(fid_nep, "nep4_polarizability %d ", para.num_types);
    }
  } else if (para.train_mode == 3) { // temperature model
    if (para.version == 3) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep3_zbl_temperature %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep3_temperature %d ", para.num_types);
      }
    } else if (para.version == 4) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep4_zbl_temperature %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep4_temperature %d ", para.num_types);
      }
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
  if (para.use_typewise_cutoff || para.use_typewise_cutoff_zbl) {
    fprintf(
      fid_nep,
      "cutoff %g %g %d %d %g %g %g\n",
      para.rc_radial,
      para.rc_angular,
      max_NN_radial,
      max_NN_angular,
      para.typewise_cutoff_radial_factor,
      para.typewise_cutoff_angular_factor,
      para.typewise_cutoff_zbl_factor);
  } else {
    fprintf(
      fid_nep,
      "cutoff %g %g %d %d\n",
      para.rc_radial,
      para.rc_angular,
      max_NN_radial,
      max_NN_angular);
  }
  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_nep, "l_max %d %d %d\n", para.L_max, para.L_max_4body, para.L_max_5body);

  fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", elite[m]);
  }
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_nep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
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
    potential->find_force(para, elite, train_set[batch_id], false, true, 1);
    float energy_shift_per_structure;
    auto rmse_energy_train_array =
      train_set[batch_id][0].get_rmse_energy(para, energy_shift_per_structure, false, true, 0);
    auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(para, false, 0);
    auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(para, false, 0);

    float rmse_energy_train = rmse_energy_train_array.back();
    float rmse_force_train = rmse_force_train_array.back();
    float rmse_virial_train = rmse_virial_train_array.back();

    // correct the last bias parameter in the NN
    if (para.train_mode == 0 || para.train_mode == 3) {
      elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
    }

    float rmse_energy_test = 0.0f;
    float rmse_force_test = 0.0f;
    float rmse_virial_test = 0.0f;
    if (has_test_set) {
      potential->find_force(para, elite, test_set, false, true, 1);
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

    if (0 == (generation + 1) % para.save_potential) {
      std::string filename;
      if (para.save_potential_format == 1) {
        time_t rawtime;
        time(&rawtime);
        struct tm* timeinfo = localtime(&rawtime);
        char buffer[200];
        strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
        filename = std::string(buffer) + std::to_string(generation + 1) + ".txt";
      } else {
        filename = "nep_gen" + std::to_string(generation + 1) + ".txt";
      }

      FILE* fid_nep = my_fopen(filename.c_str(), "w");
      write_nep_txt(fid_nep, para, elite);
      fclose(fid_nep);
    }

    if (para.train_mode == 0 || para.train_mode == 3) {
      printf(
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_energy_train,
        rmse_force_train,
        rmse_virial_train,
        rmse_energy_test,
        rmse_force_test,
        rmse_virial_test);
      fprintf(
        fid_loss_out,
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_energy_train,
        rmse_force_train,
        rmse_virial_train,
        rmse_energy_test,
        rmse_force_test,
        rmse_virial_test);
    } else {
      printf(
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
      fprintf(
        fid_loss_out,
        "%-8d%-11.5f%-11.5f%-11.5f%-13.5f%-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
    }
    fflush(stdout);
    fflush(fid_loss_out);

    if (has_test_set) {
      if (para.train_mode == 0 || para.train_mode == 3) {
        FILE* fid_force = my_fopen("force_test.out", "w");
        FILE* fid_energy = my_fopen("energy_test.out", "w");
        FILE* fid_virial = my_fopen("virial_test.out", "w");
        FILE* fid_stress = my_fopen("stress_test.out", "w");
        update_energy_force_virial(fid_energy, fid_force, fid_virial, fid_stress, test_set[0]);
        fclose(fid_energy);
        fclose(fid_force);
        fclose(fid_virial);
        fclose(fid_stress);
        if (para.charge_mode) {
          FILE* fid_charge = my_fopen("charge_test.out", "w");
          update_charge(fid_charge, test_set[0]);
          fclose(fid_charge);
          FILE* fid_bec = my_fopen("bec_test.out", "w");
          update_bec(fid_bec, test_set[0]);
          fclose(fid_bec);
        }
      } else if (para.train_mode == 1) {
        FILE* fid_dipole = my_fopen("dipole_test.out", "w");
        update_dipole(fid_dipole, test_set[0], para.atomic_v);
        fclose(fid_dipole);
      } else if (para.train_mode == 2) {
        FILE* fid_polarizability = my_fopen("polarizability_test.out", "w");
        update_polarizability(fid_polarizability, test_set[0], para.atomic_v);
        fclose(fid_polarizability);
      }
    }

    if (0 == (generation + 1) % 1000) {
      predict(para, elite);
    }
  }
}

void Fitness::update_energy_force_virial(
  FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());

  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
      int n = offset + m;
      fprintf(
        fid_force,
        "%g %g %g %g %g %g\n",
        dataset.force_cpu[n],
        dataset.force_cpu[n + dataset.N],
        dataset.force_cpu[n + dataset.N * 2],
        dataset.force_ref_cpu[n],
        dataset.force_ref_cpu[n + dataset.N],
        dataset.force_ref_cpu[n + dataset.N * 2]);
    }
  }

  output(false, 1, fid_energy, dataset.energy_cpu.data(), dataset.energy_ref_cpu.data(), dataset);

  output(false, 6, fid_virial, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  output(true, 6, fid_stress, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
}

void Fitness::update_charge(FILE* fid_charge, Dataset& dataset)
{
  dataset.charge.copy_to_host(dataset.charge_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      fprintf(fid_charge, "%g\n", dataset.charge_cpu[dataset.Na_sum_cpu[nc] + m]);
    }
  }
}

void Fitness::update_bec(FILE* fid_bec, Dataset& dataset)
{
  dataset.bec.copy_to_host(dataset.bec_cpu.data());
  output_atomic(9, fid_bec, dataset.bec_cpu.data(), dataset.bec_ref_cpu.data(), dataset);
}

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(3, fid_dipole, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::update_polarizability(FILE* fid_polarizability, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 6, fid_polarizability, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(6, fid_polarizability, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::predict(Parameters& para, float* elite)
{
  if (para.train_mode == 0 || para.train_mode == 3) {
    FILE* fid_force = my_fopen("force_train.out", "w");
    FILE* fid_energy = my_fopen("energy_train.out", "w");
    FILE* fid_virial = my_fopen("virial_train.out", "w");
    FILE* fid_stress = my_fopen("stress_train.out", "w");
    FILE* fid_charge;
    FILE* fid_bec;
    if (para.charge_mode) {
      fid_charge = my_fopen("charge_train.out", "w");
      fid_bec = my_fopen("bec_train.out", "w");
    }
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, true, 1);
      update_energy_force_virial(
        fid_energy, fid_force, fid_virial, fid_stress, train_set[batch_id][0]);
      if (para.charge_mode) {
        update_charge(fid_charge, train_set[batch_id][0]);
        update_bec(fid_bec, train_set[batch_id][0]);
      }
    }
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
    if (para.charge_mode) {
      fclose(fid_charge);
      fclose(fid_bec);
    }
  } else if (para.train_mode == 1) {
    FILE* fid_dipole = my_fopen("dipole_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, true, 1);
      update_dipole(fid_dipole, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_dipole);
  } else if (para.train_mode == 2) {
    FILE* fid_polarizability = my_fopen("polarizability_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, true, 1);
      update_polarizability(fid_polarizability, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_polarizability);
  }
}
