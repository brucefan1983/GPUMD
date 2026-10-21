/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Get the fitness
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "nep.cuh"
#include "nep_vdw.cuh"
#include "nep_charge.cuh"
#include "nep_spin.cuh"
#include "fitness_spin.cuh"
#include "snes_spin.cuh"
#include "nep_charge_vdw.cuh"
#include "tnep.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include "utilities/nep_parameters.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <cstring>

// Number of structures in one mini-batch. The first n_total % num_batches batches take one
// structure more than the rest, so the batches differ in size by at most one.
static int get_batch_size(const int batch_id, const int n_total, const int num_batches)
{
  const int batch_size_minimal = n_total / num_batches;
  const bool is_larger_batch = batch_id + batch_size_minimal * num_batches < n_total;
  return is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
}

Fitness::Fitness(Parameters& para)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));

  const bool spin_restart = fitness_spin::prepare_checkpoint(para);
  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  fitness_spin::prepare_training_data(para, structures_train, spin_restart);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  printf("Number of devices = %d\n", deviceCount);
  printf("Number of batches = %d\n", num_batches);
  int batch_size_old = para.batch_size;
  para.batch_size = (structures_train.size() - 1) / num_batches + 1;
  if (batch_size_old != para.batch_size) {
    printf("Hello, I changed the batch_size from %d to %d.\n", batch_size_old, para.batch_size);
  }
  std::vector<int> batch_sizes(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    batch_sizes[batch_id] = get_batch_size(batch_id, structures_train.size(), num_batches);
  }

  // The training combinations are resolved before the batches are constructed, which allocate
  // the total energies under para.has_ediff_combinations.
  energy_difference.read_train(para, structures_train, batch_size_old, num_batches, batch_sizes);

  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    train_set[batch_id].resize(deviceCount);
  }
  int count = 0;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    const int batch_size = batch_sizes[batch_id];
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
  energy_difference.read_test(para, structures_test, has_test_set);
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

  if (para.lambda_spin_response > 0.0f) {
    std::vector<Structure> structures_response;
    read_response_structures(para, structures_response);
    fitness_spin::prepare_response_data(para, structures_response, structures_train, structures_test);
    response_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      CHECK(gpuSetDevice(device_id));
      response_set[device_id].construct(
        para, structures_response, 0, structures_response.size(), device_id);
    }
    printf("Response fitness uses all %zu frames every generation, independently of train batches.\n",
      structures_response.size());
  }

  int N = -1;
  int Nc = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  max_NN_spin = 0;
  if (has_test_set) {
    N = test_set[0].N;
    Nc = test_set[0].Nc;
    max_NN_radial = test_set[0].max_NN_radial;
    max_NN_angular = test_set[0].max_NN_angular;
    max_NN_spin = test_set[0].max_NN_spin;
  }
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n][0].N > N) {
      N = train_set[n][0].N;
    };
    if (train_set[n][0].Nc > Nc) {
      Nc = train_set[n][0].Nc;
    };

    if (train_set[n][0].max_NN_radial > max_NN_radial) {
      max_NN_radial = train_set[n][0].max_NN_radial;
    }
    if (train_set[n][0].max_NN_angular > max_NN_angular) {
      max_NN_angular = train_set[n][0].max_NN_angular;
    }
    if (train_set[n][0].max_NN_spin > max_NN_spin) {
      max_NN_spin = train_set[n][0].max_NN_spin;
    }
  }

  if (!response_set.empty()) {
    const Dataset& response = response_set[0];
    N = std::max(N, response.N);
    Nc = std::max(Nc, response.Nc);
    max_NN_radial = std::max(max_NN_radial, response.max_NN_radial);
    max_NN_angular = std::max(max_NN_angular, response.max_NN_angular);
    max_NN_spin = std::max(max_NN_spin, response.max_NN_spin);
  }

  if (para.model_type == 1 || para.model_type == 2) {
    potential.reset(new TNEP(para, N, para.version, deviceCount));
  } else {
    if (para.charge_vdw) {
      potential.reset(new NEP_Charge_VDW(para, N, Nc, para.version, deviceCount));
    } else if (para.charge_mode) {
      potential.reset(new NEP_Charge(para, N, Nc, para.version, deviceCount));
    } else if (para.spin_mode) {
      potential.reset(new NEP_Spin_Trainer(para, N, para.version, deviceCount));
    } else if (para.vdw) {
      potential.reset(new NEP_VDW(para, N, Nc, para.version, deviceCount));
    } else {
      potential.reset(new NEP(para, N, para.version, deviceCount));
    }
  }

  if (para.prediction == 0) {
    fid_loss_out = my_fopen("loss.out", "a");
    fprintf(fid_loss_out, "# format_version 1\n");
    fprintf(fid_loss_out, "# output_interval %d\n", para.output_interval);
    fprintf(fid_loss_out, "# columns generation total L1 L2");
    if (para.model_type == 0 || para.model_type == 3) {
      if (para.spin_mode) {
        fprintf(fid_loss_out, " rmse_energy_train rmse_force_train rmse_virial_train"
          " rmse_mforce_train rmse_tau_train rmse_energy_test rmse_force_test"
          " rmse_virial_test rmse_mforce_test rmse_tau_test");
      } else if (para.charge_mode || para.charge_vdw) {
        fprintf(
          fid_loss_out,
          " rmse_energy_train rmse_force_train rmse_virial_train rmse_charge_train rmse_bec_train"
          " rmse_energy_test rmse_force_test rmse_virial_test rmse_charge_test rmse_bec_test");
      } else {
        fprintf(
          fid_loss_out,
          " rmse_energy_train rmse_force_train rmse_virial_train"
          " rmse_energy_test rmse_force_test rmse_virial_test");
      }
      if (para.has_ediff_combinations) {
        fprintf(fid_loss_out, " rmse_ediff_train rmse_ediff_test");
      }
      fprintf(fid_loss_out, "\n");
    } else if (para.model_type == 1) {
      fprintf(fid_loss_out, " rmse_dipole_train rmse_dipole_test\n");
    } else {
      fprintf(fid_loss_out, " rmse_polarizability_train rmse_polarizability_test\n");
    }
    fflush(fid_loss_out);
  }
}

Fitness::~Fitness()
{
  if (fid_loss_out != NULL) {
    fclose(fid_loss_out);
  }
}

void Fitness::initialize_q_scaler(Parameters& para)
{
  if (!para.spin_mode || para.prediction != 0 || para.fine_tune || para.import_q_scaler || para.spin_restart) {
    return;
  }

  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  std::vector<float> scaler_solution(
    para.number_of_variables * deviceCount, para.initial_para);
  if (para.spin_mode == 3) {
    // A fixed, non-degenerate reference, independent of the SNES search mean/RNG.
    // Constant projection matrices would collapse Spin3's distinct channels.
    std::mt19937 reference_rng(12345678);
    std::vector<float> reference(para.number_of_variables), sigma(reference.size());
    snes_spin::initialize_search(para, reference_rng, reference, sigma);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      std::copy(reference.begin(), reference.end(),
        scaler_solution.begin() + device_id * para.number_of_variables);
    }
  }

  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    potential->find_force(
      para,
      scaler_solution.data(),
      train_set[batch_id],
      true,
      deviceCount);
  }

  fitness_spin::finalize_q_scaler(para, deviceCount);
}

void Fitness::compute(
  const int generation,
  Parameters& para,
  const float* population,
  float* fitness_energy,
  float* fitness_force,
  float* fitness_virial,
  float* fitness_charge,
  float* fitness_bec,
  float* fitness_mforce,
  float* fitness_tau,
  float* fitness_spin_response,
  float* fitness_ediff)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  int population_iter = (para.population_size - 1) / deviceCount + 1;

  // Retain upstream's generation-zero scaler pass for ordinary models.
  // Spin3 initializes its checkpoint-aware scaler once before optimization.
  if (!para.spin_mode && generation == 0) {
    std::vector<float> dummy_solution(para.number_of_variables * deviceCount, para.initial_para);
    for (int batch = 0; batch < num_batches; ++batch) {
      potential->find_force(para, dummy_solution.data(), train_set[batch],
        !(para.fine_tune || para.import_q_scaler), deviceCount);
    }
    return;
  }

  {
    std::vector<fitness_spin::ResponseLoss> response_points;
    if (para.lambda_spin_response > 0.0f) {
      response_points.resize(para.population_size);
    }
    int batch_id = generation % num_batches;
    for (int n = 0; n < population_iter; ++n) {
      const float* individual = population + deviceCount * n * para.number_of_variables;
      potential->find_force(para, individual, train_set[batch_id], false, deviceCount);
      for (int m = 0; m < deviceCount; ++m) {
        const int population_index = deviceCount * n + m;
        if (population_index >= para.population_size) {
          continue;
        }
        float energy_shift_per_structure_not_used;
        auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
          para, energy_shift_per_structure_not_used, true, true, m);
        auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
        auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
        auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);
        auto rmse_bec_array = train_set[batch_id][m].get_rmse_bec(para, m);
        auto rmse_mforce_array =
          para.spin_mode
            ? train_set[batch_id][m].get_rmse_mforce(para, true, m)
            : std::vector<float>(para.num_types + 1, 0.0f);
        auto rmse_tau_array =
          para.spin_mode
            ? train_set[batch_id][m].get_rmse_tau(para, true, m)
            : std::vector<float>(para.num_types + 1, 0.0f);

        for (int t = 0; t <= para.num_types; ++t) {
          fitness_energy[deviceCount * n + m + t * para.population_size] =
            para.lambda_e * rmse_energy_array[t];
          fitness_force[deviceCount * n + m + t * para.population_size] =
            para.lambda_f * rmse_force_array[t];
          fitness_virial[deviceCount * n + m + t * para.population_size] =
            para.lambda_v * rmse_virial_array[t];
          fitness_charge[deviceCount * n + m + t * para.population_size] =
            para.lambda_q * rmse_charge_array[t];
          fitness_bec[deviceCount * n + m + t * para.population_size] =
            para.lambda_z * rmse_bec_array[t];
          fitness_mforce[deviceCount * n + m + t * para.population_size] =
            para.lambda_m * rmse_mforce_array[t];
          fitness_tau[deviceCount * n + m + t * para.population_size] =
            para.lambda_tau * rmse_tau_array[t];
        }

        const float rmse_ediff =
          energy_difference.get_rmse_train(train_set[batch_id][m], batch_id, m);
        if (fitness_ediff) fitness_ediff[deviceCount * n + m] = para.lambda_d * rmse_ediff;
      }
    }
    if (!response_set.empty()) {
      for (int n = 0; n < population_iter; ++n) {
        const float* individual = population + deviceCount * n * para.number_of_variables;
        // Reuse train-only scalers and baseline; never add response E/F/MF to ordinary fitness.
        potential->find_force(para, individual, response_set, false, deviceCount);
        for (int m = 0; m < deviceCount && deviceCount * n + m < para.population_size; ++m) {
          response_points[deviceCount * n + m].append(m, response_set[m]);
        }
      }
    }
    for (int p = 0; p < para.population_size; ++p) {
      const float value = para.lambda_spin_response > 0.0f
        ? para.lambda_spin_response * response_points[p].value()
        : 0.0f;
      for (int t = 0; t <= para.num_types; ++t) {
        fitness_spin_response[p + t * para.population_size] = value;
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
  Dataset& dataset,
  const int nc)
{
  for (int n = 0; n < num_components; ++n) {
    int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
    double data_nc = 0.0;
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      data_nc += prediction[offset + m];
    }
    if (!is_stress) {
      fprintf(fid, "%.16g ", data_nc / dataset.Na_cpu[nc]);
    } else {
      fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
    }
  }
  for (int n = 0; n < num_components; ++n) {
    float ref_value = reference[n * dataset.Nc + nc];
    if (is_stress) {
      if (ref_value > -1e5) {
        ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
    }
    if (n == num_components - 1) {
      fprintf(fid, "%.16g\n", static_cast<double>(ref_value));
    } else {
      fprintf(fid, "%g ", ref_value);
    }
  }
}

void Fitness::output_atomic(
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset,
  const int nc)
{
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

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, float* elite)
{
  if (para.model_type == 0) { // potential model
    if (!(para.charge_mode || para.charge_vdw)) {
      if (para.version == 4) {
        if (para.spin_mode) {
          if (para.enable_zbl) {
            fprintf(fid_nep, "nep4_spin3_zbl %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4_spin3 %d ", para.num_types);
          }
        } else if (para.enable_zbl) {
          if (para.vdw) {
            fprintf(fid_nep, "nep4_zbl_vdw %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4_zbl %d ", para.num_types);
          }
        } else {
          if (para.vdw) {
            fprintf(fid_nep, "nep4_vdw %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4 %d ", para.num_types);
          }
        }
      } 
    } else {
      if (para.charge_vdw) {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl_charge_vdw %d ", para.num_types);
        } else {
          fprintf(fid_nep, "nep4_charge_vdw %d ", para.num_types);
        }
      } else {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl_charge%d %d ", para.charge_mode, para.num_types);
        } else {
          fprintf(fid_nep, "nep4_charge%d %d ", para.charge_mode, para.num_types);
        }
      }
    }
  } else if (para.model_type == 1) { // dipole model
    if (para.version == 4) {
      fprintf(fid_nep, "nep4_dipole %d ", para.num_types);
    }
  } else if (para.model_type == 2) { // polarizability model
    if (para.version == 4) {
      fprintf(fid_nep, "nep4_polarizability %d ", para.num_types);
    }
  } else if (para.model_type == 3) { // temperature model
    if (para.version == 4) {
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
  if (para.spin_mode) {
    fitness_spin::write_checkpoint_metadata(fid_nep, para);
  }
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_nep, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(fid_nep, "zbl %g %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(fid_nep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }

  fprintf(fid_nep, "cutoff %g %g ", para.rc_radial[0], para.rc_angular[0]);
  if (para.has_multiple_cutoffs) {
    for (int n = 1; n < para.num_types; ++n) {
      fprintf(fid_nep, "%g %g ", para.rc_radial[n], para.rc_angular[n]);
    }
  }
  fprintf(
    fid_nep,
    "%d %d\n",
    para.spin_mode ? std::max(max_NN_radial, max_NN_spin) : max_NN_radial,
    max_NN_angular);

  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_nep, "l_max %d %d %d ", para.L_max, (para.has_q_222 ? 2 : 0), para.has_q_1111);
  if (para.has_q_112 || para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_112);
  }
  if (para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_123);
  }
  if (para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_233);
  }
  if (para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_134);
  }
  fprintf(fid_nep, "\n");

  if (para.num_hidden_layers == 2) {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, para.num_neurons2);
  } else {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  }

  std::vector<float> parameters_file(elite, elite + para.number_of_variables);
  const int descriptor_offset = para.number_of_variables_ann * (para.model_type == 2 ? 2 : 1);
#ifdef USE_CJ
  const int num_channels = para.num_types;
#else
  const int num_channels = para.num_types * para.num_types;
#endif
  descriptor_parameters_to_basis_major(
    parameters_file.data(),
    descriptor_offset,
    num_channels,
    para.n_max_radial,
    para.n_max_angular,
    para.basis_size_radial,
    para.basis_size_angular);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", parameters_file[m]);
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

void Fitness::get_save_potential_label(Parameters& para, const int generation, std::string& label) {
    if (para.save_potential_format == 1) {
      time_t rawtime;
      time(&rawtime);
      struct tm* timeinfo = localtime(&rawtime);
      char buffer[200];
      strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
      label = std::string(buffer) + std::to_string(generation + 1);
    } else {
      label = "nep_gen" + std::to_string(generation + 1);
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
  if (0 == (generation + 1) % para.output_interval) {
    int batch_id = generation % num_batches;
    potential->find_force(para, elite, train_set[batch_id], false, 1);
    float energy_shift_per_structure;
    auto rmse_energy_train_array =
      train_set[batch_id][0].get_rmse_energy(para, energy_shift_per_structure, false, true, 0);
    auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(para, false, 0);
    auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(para, false, 0);
    auto rmse_charge_train_array = train_set[batch_id][0].get_rmse_charge(para, 0);
    auto rmse_bec_train_array = train_set[batch_id][0].get_rmse_bec(para, 0);
    auto rmse_mforce_train_array =
      para.spin_mode
        ? train_set[batch_id][0].get_rmse_mforce(para, false, 0)
        : std::vector<float>(para.num_types + 1, 0.0f);
    auto rmse_tau_train_array =
      para.spin_mode
        ? train_set[batch_id][0].get_rmse_tau(para, false, 0)
        : std::vector<float>(para.num_types + 1, 0.0f);

    float rmse_energy_train = rmse_energy_train_array.back();
    float rmse_force_train = rmse_force_train_array.back();
    float rmse_virial_train = rmse_virial_train_array.back();
    float rmse_charge_train = rmse_charge_train_array.back();
    float rmse_bec_train = rmse_bec_train_array.back();
    float rmse_mforce_train = rmse_mforce_train_array.back();
    float rmse_tau_train = rmse_tau_train_array.back();

    float rmse_ediff_train = 0.0f;
    if (para.has_ediff_combinations) {
      rmse_ediff_train = energy_difference.get_rmse_train(train_set[batch_id][0], batch_id, 0);
    }

    // correct the last bias parameter in the NN
    if (para.model_type == 0 || para.model_type == 3) {
      elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
    }

    float rmse_energy_test = 0.0f;
    float rmse_force_test = 0.0f;
    float rmse_virial_test = 0.0f;
    float rmse_charge_test = 0.0f;
    float rmse_bec_test = 0.0f;
    float rmse_mforce_test = 0.0f;
    float rmse_tau_test = 0.0f;
    float rmse_ediff_test = 0.0f;
    if (has_test_set) {
      potential->find_force(para, elite, test_set, false, 1);
      float energy_shift_per_structure_not_used;
      auto rmse_energy_test_array =
        test_set[0].get_rmse_energy(para, energy_shift_per_structure_not_used, false, false, 0);
      auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
      auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
      auto rmse_charge_test_array = test_set[0].get_rmse_charge(para, 0);
      auto rmse_bec_test_array = test_set[0].get_rmse_bec(para, 0);
      auto rmse_mforce_test_array =
        para.spin_mode
          ? test_set[0].get_rmse_mforce(para, false, 0)
          : std::vector<float>(para.num_types + 1, 0.0f);
      auto rmse_tau_test_array =
        para.spin_mode
          ? test_set[0].get_rmse_tau(para, false, 0)
          : std::vector<float>(para.num_types + 1, 0.0f);
      rmse_energy_test = rmse_energy_test_array.back();
      rmse_force_test = rmse_force_test_array.back();
      rmse_virial_test = rmse_virial_test_array.back();
      rmse_charge_test = rmse_charge_test_array.back();
      rmse_bec_test = rmse_bec_test_array.back();
      rmse_mforce_test = rmse_mforce_test_array.back();
      rmse_tau_test = rmse_tau_test_array.back();
      rmse_ediff_test = energy_difference.get_rmse_test(test_set[0], 0);
    }

    FILE* fid_nep = my_fopen("nep.txt", "w");
    write_nep_txt(fid_nep, para, elite);
    fclose(fid_nep);

    if (0 == (generation + 1) % para.save_potential) {
      std::string filename;
      get_save_potential_label(para, generation, filename);
      filename += ".txt";

      FILE* fid_nep = my_fopen(filename.c_str(), "w");
      write_nep_txt(fid_nep, para, elite);
      fclose(fid_nep);
    }

    auto write_row = [&](FILE* fid) {
      if (para.model_type == 0 || para.model_type == 3) {
        const char* ediff_format = " %-13.5f %-13.5f";
        if (para.spin_mode) {
          fprintf(fid, "%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f",
            generation + 1, loss_total, loss_L1, loss_L2,
            rmse_energy_train, rmse_force_train, rmse_virial_train, rmse_mforce_train,
            rmse_tau_train, rmse_energy_test, rmse_force_test, rmse_virial_test,
            rmse_mforce_test, rmse_tau_test);
        } else if (!(para.charge_mode || para.charge_vdw)) {
          // NEP models
          fprintf(
            fid,
            "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f",
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
          // qNEP models:
          fprintf(
            fid,
            "%-8d %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f "
            "%-9.5f %-9.5f",
            generation + 1,
            loss_total,
            loss_L1,
            loss_L2,
            rmse_energy_train,
            rmse_force_train,
            rmse_virial_train,
            rmse_charge_train,
            rmse_bec_train,
            rmse_energy_test,
            rmse_force_test,
            rmse_virial_test,
            rmse_charge_test,
            rmse_bec_test);
          ediff_format = " %-9.5f %-9.5f";
        }
        // The ediff columns follow all others, so that the other columns keep their positions.
        if (para.has_ediff_combinations) {
          fprintf(fid, ediff_format, rmse_ediff_train, rmse_ediff_test);
        }
      } else {
        // TNEP models:
        fprintf(
          fid,
          "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_virial_train,
          rmse_virial_test);
      }
      fprintf(fid, "\n");
    };
    write_row(stdout);
    write_row(fid_loss_out);
    fflush(stdout);
    fflush(fid_loss_out);

    if (has_test_set) {
      copy_predictions_to_host(para, test_set[0]);
      std::vector<std::pair<Dataset*, int>> structures;
      for (int nc = 0; nc < test_set[0].Nc; ++nc) {
        structures.emplace_back(&test_set[0], nc);
      }
      write_predictions(para, "test", structures);
    }
  }

  if (0 == (generation + 1) % 1000) {
    predict(para, elite);
  }
}

void Fitness::copy_predictions_to_host(Parameters& para, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());
  if (para.spin_mode) dataset.mforce.copy_to_host(dataset.mforce_cpu.data());
  if (para.charge_mode || para.charge_vdw) {
    dataset.charge.copy_to_host(dataset.charge_cpu.data());
    if (para.has_bec) {
      dataset.bec.copy_to_host(dataset.bec_cpu.data());
    }
  }
}

void Fitness::update_energy_force_virial(
  FILE* fid_energy,
  FILE* fid_force,
  FILE* fid_virial,
  FILE* fid_stress,
  Dataset& dataset,
  const int nc)
{
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

  output(
    false, 1, fid_energy, dataset.energy_cpu.data(), dataset.energy_ref_cpu.data(), dataset, nc);

  output(
    false, 6, fid_virial, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset, nc);
  output(
    true, 6, fid_stress, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset, nc);
}

void Fitness::update_charge(FILE* fid_charge, Dataset& dataset, const int nc)
{
  for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
    fprintf(fid_charge, "%g\n", dataset.charge_cpu[dataset.Na_sum_cpu[nc] + m]);
  }
}

void Fitness::update_bec(FILE* fid_bec, Dataset& dataset, const int nc)
{
  output_atomic(9, fid_bec, dataset.bec_cpu.data(), dataset.bec_ref_cpu.data(), dataset, nc);
}

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset, bool atomic, const int nc)
{
  if (!atomic) {
    output(
      false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset, nc);
  } else {
    output_atomic(
      3, fid_dipole, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset, nc);
  }
}

void Fitness::update_polarizability(
  FILE* fid_polarizability, Dataset& dataset, bool atomic, const int nc)
{
  if (!atomic) {
    output(
      false,
      6,
      fid_polarizability,
      dataset.virial_cpu.data(),
      dataset.virial_ref_cpu.data(),
      dataset,
      nc);
  } else {
    output_atomic(
      6,
      fid_polarizability,
      dataset.virial_cpu.data(),
      dataset.avirial_ref_cpu.data(),
      dataset,
      nc);
  }
}

void Fitness::write_predictions(
  Parameters& para,
  const std::string& label,
  const std::vector<std::pair<Dataset*, int>>& structures)
{
  auto open = [&label](const char* quantity) {
    return my_fopen((std::string(quantity) + "_" + label + ".out").c_str(), "w");
  };
  if (para.model_type == 0 || para.model_type == 3) {
    const bool has_charge = para.charge_mode || para.charge_vdw;
    FILE* fid_force = open("force");
    FILE* fid_energy = open("energy");
    FILE* fid_virial = open("virial");
    FILE* fid_stress = open("stress");
    FILE* fid_mforce = para.spin_mode ? open("mforce") : nullptr;
    FILE* fid_charge = has_charge ? open("charge") : nullptr;
    FILE* fid_bec = has_charge && para.has_bec ? open("bec") : nullptr;
    for (const auto& structure : structures) {
      Dataset& dataset = *structure.first;
      if (fid_mforce) {
        const int nc = structure.second;
        for (int n = dataset.Na_sum_cpu[nc]; n < dataset.Na_sum_cpu[nc] + dataset.Na_cpu[nc]; ++n) {
          fprintf(fid_mforce, "%g %g %g %g %g %g\n",
            dataset.mforce_cpu[n], dataset.mforce_cpu[n + dataset.N], dataset.mforce_cpu[n + 2 * dataset.N],
            dataset.mforce_ref_cpu[n], dataset.mforce_ref_cpu[n + dataset.N], dataset.mforce_ref_cpu[n + 2 * dataset.N]);
        }
      }
      update_energy_force_virial(
        fid_energy, fid_force, fid_virial, fid_stress, dataset, structure.second);
      if (fid_charge) {
        update_charge(fid_charge, dataset, structure.second);
      }
      if (fid_bec) {
        update_bec(fid_bec, dataset, structure.second);
      }
    }
    fclose(fid_energy);
    if (fid_mforce) fclose(fid_mforce);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
    if (fid_charge) {
      fclose(fid_charge);
    }
    if (fid_bec) {
      fclose(fid_bec);
    }
  } else if (para.model_type == 1) {
    FILE* fid_dipole = open("dipole");
    for (const auto& structure : structures) {
      update_dipole(fid_dipole, *structure.first, para.atomic_v, structure.second);
    }
    fclose(fid_dipole);
  } else if (para.model_type == 2) {
    FILE* fid_polarizability = open("polarizability");
    for (const auto& structure : structures) {
      update_polarizability(fid_polarizability, *structure.first, para.atomic_v, structure.second);
    }
    fclose(fid_polarizability);
  }
}

void Fitness::predict(Parameters& para, float* elite)
{
  // The batches need not follow the order of train.xyz. predict evaluates every batch first and
  // then writes the structures in the order of index_in_file.
  std::vector<std::pair<Dataset*, int>> structures;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    Dataset& dataset = train_set[batch_id][0];
    potential->find_force(para, elite, train_set[batch_id], false, 1);
    copy_predictions_to_host(para, dataset);
    for (int nc = 0; nc < dataset.Nc; ++nc) {
      structures.emplace_back(&dataset, nc);
    }
  }
  std::sort(
    structures.begin(),
    structures.end(),
    [](const std::pair<Dataset*, int>& a, const std::pair<Dataset*, int>& b) {
      return a.first->structures[a.second].index_in_file <
             b.first->structures[b.second].index_in_file;
    });
  write_predictions(para, "train", structures);
}
