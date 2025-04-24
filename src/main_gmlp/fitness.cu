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
#include "gmlp.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/least_square.cuh"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

Fitness::Fitness(Parameters& para, Adam* adam)
  : optimizer(adam)
{
  maximum_epochs = para.epoch;
  number_of_variables = para.number_of_variables;
  number_of_variables_ann = para.number_of_variables_ann;
  number_of_variables_descriptor = para.number_of_variables_descriptor;
  lr = para.lr;
  start_lr = para.start_lr;
  stop_lr = para.stop_lr;
  decay_step = para.decay_step;
  start_pref_e = para.start_pref_e;  
  start_pref_f = para.start_pref_f;
  start_pref_v = para.start_pref_v;
  stop_pref_e = para.stop_pref_e;
  stop_pref_f = para.stop_pref_f;
  stop_pref_v = para.stop_pref_v;

  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));

  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  maximum_steps = num_batches * maximum_epochs;
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
      CHECK(cudaSetDevice(device_id));
      train_set[batch_id][device_id].construct(
        para, structures_train, true, count - batch_size, count, device_id);
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
      CHECK(cudaSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, false, 0, structures_test.size(), device_id);
      print_line_2();
    }
  }

  N = -1;
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
    new GMLP(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
    
  optimizer->initialize_parameters(para);

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

void Fitness::compute(Parameters& para)
{
  print_line_1();
  if (para.prediction == 0) {
    printf("Started training.\n");
  } else {
    printf("Started predicting.\n");
  }
  print_line_2();

  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));

  if (para.prediction == 0) {
    float* parameters = optimizer->get_parameters();
    std::vector<int> batch_indices(num_batches);
    std::vector<std::vector<int>> batch_type_sums(num_batches);
    std::vector<float> batch_energies(num_batches);
    for (int n = 0; n < num_batches; ++n) {
      batch_indices[n] = n;
      potential->find_force(
        para,
        parameters,
        false,
        train_set[n],
#ifdef USE_FIXED_SCALER
        false,
#else
        true,
#endif
        true,
        1);
      std::vector<int> type_sum_host(para.num_types);
      train_set[n][0].type_sum.copy_to_host(type_sum_host.data());
      batch_type_sums[n] = type_sum_host;
      batch_energies[n] = train_set[n][0].sum_energy_ref;
    }
    if (para.energy_shift) {
      computeMultiBatchEnergyShift(
        para.num_types,
        num_batches,
        batch_type_sums,
        batch_energies,
        para.energy_shift_gpu.data(),
        false);  // Is it output in detail
      para.calculate_energy_shift = false;
    
      std::vector<float> energy_per_type_host(para.num_types);
      CHECK(cudaMemcpy(energy_per_type_host.data(), para.energy_shift_gpu.data(),
                      sizeof(float) * para.num_types, cudaMemcpyDeviceToHost));
      for (int i = 0; i < para.num_types; ++i) {
        printf("biased %d initialization of neural networks = %f\n", i, energy_per_type_host[i]);
      }
      print_line_2();
    optimizer->initialize_parameters(para);
    }
    printf(
      "%-8s%-13s%-13s%-13s%-13s%-13s%-13s%-13s%-15s%-10s\n", 
      "Epoch",
      "Total-Loss",
      "RMSE-E-Train",
      "RMSE-F-Train", 
      "RMSE-V-Train",
      "RMSE-E-Test",
      "RMSE-F-Test",
      "RMSE-V-Test",
      "Learning-Rate",
      "Time(s)");
    float mse_energy;
    float mse_force;
    float mse_virial;
    int count;
    int count_virial;
    int epoch = 0;
    clock_t time_begin;
    clock_t time_finish;
    static float track_total_time = 0.0f; 
    for (int step = 0; step < maximum_steps; ++step) {
      int batch_id = step % num_batches;
      if (batch_id == 0) {
        std::random_device rd;
        std::mt19937 g(rd());
        // std::mt19937 g(2025);
        std::shuffle(batch_indices.begin(), batch_indices.end(), g);
        time_begin = clock();
        mse_energy = 0.0f;
        mse_force = 0.0f;
        mse_virial = 0.0f;
        count = 0;
        count_virial = 0;
      }
      batch_id = batch_indices[batch_id];
      int Nc = train_set[batch_id][0].Nc;
      int virial_Nc = train_set[batch_id][0].sum_virial_Nc;
      // printf("Finding force for batch %d\n", batch_id);
      if (maximum_epochs <= 100) {
        update_learning_rate_cos(lr, step, num_batches, para);
      } else {
        update_learning_rate_cos_restart(lr, step, num_batches, para);
      }
      potential->find_force(
      para,
      parameters,
      true,
      train_set[batch_id],
      false,
      true,
      deviceCount);
      auto rmse_energy_array = train_set[batch_id][0].get_rmse_energy(para, true, 0);
      auto rmse_force_array = train_set[batch_id][0].get_rmse_force(para, true, 0);
      auto rmse_virial_array = train_set[batch_id][0].get_rmse_virial(para, true, 0);
      float mse_energy_train = rmse_energy_array.back();
      float mse_force_train = rmse_force_array.back();
      float mse_virial_train = rmse_virial_array.back();
      mse_energy += mse_energy_train * Nc;
      mse_force += mse_force_train * Nc;
      mse_virial += mse_virial_train * virial_Nc;
      count += Nc;
      count_virial += virial_Nc;
      auto& grad = potential->getGradients();
      optimizer->update(lr, grad.grad_sum.data(), num_batches, maximum_steps);

      if ((step + 1) % num_batches == 0) {
        time_finish = clock();
        float time_used = (time_finish - time_begin) / float(CLOCKS_PER_SEC);
        track_total_time += time_used; 
        float rmse_energy_train = sqrt(mse_energy / count);
        float rmse_force_train = sqrt(mse_force / count);
        float rmse_virial_train = count_virial > 0 ? sqrt(mse_virial / count_virial) : 0.0f;
        float total_loss_train = para.lambda_e * rmse_energy_train + para.lambda_f * rmse_force_train + para.lambda_v * rmse_virial_train;
        report_error(
          para,
          track_total_time, 
          epoch,
          total_loss_train,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          lr,
          parameters
        );
        optimizer->output_parameters(para);
        epoch++;
      }
    } // end of step loop
  } else {
    std::ifstream input("gmlp.txt");
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open gmlp.txt.");
    }
    std::vector<std::string> tokens;
    float parameters[number_of_variables];
    tokens = get_tokens(input);
    int num_lines_to_be_skipped = 5;
    if (
      tokens[0] == "gmlp_zbl") {
      num_lines_to_be_skipped = 6;
    }

    for (int n = 0; n < num_lines_to_be_skipped; ++n) {
      tokens = get_tokens(input);
    }
    for (int n = 0; n < number_of_variables; ++n) {
      tokens = get_tokens(input);
      parameters[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
    for (int d = 0; d < para.dim; ++d) {
      tokens = get_tokens(input);
      para.q_scaler_cpu[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
    para.q_scaler_gpu[0].copy_from_host(para.q_scaler_cpu.data());
    predict(para, parameters);
  }
}

void Fitness::update_learning_rate(float& lr, int step, Parameters& para) {
  if (step >= maximum_steps) {
    lr = stop_lr;
  } else if (step % decay_step == 0 && step != 0) {
    decay_rate = exp(log(stop_lr / start_lr) / (maximum_steps / decay_step));
    lr = start_lr * pow(decay_rate, step / decay_step);
  }
  para.lambda_e = stop_pref_e + (start_pref_e - stop_pref_e) * lr / start_lr;
  para.lambda_f = stop_pref_f + (start_pref_f - stop_pref_f) * lr / start_lr;
  para.lambda_v = stop_pref_v + (start_pref_v - stop_pref_v) * lr / start_lr;
}

void Fitness::update_learning_rate_cos(float& lr, int step, int num_batches, Parameters& para) {
  const int warmup_epochs = 1;  // warmup & restart
  const int warmup_steps = warmup_epochs * num_batches;
  float progress, smooth_progress;
  if (step < warmup_steps) {
    progress = float(step) / warmup_steps;
    // lr = start_lr - (start_lr - stop_lr) * (0.5f * (1.0f + cosf((1.0f - progress) * PI)));
    // smooth_progress = 0.5f * (1.0f - cosf(progress * PI));
    // para.lambda_e = start_pref_e + (stop_pref_e - start_pref_e) * smooth_progress;
    // para.lambda_f = start_pref_f - 0.2f * start_pref_f * smooth_progress;
    // para.lambda_v = start_pref_v + (stop_pref_v - start_pref_v) * smooth_progress;
    lr = stop_lr + progress * (start_lr - stop_lr);
    para.lambda_e = start_pref_e;
    para.lambda_f = start_pref_f;
    para.lambda_v = start_pref_v;
    return;
  }
  progress = float(step - warmup_steps) / (maximum_steps - warmup_steps);
  smooth_progress = 0.5f * (1.0f + cosf(PI * progress));
  lr = stop_lr + (start_lr - stop_lr) * smooth_progress;
  // para.lambda_e = stop_pref_e;
  // para.lambda_f = stop_pref_f + (0.8f * start_pref_f - stop_pref_f) * smooth_progress;
  // para.lambda_v = stop_pref_v;
  if (progress < 0.3f) {
    para.lambda_e = stop_pref_e + 0.5f * (start_pref_e - stop_pref_e);
    para.lambda_f = start_pref_f;
    para.lambda_v = start_pref_v; 
  } else if (progress < 0.7f) {
    float mid_progress = (progress - 0.3f) / 0.4f;
    para.lambda_e = stop_pref_e + 0.5f * (start_pref_e - stop_pref_e) * (1.0f - mid_progress);
    para.lambda_f = stop_pref_f + (start_pref_f - stop_pref_f) * (1.0f - mid_progress);
    para.lambda_v = stop_pref_v + (start_pref_v - stop_pref_v) * (1.0f - mid_progress);
  } else {
    para.lambda_e = stop_pref_e;
    para.lambda_f = stop_pref_f;
    para.lambda_v = stop_pref_v;
  }
}

void Fitness::update_learning_rate_cos_restart(float& lr, int step, int num_batches, Parameters& para) {
  const int warmup_epochs = 1;  // warmup
  const int warmup_steps = warmup_epochs * num_batches;
  float progress, smooth_progress;
  if (step < warmup_steps) {
    progress = float(step) / warmup_steps;
    lr = stop_lr + progress * (start_lr - stop_lr);
    smooth_progress = 0.5f * (1.0f - cosf(progress * PI));
    para.lambda_e = start_pref_e;
    para.lambda_f = start_pref_f;
    para.lambda_v = start_pref_v;
    return;
  }
  const int initial_restart_period = 100 * num_batches;  // Initial restart cycle (100 epochs)
  const int min_cycle_length = 10 * num_batches;
  const float period_factor = 0.7f;                     // Period length decay factor
  const float decay_factor = 0.12f;                      // Learning rate upper limit decay factor
  
  int steps_since_warmup = step - warmup_steps;
  int current_cycle = 0;
  int cycle_start_step = 0;
  int cycle_length = initial_restart_period;
  
  int cumulative_steps = 0;
  while (cumulative_steps + cycle_length <= steps_since_warmup) {
    cumulative_steps += cycle_length;
    cycle_start_step = cumulative_steps;
    current_cycle++;
    cycle_length = int(initial_restart_period * powf(period_factor, current_cycle));
    if (cycle_length < min_cycle_length) cycle_length = min_cycle_length;
  }
  
  int steps_in_current_cycle = steps_since_warmup - cycle_start_step;
  float cycle_progress = float(steps_in_current_cycle) / cycle_length;

  float cycle_max_lr = start_lr * powf(decay_factor, current_cycle);
  if (cycle_max_lr < stop_lr) cycle_max_lr = stop_lr;
  
  smooth_progress = 0.5f * (1.0f + cosf(PI * cycle_progress));
  lr = stop_lr + (cycle_max_lr - stop_lr) * smooth_progress;
  
  if (cycle_progress < 0.3f) {
    para.lambda_e = stop_pref_e + 0.5f * (start_pref_e - stop_pref_e);
    para.lambda_f = start_pref_f;
    para.lambda_v = start_pref_v; 
  } else if (cycle_progress < 0.7f) {
    float mid_progress = (cycle_progress - 0.3f) / 0.4f;
    para.lambda_e = stop_pref_e + 0.5f * (start_pref_e - stop_pref_e) * (1.0f - mid_progress);
    para.lambda_f = stop_pref_f + (start_pref_f - stop_pref_f) * (1.0f - mid_progress);
    para.lambda_v = stop_pref_v + (start_pref_v - stop_pref_v) * (1.0f - mid_progress);
  } else {
    para.lambda_e = stop_pref_e;
    para.lambda_f = stop_pref_f;
    para.lambda_v = stop_pref_v;
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
        // ref_value /= dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

void Fitness::write_gmlp_txt(FILE* fid_gmlp, Parameters& para, float* parameters)
{
  if (para.version == 1) {
    if (para.enable_zbl) {
      fprintf(fid_gmlp, "gmlp_zbl %d ", para.num_types);
    } else {
      fprintf(fid_gmlp, "gmlp %d ", para.num_types);
    }
  }

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_gmlp, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_gmlp, "\n");
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_gmlp, "zbl 0 0\n");
    } else {
      fprintf(fid_gmlp, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }
  if (para.use_typewise_cutoff || para.use_typewise_cutoff_zbl) {
    fprintf(
      fid_gmlp,
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
      fid_gmlp,
      "cutoff %g %g %d %d\n",
      para.rc_radial,
      para.rc_angular,
      max_NN_radial,
      max_NN_angular);
  }
  fprintf(fid_gmlp, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_gmlp, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_gmlp, "l_max %d 0 0\n", para.L_max);

  fprintf(fid_gmlp, "ANN %d %d\n", para.num_neurons1, 0);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_gmlp, "%15.7e\n", parameters[m]);
  }
  CHECK(cudaSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_gmlp, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_gmlp, "%15.7e\n", para.zbl_para[d]);
    }
  }
}

void Fitness::report_error(
  Parameters& para,
  float time_used,
  const int epoch,
  const float loss_total,
  const float rmse_energy_train,
  const float rmse_force_train,
  const float rmse_virial_train,
  const float lr,
  float* parameters)
{
  float rmse_energy_test = 0.0f;
  float rmse_force_test = 0.0f;
  float rmse_virial_test = 0.0f;
  if (has_test_set) {
    potential->find_force(para, parameters, false, test_set, false, true, 1);
    auto rmse_energy_test_array = test_set[0].get_rmse_energy(para, false, 0);
    auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
    auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
    rmse_energy_test = sqrt(rmse_energy_test_array.back());
    rmse_force_test = sqrt(rmse_force_test_array.back());
    rmse_virial_test = sqrt(rmse_virial_test_array.back()); 
  }

  FILE* fid_gmlp = my_fopen("gmlp.txt", "w");
  write_gmlp_txt(fid_gmlp, para, parameters);
  fclose(fid_gmlp);

  if (0 == (epoch + 1) % 100) {
    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo = localtime(&rawtime);
    char buffer[200];
    strftime(buffer, sizeof(buffer), "gmlp_y%Y_m%m_d%d_h%H_m%M_s%S_epoch", timeinfo);
    std::string filename(buffer + std::to_string(epoch + 1) + ".txt");

    FILE* fid_gmlp = my_fopen(filename.c_str(), "w");
    write_gmlp_txt(fid_gmlp, para, parameters);
    fclose(fid_gmlp);
  }

  printf(
    "%-8d%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-15.7f%-13.5f\n", 
    epoch + 1,
    loss_total,
    rmse_energy_train,
    rmse_force_train,
    rmse_virial_train,
    rmse_energy_test,
    rmse_force_test,
    rmse_virial_test,
    lr,
    time_used);
  fprintf(
    fid_loss_out,
    "%-8d%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-15.7f%-13.5f\n",
    epoch + 1,
    loss_total,
    rmse_energy_train,
    rmse_force_train,
    rmse_virial_train,
    rmse_energy_test,
    rmse_force_test,
    rmse_virial_test,
    lr,
    time_used);
  fflush(stdout);
  fflush(fid_loss_out);

  if (has_test_set) {
    FILE* fid_force = my_fopen("force_test.out", "w");
    FILE* fid_energy = my_fopen("energy_test.out", "w");
    FILE* fid_virial = my_fopen("virial_test.out", "w");
    FILE* fid_stress = my_fopen("stress_test.out", "w");
    update_energy_force_virial(fid_energy, fid_force, fid_virial, fid_stress, test_set[0]);
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
  }

  if (0 == (epoch + 1) % 10) {
    predict(para, parameters);
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

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  output(false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
}

void Fitness::update_polarizability(FILE* fid_polarizability, Dataset& dataset)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  output(
    false,
    6,
    fid_polarizability,
    dataset.virial_cpu.data(),
    dataset.virial_ref_cpu.data(),
    dataset);
}

void Fitness::predict(Parameters& para, float* parameters)
{
  FILE* fid_force = my_fopen("force_train.out", "w");
  FILE* fid_energy = my_fopen("energy_train.out", "w");
  FILE* fid_virial = my_fopen("virial_train.out", "w");
  FILE* fid_stress = my_fopen("stress_train.out", "w");
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    potential->find_force(para, parameters, false, train_set[batch_id], false, true, 1);
    update_energy_force_virial(
      fid_energy, fid_force, fid_virial, fid_stress, train_set[batch_id][0]);
  }
  fclose(fid_energy);
  fclose(fid_force);
  fclose(fid_virial);
  fclose(fid_stress);
}
