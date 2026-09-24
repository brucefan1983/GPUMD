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

#include "fitness.cuh"
#include "gnep.cuh"
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
#include <thread>
#include <vector>

#ifdef GNEP_TEST_DIAGNOSTICS
static void write_diagnostic_values(
  const char* filename, const float* values, const int count)
{
  FILE* output = my_fopen(filename, "w");
  for (int i = 0; i < count; ++i) {
    fprintf(output, "%.9e\n", values[i]);
  }
  fclose(output);
}

static void write_diagnostic_values(
  const char* filename, const std::vector<float>& values)
{
  write_diagnostic_values(filename, values.data(), static_cast<int>(values.size()));
}

static void write_diagnostic_values(
  const char* filename, const std::vector<double>& values)
{
  FILE* output = my_fopen(filename, "w");
  for (const double value : values) {
    fprintf(output, "%.17e\n", value);
  }
  fclose(output);
}
#endif

static __global__ void reduce_configuration_gradients(
  const int num_configurations,
  const int num_variables,
  const double* configuration_gradients,
  double* local_gradient)
{
  const int variable = threadIdx.x + blockIdx.x * blockDim.x;
  if (variable >= num_variables) {
    return;
  }
  double sum = 0.0;
  for (int configuration = 0; configuration < num_configurations; ++configuration) {
    sum += configuration_gradients[configuration * num_variables + variable];
  }
  local_gradient[variable] = sum;
}

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
  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));

#ifdef GNEP_TEST_DIAGNOSTICS
  std::remove("gnep_diagnostic_shards.tsv");
#endif

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
  batch_indices.resize(num_batches);
  batch_type_sums.resize(num_batches);
  batch_energies.resize(num_batches);
  std::vector<GNEPDeviceCapacity> capacities(deviceCount);
  int count = 0;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    const int batch_size_minimal = structures_train.size() / num_batches;
    const bool is_larger_batch =
      batch_id + batch_size_minimal * num_batches < structures_train.size();
    const int batch_size = is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
    count += batch_size;
    const int batch_begin = count - batch_size;
    const int active_devices = std::min(deviceCount, batch_size);
    GNEPTrainingBatch& batch = train_set[batch_id];
    batch.num_configurations = batch_size;
    batch.shards.resize(active_devices);
    batch.configuration_indices.resize(active_devices);
    batch.output_structures.insert(
      batch.output_structures.end(),
      structures_train.begin() + batch_begin,
      structures_train.begin() + count);

    std::vector<int> ranked_indices(batch_size);
    for (int i = 0; i < batch_size; ++i) {
      ranked_indices[i] = batch_begin + i;
    }
    std::stable_sort(
      ranked_indices.begin(), ranked_indices.end(), [&](const int lhs, const int rhs) {
        if (structures_train[lhs].num_atom != structures_train[rhs].num_atom) {
          return structures_train[lhs].num_atom > structures_train[rhs].num_atom;
        }
        return lhs < rhs;
      });
    std::vector<int> atom_load(active_devices, 0);
    for (const int configuration_index : ranked_indices) {
      int target = 0;
      for (int device_id = 1; device_id < active_devices; ++device_id) {
        if (atom_load[device_id] < atom_load[target]) {
          target = device_id;
        }
      }
      batch.configuration_indices[target].push_back(configuration_index);
      atom_load[target] += structures_train[configuration_index].num_atom;
    }
    for (int device_id = 0; device_id < active_devices; ++device_id) {
      std::sort(
        batch.configuration_indices[device_id].begin(),
        batch.configuration_indices[device_id].end());
#ifdef GNEP_TEST_DIAGNOSTICS
      FILE* shard_output = my_fopen("gnep_diagnostic_shards.tsv", "a");
      for (const int configuration_index : batch.configuration_indices[device_id]) {
        fprintf(
          shard_output,
          "%d\t%d\t%d\t%d\n",
          batch_id,
          device_id,
          configuration_index,
          structures_train[configuration_index].num_atom);
      }
      fclose(shard_output);
#endif
    }

    printf("\nBatch %d:\n", batch_id);
    printf(
      "Number of configurations = %d; active devices = %d.\n",
      batch_size,
      active_devices);
    for (int device_id = 0; device_id < active_devices; ++device_id) {
      print_line_1();
      printf(
        "Constructing shard on device %d: %zu configurations, %d atoms.\n",
        device_id,
        batch.configuration_indices[device_id].size(),
        atom_load[device_id]);
      CHECK(cudaSetDevice(device_id));
      std::vector<Structure> shard_structures;
      shard_structures.reserve(batch.configuration_indices[device_id].size());
      for (const int configuration_index : batch.configuration_indices[device_id]) {
        shard_structures.push_back(structures_train[configuration_index]);
      }
      batch.shards[device_id].construct(
        para, shard_structures, true, 0, shard_structures.size(), device_id);
      const Dataset& shard = batch.shards[device_id];
      capacities[device_id].max_atoms = std::max(capacities[device_id].max_atoms, shard.N);
      capacities[device_id].max_training_atoms =
        std::max(capacities[device_id].max_training_atoms, shard.N);
      capacities[device_id].max_configurations =
        std::max(capacities[device_id].max_configurations, shard.Nc);
      capacities[device_id].max_radial_pairs =
        std::max(capacities[device_id].max_radial_pairs, shard.N * shard.max_NN_radial);
      capacities[device_id].max_angular_pairs =
        std::max(capacities[device_id].max_angular_pairs, shard.N * shard.max_NN_angular);
      print_line_2();
    }

    batch_type_sums[batch_id].assign(para.num_types, 0);
    for (int configuration_index = batch_begin; configuration_index < count; ++configuration_index) {
      const Structure& structure = structures_train[configuration_index];
      batch.num_atoms += structure.num_atom;
      batch.virial_components += structure.has_virial ? 6 : 0;
      batch_energies[batch_id] += structure.energy;
      for (const int type : structure.type) {
        ++batch_type_sums[batch_id][type];
      }
    }
    int batch_max_radial_neighbors = 0;
    int batch_max_angular_neighbors = 0;
    for (const Dataset& shard : batch.shards) {
      batch_max_radial_neighbors =
        std::max(batch_max_radial_neighbors, shard.max_NN_radial);
      batch_max_angular_neighbors =
        std::max(batch_max_angular_neighbors, shard.max_NN_angular);
    }
    capacities[0].max_atoms = std::max(capacities[0].max_atoms, batch.num_atoms);
    capacities[0].max_radial_pairs = std::max(
      capacities[0].max_radial_pairs, batch.num_atoms * batch_max_radial_neighbors);
    capacities[0].max_angular_pairs = std::max(
      capacities[0].max_angular_pairs, batch.num_atoms * batch_max_angular_neighbors);
    batch_indices[batch_id] = batch_id;
  }

  std::vector<Structure> structures_test;
  has_test_set = read_structures(false, para, structures_test);
  if (has_test_set) {
    test_set.resize(1);
    print_line_1();
    printf("Constructing test_set on device 0.\n");
    CHECK(cudaSetDevice(0));
    test_set[0].construct(para, structures_test, false, 0, structures_test.size(), 0);
    capacities[0].max_atoms = std::max(capacities[0].max_atoms, test_set[0].N);
    capacities[0].max_radial_pairs =
      std::max(capacities[0].max_radial_pairs, test_set[0].N * test_set[0].max_NN_radial);
    capacities[0].max_angular_pairs =
      std::max(capacities[0].max_angular_pairs, test_set[0].N * test_set[0].max_NN_angular);
    print_line_2();
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
    GNEPTrainingBatch& batch = train_set[n];
    int batch_max_radial_neighbors = 0;
    int batch_max_angular_neighbors = 0;
    for (const Dataset& shard : batch.shards) {
      batch_max_radial_neighbors =
        std::max(batch_max_radial_neighbors, shard.max_NN_radial);
      batch_max_angular_neighbors =
        std::max(batch_max_angular_neighbors, shard.max_NN_angular);
    }
    if (batch.num_atoms > N) {
      N = batch.num_atoms;
    };
    if (batch.num_atoms * batch_max_radial_neighbors > N_times_max_NN_radial) {
      N_times_max_NN_radial = batch.num_atoms * batch_max_radial_neighbors;
    };
    if (batch.num_atoms * batch_max_angular_neighbors > N_times_max_NN_angular) {
      N_times_max_NN_angular = batch.num_atoms * batch_max_angular_neighbors;
    };

    if (batch_max_radial_neighbors > max_NN_radial) {
      max_NN_radial = batch_max_radial_neighbors;
    }
    if (batch_max_angular_neighbors > max_NN_angular) {
      max_NN_angular = batch_max_angular_neighbors;
    }
  }

  potential.reset(new GNEP(para, capacities));
    
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
    if (para.energy_shift) {
      CHECK(cudaSetDevice(0));
      computeMultiBatchEnergyShiftUniform(
        para.num_types,
        num_batches,
        batch_type_sums,
        batch_energies,
        para.energy_shift_gpu.data(),
        false);  // Is it output in detail
    
      std::vector<float> energy_per_type_host(para.num_types);
      CHECK(cudaMemcpy(energy_per_type_host.data(), para.energy_shift_gpu.data(),
                      sizeof(float) * para.num_types, cudaMemcpyDeviceToHost));
      for (int i = 0; i < para.num_types; ++i) {
        printf("biased %d initialization of neural networks = %f\n", i, energy_per_type_host[i]);
      }
      print_line_2();
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

    optimizer->initialize_parameters(para);
    float* parameters = optimizer->get_parameters();
#ifdef GNEP_TEST_DIAGNOSTICS
    write_diagnostic_values(
      "gnep_diagnostic_initial_parameters.txt", parameters, number_of_variables);
#endif
    for (int n = 0; n < num_batches; ++n) {
      GNEPTrainingBatch& batch = train_set[n];
      std::vector<std::thread> workers;
      for (int device_id = 0; device_id < static_cast<int>(batch.shards.size()); ++device_id) {
        workers.emplace_back([&, device_id]() {
          potential->find_force(
            para,
            parameters,
            false,
            batch.shards,
#ifdef USE_FIXED_SCALER
            false,
#else
            true,
#endif
            true,
            device_id,
            1,
            batch.num_configurations,
            batch.num_atoms,
            batch.virial_components);
        });
      }
      for (std::thread& worker : workers) {
        worker.join();
      }
    }
    para.reduce_and_broadcast_scaler();
#ifdef GNEP_TEST_DIAGNOSTICS
    write_diagnostic_values("gnep_diagnostic_scaler.txt", para.q_scaler_cpu);
#endif
    double energy_squared_error = 0.0;
    double force_squared_error = 0.0;
    double virial_squared_error = 0.0;
    int count = 0;
    int count_force = 0;
    int count_virial = 0;
    int epoch = 0;
    std::chrono::steady_clock::time_point time_begin;
    static float track_total_time = 0.0f; 
    std::mt19937 shuffle_rng;
    if (para.is_seed_set) {
      shuffle_rng.seed(static_cast<unsigned int>(para.seed));
    } else {
      std::random_device rd;
      shuffle_rng.seed(rd());
    }
    for (int step = 0; step < maximum_steps; ++step) {
      int batch_id = step % num_batches;
      if (batch_id == 0) {
        std::shuffle(batch_indices.begin(), batch_indices.end(), shuffle_rng);
        time_begin = std::chrono::steady_clock::now();
        energy_squared_error = 0.0;
        force_squared_error = 0.0;
        virial_squared_error = 0.0;
        count = 0;
        count_force = 0;
        count_virial = 0;
      }
      batch_id = batch_indices[batch_id];
      GNEPTrainingBatch& batch = train_set[batch_id];
      int Nc = batch.num_configurations;
      if (para.lr_restart_enable) {
        update_learning_rate_cos_restart(lr, step, num_batches, para);
      } else {
        update_learning_rate_cos(lr, step, num_batches, para);
      }
      const auto step_begin = std::chrono::steady_clock::now();
      std::vector<std::thread> workers;
      for (int device_id = 0; device_id < static_cast<int>(batch.shards.size()); ++device_id) {
        workers.emplace_back([&, device_id]() {
          potential->find_force(
            para,
            parameters,
            true,
            batch.shards,
            false,
            true,
            device_id,
            1,
            batch.num_configurations,
            batch.num_atoms,
            batch.virial_components);
        });
      }
      for (std::thread& worker : workers) {
        worker.join();
      }

      double energy_error_sum = 0.0;
      double force_error_sum = 0.0;
      double virial_error_sum = 0.0;
      std::vector<double> reduced_gradient(number_of_variables, 0.0);
      for (int device_id = 0; device_id < static_cast<int>(batch.shards.size()); ++device_id) {
        Dataset& shard = batch.shards[device_id];
        const auto mse_energy_array = shard.get_mse_energy(para, true, device_id);
        shard.get_mse_force(para, true, device_id);
        const auto mse_virial_array = shard.get_mse_virial(para, true, device_id);
        energy_error_sum += mse_energy_array.back() * shard.Nc;
        // Rebuild the global force numerator from complete configuration
        // shards. Division by 3*N_global happens only after all shards have
        // contributed.
        for (int configuration = 0; configuration < shard.Nc; ++configuration) {
          const double weight = shard.weight_cpu[configuration];
          force_error_sum += weight * weight * shard.error_cpu_f[configuration];
        }
        virial_error_sum += mse_virial_array.back() * shard.sum_virial_Nc * 6;

        CHECK(cudaSetDevice(device_id));
        Gradients& gradients = potential->getGradients(device_id);
        const int reduction_block_size = 256;
        const int reduction_grid_size =
          (number_of_variables + reduction_block_size - 1) / reduction_block_size;
        reduce_configuration_gradients<<<reduction_grid_size, reduction_block_size>>>(
          shard.Nc,
          number_of_variables,
          gradients.grad_sum.data(),
          gradients.local_sum.data());
        GPU_CHECK_KERNEL
        std::vector<double> local_gradient(number_of_variables, 0.0);
        gradients.local_sum.copy_to_host(local_gradient.data());
#ifdef GNEP_TEST_DIAGNOSTICS
        if (step == 0) {
          const std::string filename =
            "gnep_diagnostic_gradient_device" + std::to_string(device_id) + ".txt";
          write_diagnostic_values(filename.c_str(), local_gradient);
        }
#endif
        for (int variable = 0; variable < number_of_variables; ++variable) {
          reduced_gradient[variable] += local_gradient[variable];
        }
      }
      std::vector<float> global_gradient(number_of_variables);
      for (int variable = 0; variable < number_of_variables; ++variable) {
        global_gradient[variable] = static_cast<float>(reduced_gradient[variable]);
      }
#ifdef GNEP_TEST_DIAGNOSTICS
      if (step == 0) {
        write_diagnostic_values("gnep_diagnostic_global_gradient.txt", global_gradient);
      }
#endif

      float mse_energy_train = static_cast<float>(energy_error_sum / batch.num_configurations);
      float mse_force_train = static_cast<float>(
        force_error_sum / (batch.num_atoms * 3.0));
      float mse_virial_train = batch.virial_components > 0
        ? static_cast<float>(virial_error_sum / batch.virial_components)
        : 0.0f;
#ifdef GNEP_TEST_DIAGNOSTICS
      if (step == 0) {
        FILE* metrics = my_fopen("gnep_diagnostic_metrics.tsv", "w");
        fprintf(
          metrics,
          "global\t%d\t%d\t%d\t%.17e\t%.17e\t%.17e\n",
          batch.num_configurations,
          batch.num_atoms,
          batch.virial_components,
          mse_energy_train,
          mse_force_train,
          mse_virial_train);
        for (int device_id = 0; device_id < static_cast<int>(batch.shards.size()); ++device_id) {
          const Dataset& shard = batch.shards[device_id];
          fprintf(
            metrics,
            "device%d\t%d\t%d\t%d\n",
            device_id,
            shard.Nc,
            shard.N,
            shard.sum_virial_Nc * 6);
        }
        fclose(metrics);
      }
#endif
      energy_squared_error += energy_error_sum;
      force_squared_error += force_error_sum;
      virial_squared_error += virial_error_sum;
      count += Nc;
      count_force += batch.num_atoms * 3;
      count_virial += batch.virial_components;
      optimizer->update(lr, global_gradient);
#ifdef GNEP_TEST_DIAGNOSTICS
      if (step == 0) {
        write_diagnostic_values(
          "gnep_diagnostic_updated_parameters.txt", parameters, number_of_variables);
        std::vector<float> first_moment;
        std::vector<float> second_moment;
        optimizer->copy_moments_to_host(first_moment, second_moment);
        write_diagnostic_values("gnep_diagnostic_first_moment.txt", first_moment);
        write_diagnostic_values("gnep_diagnostic_second_moment.txt", second_moment);
      }
#endif
      const float step_seconds = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - step_begin).count();
      printf(
        "GNEP step %d: visible_gpus=%d active_gpus=%zu configurations=%d atoms=%d wall_time=%.6f s\n",
        step + 1,
        deviceCount,
        batch.shards.size(),
        batch.num_configurations,
        batch.num_atoms,
        step_seconds);

      if ((step + 1) % num_batches == 0) {
        float time_used = std::chrono::duration<float>(
          std::chrono::steady_clock::now() - time_begin).count();
        track_total_time += time_used; 
        float rmse_energy_train = sqrt(energy_squared_error / count);
        float rmse_force_train = sqrt(force_squared_error / count_force);
        float rmse_virial_train = count_virial > 0
          ? sqrt(virial_squared_error / count_virial)
          : 0.0f;
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
    std::ifstream input("nep.txt");
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open nep.txt.");
    }
    std::vector<std::string> tokens;
    std::vector<float> parameters(number_of_variables);
    tokens = get_tokens(input);
    int num_lines_to_be_skipped = 5;
    if (
      tokens[0] == "nep5_zbl") {
      num_lines_to_be_skipped = 6;
    }

    for (int n = 0; n < num_lines_to_be_skipped; ++n) {
      tokens = get_tokens(input);
    }
    for (int n = 0; n < number_of_variables_ann; ++n) {
      tokens = get_tokens(input);
      parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    tokens = get_tokens(input);
    for (int n = number_of_variables_ann; n < number_of_variables; ++n) {
      tokens = get_tokens(input);
      parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    for (int d = 0; d < para.dim; ++d) {
      tokens = get_tokens(input);
      para.q_scaler_cpu[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    para.q_scaler_gpu(0).copy_from_host(para.q_scaler_cpu.data());
    predict(para, parameters.data());
  }
}

void Fitness::update_learning_rate_cos(float& lr, int step, int num_batches, Parameters& para) {
  const int warmup_epochs = 1; 
  const int warmup_steps = warmup_epochs * num_batches;
  float progress, smooth_progress;
  if (step < warmup_steps) {
    progress = float(step) / warmup_steps;
    lr = stop_lr + progress * (start_lr - stop_lr);
    return;
  }
  progress = float(step - warmup_steps) / (maximum_steps - warmup_steps);
  smooth_progress = 0.5f * (1.0f + cosf(PI * progress));
  lr = stop_lr + (start_lr - stop_lr) * smooth_progress;
}

void Fitness::update_learning_rate_cos_restart(float& lr, int step, int num_batches, Parameters& para) {
  const int warmup_epochs = para.lr_warmup_epochs;
  const int warmup_steps = warmup_epochs * num_batches;
  float progress, smooth_progress;
  if (step < warmup_steps) {
    progress = float(step) / warmup_steps;
    lr = stop_lr + progress * (start_lr - stop_lr);
    return;
  }
  const int initial_restart_period = para.lr_restart_initial_period_epochs * num_batches;
  const float period_factor = para.lr_restart_period_factor;
  const float decay_factor = para.lr_restart_decay_factor;
  
  int steps_since_warmup = step - warmup_steps;
  int total_steps = maximum_steps - warmup_steps; 
  int current_cycle = 0;
  int cycle_start_step = 0;
  int cycle_length = initial_restart_period;
  
  int cumulative_steps = 0;
  while (cumulative_steps + cycle_length <= steps_since_warmup) {
    cumulative_steps += cycle_length;
    cycle_start_step = cumulative_steps;
    current_cycle++;
    cycle_length = int(initial_restart_period * powf(period_factor, current_cycle));
  }

  if (cumulative_steps + cycle_length > total_steps) {
    cycle_length = total_steps - cumulative_steps;
  }
  
  int steps_in_current_cycle = steps_since_warmup - cycle_start_step;
  float cycle_progress = float(steps_in_current_cycle) / cycle_length;

  float cycle_max_lr = start_lr * powf(decay_factor, current_cycle);
  if (cycle_max_lr < stop_lr) cycle_max_lr = stop_lr;
  
  smooth_progress = 0.5f * (1.0f + cosf(PI * cycle_progress));
  lr = stop_lr + (cycle_max_lr - stop_lr) * smooth_progress;
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

void Fitness::write_gnep_txt(FILE* fid_gnep, Parameters& para, float* parameters)
{
  if (para.enable_zbl) {
    fprintf(fid_gnep, "nep5_zbl %d ", para.num_types);
  } else {
    fprintf(fid_gnep, "nep5 %d ", para.num_types);
  }

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_gnep, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_gnep, "\n");
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_gnep, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(fid_gnep, "zbl %g %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(fid_gnep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }

  fprintf(fid_gnep, "cutoff %g %g ", para.rc_radial[0], para.rc_angular[0]);
  if (para.has_multiple_cutoffs) {
    for (int n = 1; n < para.num_types; ++n) {
      fprintf(fid_gnep, "%g %g ", para.rc_radial[n], para.rc_angular[n]);
    }
  }
  fprintf(fid_gnep, "%d %d\n", max_NN_radial, max_NN_angular);

  fprintf(fid_gnep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_gnep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_gnep, "l_max %d 0 0\n", para.L_max);

  fprintf(fid_gnep, "ANN %d %d\n", para.num_neurons1, 0);
  for (int m = 0; m < para.number_of_variables_ann; ++m) {
    fprintf(fid_gnep, "%15.7e\n", parameters[m]);
  }
  fprintf(fid_gnep, "%15.7e\n", 0.0);
  for (int m = para.number_of_variables_ann; m < para.number_of_variables; ++m) {
    fprintf(fid_gnep, "%15.7e\n", parameters[m]);
  }
  CHECK(cudaSetDevice(0));
  para.q_scaler_gpu(0).copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_gnep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_gnep, "%15.7e\n", para.zbl_para[d]);
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
    potential->find_force(
      para,
      parameters,
      false,
      test_set,
      false,
      true,
      0,
      1,
      test_set[0].Nc,
      test_set[0].N,
      test_set[0].sum_virial_Nc * 6);
    auto mse_energy_test_array = test_set[0].get_mse_energy(para, false, 0);
    auto mse_force_test_array = test_set[0].get_mse_force(para, false, 0);
    auto mse_virial_test_array = test_set[0].get_mse_virial(para, false, 0);
    rmse_energy_test = sqrt(mse_energy_test_array.back());
    rmse_force_test = sqrt(mse_force_test_array.back());
    rmse_virial_test = sqrt(mse_virial_test_array.back()); 
  }

  FILE* fid_gnep = my_fopen("nep.txt", "w");
  write_gnep_txt(fid_gnep, para, parameters);
  fclose(fid_gnep);

  if (0 == (epoch + 1) % 100) {
    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo = localtime(&rawtime);
    char buffer[200];
    strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_epoch", timeinfo);
    std::string filename(buffer + std::to_string(epoch + 1) + ".txt");

    FILE* fid_gnep = my_fopen(filename.c_str(), "w");
    write_gnep_txt(fid_gnep, para, parameters);
    fclose(fid_gnep);
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
    GNEPTrainingBatch& batch = train_set[batch_id];
    std::vector<Dataset> output_dataset(1);
    output_dataset[0].construct(
      para, batch.output_structures, false, 0, batch.output_structures.size(), 0);
    potential->find_force(
      para,
      parameters,
      false,
      output_dataset,
      false,
      true,
      0,
      1,
      batch.num_configurations,
      batch.num_atoms,
      batch.virial_components);
    update_energy_force_virial(
      fid_energy, fid_force, fid_virial, fid_stress, output_dataset[0]);
  }
  fclose(fid_energy);
  fclose(fid_force);
  fclose(fid_virial);
  fclose(fid_stress);
}
