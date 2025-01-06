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

Fitness::Fitness(Parameters& para, Adam* adam)
  : optimizer(adam)
{
  maximum_generation = para.maximum_generation;
  number_of_variables = para.number_of_variables;
  number_of_variables_ann = para.number_of_variables_ann;
  number_of_variables_descriptor = para.number_of_variables_descriptor;
  // start_lr = para.start_lr;
  // stop_lr = para.stop_lr;
  // decay_step = para.decay_step;

  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));

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
    new NEP3(para, N, N_times_max_NN_radial, N_times_max_NN_angular, para.version, deviceCount));
    
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

  if (para.prediction == 0) {

    if (para.train_mode == 0 || para.train_mode == 3) {
      printf(
        "%-8s%-11s%-13s%-13s%-13s%-13s%-13s%-13s%-20s\n", 
        "Step",
        "Total-Loss",
        "RMSE-E-Train",
        "RMSE-F-Train", 
        "RMSE-V-Train",
        "RMSE-E-Test",
        "RMSE-F-Test",
        "RMSE-V-Test",
        "Learning-Rate");
    } else {
      printf(
        "%-8s%-11s%-13s%-13s%-20s\n",
        "Step", 
        "Total-Loss",
        "RMSE-P-Train",
        "RMSE-P-Test",
        "Learning-Rate");
    }
  }
  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount));

  if (para.prediction == 0) {
    double* parameters = optimizer->get_parameters();
    for (int n = 0; n < num_batches; ++n) {
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
    }
    double mse_energy;
    double mse_force;
    double mse_virial;
    int count;
    for (int step = 0; step < maximum_generation; ++step) {
      int batch_id = step % num_batches;
      int Nc = train_set[batch_id][0].Nc;
      if (batch_id == 0) {
        mse_energy = 0.0;
        mse_force = 0.0;
        mse_virial = 0.0;
        count = 0;
      }
      // printf("Finding force for batch %d\n", batch_id);
      update_learning_rate(lr, step);
      para.lambda_e = 1.0 + (0.02 - 1.0) * lr / start_lr;
      para.lambda_f = 1.0 + (1000.0 - 1.0) * lr / start_lr;
      para.lambda_v = 1.0 + (50.0 - 1.0) * lr / start_lr;
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
      double mse_energy_train = rmse_energy_array.back();
      double mse_force_train = rmse_force_array.back();
      double mse_virial_train = rmse_virial_array.back();
      mse_energy += mse_energy_train * Nc;
      mse_force += mse_force_train * Nc;
      mse_virial += mse_virial_train * Nc;
      count += Nc;
      optimizer->update(lr, train_set[batch_id][0].gradients.grad_sum.data());

      if ((step + 1) % num_batches == 0) {
        double rmse_energy_train = sqrt(mse_energy / count);
        double rmse_force_train = sqrt(mse_force / count);
        double rmse_virial_train = sqrt(mse_virial / count);
        double total_loss_train = para.lambda_e * rmse_energy_train + para.lambda_f * rmse_force_train + para.lambda_v * rmse_virial_train;
        report_error(
          para,
          step,
          total_loss_train,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          lr,
          optimizer->get_parameters()
        );
        optimizer->output_parameters(para);
      }
    } // end of step loop
  } else {
    std::ifstream input("nep.txt");
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open nep.txt.");
    }
    std::vector<std::string> tokens;
    double parameters[number_of_variables];
    tokens = get_tokens(input);
    int num_lines_to_be_skipped = 5;
    if (
      tokens[0] == "nep3_zbl" || tokens[0] == "nep4_zbl" || tokens[0] == "nep3_zbl_temperature" ||
      tokens[0] == "nep4_zbl_temperature") {
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

void Fitness::update_learning_rate(double& lr, int step) {
  if (step >= maximum_generation) {
    lr = stop_lr;
  } else if (step % decay_step == 0 && step != 0) {
    decay_rate = exp(log(stop_lr / start_lr) / (maximum_generation / decay_step));
    lr = start_lr * pow(decay_rate, step / decay_step);
  }
}

void Fitness::output(
  bool is_stress,
  int num_components,
  FILE* fid,
  double* prediction,
  double* reference,
  Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int n = 0; n < num_components; ++n) {
      int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
      double data_nc = 0.0;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        data_nc += prediction[offset + m];
      }
      if (!is_stress) {
        fprintf(fid, "%g ", data_nc);
      } else {
        fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
      }
    }
    for (int n = 0; n < num_components; ++n) {
      double ref_value = reference[n * dataset.Nc + nc];
      if (is_stress) {
        // ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
        ref_value /= dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
      }
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, double* parameters)
{
  if (para.train_mode == 0) { // potential model
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
    } else if (para.version == 5) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep5_zbl %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep5 %d ", para.num_types);
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
    fprintf(fid_nep, "%15.7e\n", parameters[m]);
  }
  CHECK(cudaSetDevice(0));
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
  const double loss_total,
  const double rmse_energy_train,
  const double rmse_force_train,
  const double rmse_virial_train,
  const double lr,
  double* parameters)
{
  double rmse_energy_test = 0.0;
  double rmse_force_test = 0.0;
  double rmse_virial_test = 0.0;
  if (has_test_set) {
    potential->find_force(para, parameters, false, test_set, false, true, 1);
    auto rmse_energy_test_array = test_set[0].get_rmse_energy(para, false, 0);
    auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
    auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
    rmse_energy_test = sqrt(rmse_energy_test_array.back());
    rmse_force_test = sqrt(rmse_force_test_array.back());
    rmse_virial_test = sqrt(rmse_virial_test_array.back()); 
  }

  FILE* fid_nep = my_fopen("nep.txt", "w");
  write_nep_txt(fid_nep, para, parameters);
  fclose(fid_nep);

  if (0 == (generation + 1) % 100000) {
    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo = localtime(&rawtime);
    char buffer[200];
    strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
    std::string filename(buffer + std::to_string(generation + 1) + ".txt");

    FILE* fid_nep = my_fopen(filename.c_str(), "w");
    write_nep_txt(fid_nep, para, parameters);
    fclose(fid_nep);
  }

  if (para.train_mode == 0 || para.train_mode == 3) {
    printf(
      "%-8d%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-20.7f\n", 
      generation + 1,
      loss_total,
      rmse_energy_train,
      rmse_force_train,
      rmse_virial_train,
      rmse_energy_test,
      rmse_force_test,
      rmse_virial_test,
      lr);
    fprintf(
      fid_loss_out,
      "%-8d%-11.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-13.5f%-20.7f\n",
      generation + 1,
      loss_total,
      rmse_energy_train,
      rmse_force_train,
      rmse_virial_train,
      rmse_energy_test,
      rmse_force_test,
      rmse_virial_test,
      lr);
  } else {
    printf(
      "%-8d%-11.5f%-13.5f%-13.5f%-20.7f\n",
      generation + 1,
      loss_total,
      rmse_virial_train,
      rmse_virial_test,
      lr);
    fprintf(
      fid_loss_out,
      "%-8d%-11.5f%-13.5f%-13.5f%-20.7f\n",
      generation + 1,
      loss_total,
      rmse_virial_train,
      rmse_virial_test,
      lr);
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

void Fitness::predict(Parameters& para, double* parameters)
{
  if (para.train_mode == 0 || para.train_mode == 3) {
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
  } else if (para.train_mode == 1) {
    FILE* fid_dipole = my_fopen("dipole_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, parameters, false, train_set[batch_id], false, true, 1);
      update_dipole(fid_dipole, train_set[batch_id][0]);
    }
    fclose(fid_dipole);
  } else if (para.train_mode == 2) {
    FILE* fid_polarizability = my_fopen("polarizability_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, parameters, false, train_set[batch_id], false, true, 1);
      update_polarizability(fid_polarizability, train_set[batch_id][0]);
    }
    fclose(fid_polarizability);
  }
}
