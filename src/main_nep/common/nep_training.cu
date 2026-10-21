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

#include "nep_training.cuh"
#include "nep_io.cuh"
#include "energy_difference.cuh"
#include "nep_parameters.cuh"
#include "main_nep/structure.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <numeric>

// Number of structures in one mini-batch. The first n_total % num_batches batches take one
// structure more than the rest, so the batches differ in size by at most one.
static int get_batch_size(const int batch_id, const int n_total, const int num_batches)
{
  const int batch_size_minimal = n_total / num_batches;
  const bool is_larger_batch = batch_id < n_total % num_batches;
  return is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
}

static void find_permuted_indices(
  const int num_batches,
  const std::vector<Structure>& structures,
  std::vector<int>& permuted_indices)
{
  std::vector<double> energy(structures.size());
  for (int nc = 0; nc < structures.size(); ++nc) {
    energy[nc] = structures[nc].energy;
  }

  std::vector<int> energy_index(structures.size());
  std::iota(energy_index.begin(), energy_index.end(), 0);
  std::stable_sort(energy_index.begin(), energy_index.end(), [&energy](size_t i1, size_t i2) {
    return energy[i1] < energy[i2];
  });

  int count = 0;
  for (int b = 0; b < num_batches; ++b) {
    const int batch_size = get_batch_size(b, structures.size(), num_batches);
    for (int c = 0; c < batch_size; ++c) {
      permuted_indices[count + c] = energy_index[b + num_batches * c];
    }
    count += batch_size;
  }
}

static void reorder(const int num_batches, std::vector<Structure>& structures)
{
  std::vector<int> configuration_id(structures.size());
  find_permuted_indices(num_batches, structures, configuration_id);

  std::vector<Structure> structures_reordered;
  structures_reordered.reserve(structures.size());
  for (int nc = 0; nc < structures.size(); ++nc) {
    structures_reordered.push_back(structures[configuration_id[nc]]);
  }
  structures.swap(structures_reordered);
}

NEP_Training::NEP_Training(NEP_Parameters& para, NEP_Data_Reader& reader)
{
  CHECK(gpuGetDeviceCount(&device_count));
  prepare_datasets(para, device_count, reader);
  find_dataset_sizes(N, Nc);
}

void NEP_Training::prepare_datasets(
  NEP_Parameters& para, const int deviceCount, NEP_Data_Reader& reader)
{
  std::vector<Structure> structures_train;
  reader.read_structures(true, structures_train);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  const bool uses_grouped_batches = reader.uses_grouped_batches();
  if (
    (para.prediction == 0) && (para.batch_size < structures_train.size()) && !uses_grouped_batches) {
    reorder(num_batches, structures_train);
  }
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

  reader.prepare_train(structures_train, batch_size_old, num_batches, batch_sizes);

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
        para, structures_train, count - batch_size, count, device_id, batch_id);
      reader.prepare_dataset(train_set[batch_id][device_id]);
      print_line_2();
    }
  }

  std::vector<Structure> structures_test;
  has_test_set = reader.read_structures(false, structures_test);
  reader.prepare_test(structures_test, has_test_set);
  if (has_test_set) {
    test_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing test_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id, -1);
      reader.prepare_dataset(test_set[device_id]);
      print_line_2();
    }
  }
}

void NEP_Training::find_dataset_sizes(int& N, int& Nc)
{
  N = -1;
  Nc = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  if (has_test_set) {
    N = test_set[0].N;
    Nc = test_set[0].Nc;
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

    if (train_set[n][0].max_NN_radial > max_NN_radial) {
      max_NN_radial = train_set[n][0].max_NN_radial;
    }
    if (train_set[n][0].max_NN_angular > max_NN_angular) {
      max_NN_angular = train_set[n][0].max_NN_angular;
    }
  }
}

NEP_Training::~NEP_Training()
{
  if (fid_loss_out != NULL) {
    fclose(fid_loss_out);
  }
}

void NEP_Training::open_loss_file(
  const NEP_Parameters& para, const char* columns, bool include_ediff)
{
  fid_loss_out = my_fopen("loss.out", "a");
  fprintf(fid_loss_out, "# format_version 1\n");
  fprintf(fid_loss_out, "# output_interval %d\n", para.output_interval);
  fprintf(fid_loss_out, "# columns generation total L1 L2%s", columns);
  if (include_ediff) {
    fprintf(fid_loss_out, " rmse_ediff_train rmse_ediff_test");
  }
  fprintf(fid_loss_out, "\n");
  fflush(fid_loss_out);
}

void NEP_Training::write_model(NEP_Parameters& para, const TrainingReport& report)
{
  FILE* file = my_fopen("nep.txt", "w");
  write_nep_model(file, para, report.parameters, max_NN_radial, max_NN_angular);
  fclose(file);
  if (!report.checkpoint.empty()) {
    file = my_fopen(report.checkpoint.c_str(), "w");
    write_nep_model(file, para, report.parameters, max_NN_radial, max_NN_angular);
    fclose(file);
  }
}

void NEP_Training::get_prediction_order(std::vector<std::pair<Dataset*, int>>& structures)
{
  structures.clear();
  for (int batch = 0; batch < num_batches; ++batch) {
    Dataset& data = train_set[batch][0];
    for (int nc = 0; nc < data.Nc; ++nc) {
      structures.emplace_back(&data, nc);
    }
  }
  std::sort(
    structures.begin(),
    structures.end(),
    [](const std::pair<Dataset*, int>& a, const std::pair<Dataset*, int>& b) {
      return a.first->structures[a.second].index_in_file <
             b.first->structures[b.second].index_in_file;
    });
}

void NEP_Energy_Loss::initialize(const NEP_Energy_Parameters& para)
{
  fitness_energy.resize(para.population_size * (para.num_types + 1));
  fitness_force.resize(para.population_size * (para.num_types + 1));
  fitness_virial.resize(para.population_size * (para.num_types + 1));
  fitness_ediff.resize(para.population_size, 0.0f);
}

void NEP_Energy_Loss::evaluate(
  NEP_Energy_Parameters& para, Dataset& data, EnergyDifference& energy_difference,
  int batch, int candidate, int device_id)
{
  double energy_shift_per_structure_not_used;
  auto rmse_energy_array =
    data.get_rmse_energy(para, energy_shift_per_structure_not_used, true, true, device_id);
  auto rmse_force_array = data.get_rmse_force(para, true, device_id);
  auto rmse_virial_array = data.get_rmse_virial(para, true, device_id);
  for (int t = 0; t <= para.num_types; ++t) {
    int index = candidate + t * para.population_size;
    fitness_energy[index] = para.lambda_e * rmse_energy_array[t];
    fitness_force[index] = para.lambda_f * rmse_force_array[t];
    fitness_virial[index] = para.lambda_v * rmse_virial_array[t];
  }
  fitness_ediff[candidate] =
    para.lambda_d * energy_difference.get_rmse_train(data, batch, device_id);
}

void NEP_Energy_Loss::accumulate_fitness(const NEP_Energy_Parameters& para, float* fitness_total)
{
  for (int t = 0; t <= para.num_types; ++t) {
    for (int p = 0; p < para.population_size; ++p) {
      int index = p + t * para.population_size;
      fitness_total[index] = fitness_total[index] + fitness_energy[index] +
                             fitness_force[index] + fitness_virial[index];
    }
  }
}

void NEP_Energy_Loss::accumulate_energy_difference(
  const NEP_Energy_Parameters& para, float* fitness_total)
{
  for (int t = 0; t <= para.num_types; ++t) {
    for (int p = 0; p < para.population_size; ++p) {
      fitness_total[p + t * para.population_size] += fitness_ediff[p];
    }
  }
}

NEP_Energy_Error NEP_Energy_Loss::get_error(
  NEP_Energy_Parameters& para, Dataset& data, EnergyDifference& energy_difference, bool is_train, int batch)
{
  NEP_Energy_Error error;
  error.energy = data.get_rmse_energy(para, error.energy_shift, false, is_train, 0).back();
  error.force = data.get_rmse_force(para, false, 0).back();
  error.virial = data.get_rmse_virial(para, false, 0).back();
  if (para.has_ediff_combinations) {
    error.ediff = is_train ? energy_difference.get_rmse_train(data, batch, 0) :
                            energy_difference.get_rmse_test(data, 0);
  }
  return error;
}
