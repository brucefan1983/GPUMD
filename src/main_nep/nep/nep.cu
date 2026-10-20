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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep.cuh"
#include "main_nep/common/nep_output.cuh"
#include "main_nep/xyz_reader.cuh"
#include "main_nep/common/nep_energy_reader.cuh"
#include "main_nep/common/nep_model.cuh"
#include "main_nep/common/nep_init.cuh"
#include "main_nep/common/nep_io.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "utilities/error.cuh"
#include <algorithm>

void NEP::initialize(NEP_Potential_Parameters& para, int N, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    initialize_device(para, N, device_id);
  }
  initialize_compiled_kernel(para);
}

void NEP::initialize_model(const NEP_Potential_Parameters& para)
{
  paramb.version = para.version;
  paramb.num_types = para.num_types;
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial[t] = para.rc_radial[t];
    paramb.rc_angular[t] = para.rc_angular[t];
  }
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.has_q_222 = para.has_q_222;
  paramb.has_q_1111 = para.has_q_1111;
  paramb.has_q_112 = para.has_q_112;
  paramb.has_q_123 = para.has_q_123;
  paramb.has_q_233 = para.has_q_233;
  paramb.has_q_134 = para.has_q_134;
  paramb.num_L = para.dim_angular / (para.n_max_angular + 1);
  paramb.dim_angular = para.dim_angular;

  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
#ifdef USE_CJ
  paramb.num_c_radial =
    paramb.num_types * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#else
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#endif

  initialize_nep_zbl(para, zbl);
}

void NEP::initialize_device(const NEP_Potential_Parameters& para, int N, int device_id)
{
  NEP_ANN& ann = annmb[device_id];
  NEP_Data& data = nep_data[device_id];
  ann.dim = para.dim;
  ann.num_neurons1 = para.num_neurons1;
  ann.num_hidden_layers = para.num_hidden_layers;
  ann.num_para = para.number_of_variables;
  if (para.num_hidden_layers == 2) {
    ann.num_neurons2 = para.num_neurons2;
  }
  ann.one_ann_no_bias = para.number_of_variables_ann_1;

  data.descriptors.resize(N * ann.dim);
  data.Fp.resize(N * ann.dim);
  data.sum_fxyz.resize(
    N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  data.parameters.resize(ann.num_para);
}

namespace
{
NEP_Compile_Config make_compile_config(
  const NEP_Potential_Parameters& para)
{
  NEP_Compile_Config config = make_nep_compile_config(para);
  config.model_name = "NEP";
  config.source_file = "main_nep/nep/nep_specialized.cu";
  return config;
}
} // namespace

NEP_Compiled_Evaluation::NEP_Compiled_Evaluation(const NEP_Compile_Config& config)
  : library(config)
{
  if (!library.is_valid()) {
    return;
  }
  valid = common.load(library);
  valid &= library.load_symbol("nep_train_launch_ann_nep", ann_nep_);
  valid &= library.load_symbol("nep_train_launch_ann_temperature", ann_temperature_);
}

void NEP::initialize_compiled_kernel(const NEP_Potential_Parameters& para)
{
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    compiled_kernel_.reset(new NEP_Compiled_Evaluation(
      make_compile_config(para)));
    if (!compiled_kernel_->is_valid()) {
      compiled_kernel_.reset();
    }
  }
}

void NEP::find_force(
  NEP_Potential_Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{
  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    bind_nep_parameters(
      paramb.num_types, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    neighbor[device_id].prepare(para, dataset[device_id], device_id);

    compute_nep_descriptors(
      paramb, annmb[device_id].c, dataset[device_id], neighbor[device_id],
      nep_data[device_id], compiled_kernel_.get());

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      output_nep_descriptors(
        dataset[device_id], annmb[device_id].dim, para.output_descriptor,
        para.q_scaler_cpu, nep_data[device_id].descriptors);
    }

    if (calculate_q_scaler) {
      update_nep_q_scaler(
        dataset[device_id].N, annmb[device_id].dim, nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(), para.q_scaler_max[device_id].data(),
        para.q_scaler_min[device_id].data());
    }

    zero_nep_outputs(dataset[device_id]);

    evaluate_nep_ann(
      annmb[device_id],
      para.model_type == 3 ? temperature_data.get(dataset[device_id]).temperature.data() : nullptr,
      dataset[device_id], nep_data[device_id],
      para.q_scaler_gpu[device_id].data(), compiled_kernel_.get());

    compute_nep_forces(
      paramb, annmb[device_id].c, dataset[device_id], neighbor[device_id],
      nep_data[device_id], compiled_kernel_.get());

    if (zbl.enabled) {
      add_nep_zbl(zbl, dataset[device_id], neighbor[device_id]);
    }
  }
}

void NEP::configure(NEP_Potential_Parameters& para)
{
  para.file_tag = std::string(para.enable_zbl ? "nep4_zbl" : "nep4") + (para.model_type == 3 ? "_temperature" : "");
  calculate_nep_model_parameters(para, para.model_type == 3 ? 1 : 0, 0, 1, 1);
}

void NEP::initialize_parameters(
  std::mt19937& rng,
  std::vector<float>& mu, std::vector<float>& sigma)
{
  NEP_Potential_Parameters& para = *model_parameters;
  initialize_nep_mu_and_sigma(para, rng, mu, sigma, 1);
}

void NEP::parameter_groups(std::vector<int>& groups)
{
  NEP_Potential_Parameters& para = *model_parameters;
  find_nep_type_of_variable(para, groups, 1, 1);
}

void NEP::read_model(float* parameters)
{
  read_nep_model("nep.txt", *model_parameters, parameters);
}

void NEP::write_restart(
  const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma)
{
  write_nep_restart(filename, *model_parameters, mu, sigma);
}

namespace {
class NEP_Reader : public NEP_Energy_Reader
{
public:
  NEP_Reader(NEP_Potential_Parameters& para, EnergyDifference& energy_difference,
    NEP_Batch_Data<NEP_Temperature_Data>& temperature_data)
    : NEP_Energy_Reader(para, energy_difference), model_parameters(para), temperature_data(temperature_data) {}

  void prepare_dataset(Dataset& dataset) override
  {
    NEP_Energy_Reader::prepare_dataset(dataset);
    if (model_parameters.model_type == 3) {
      std::vector<float> temperature_cpu(dataset.N);
      for (int n = 0; n < dataset.Nc; ++n) {
        float temperature = temperatures[dataset.structures[n].index_in_file];
        for (int na = 0; na < dataset.Na_cpu[n]; ++na) {
          temperature_cpu[dataset.Na_sum_cpu[n] + na] = temperature;
        }
      }
      auto& data = temperature_data.create(dataset);
      data.temperature.resize(dataset.N);
      data.temperature.copy_from_host(temperature_cpu.data());
    }
  }

  const char* energy_difference_conflict(const Structure& a, const Structure& b) const override
  {
    if (model_parameters.model_type == 3 && temperatures[a.index_in_file] != temperatures[b.index_in_file]) {
      return nullptr;
    }
    return NEP_Energy_Reader::energy_difference_conflict(a, b);
  }

private:
  NEP_Potential_Parameters& model_parameters;
  NEP_Batch_Data<NEP_Temperature_Data>& temperature_data;
  std::vector<float> temperatures;
  void begin_reading() override { temperatures.clear(); }
  void read_structure(const XYZ_Record& record, Structure& structure) override
  {
    NEP_Energy_Reader::read_structure(record, structure);
    const auto& tokens = record.tokens;
    const auto& xyz_filename = record.filename;
    int line_number = record.comment_line_number;
    bool has_temperature = false;
    float temperature = 0.0f;
    for (const auto& token : tokens) {
      const std::string temperature_string = "temperature=";
      if (token.substr(0, temperature_string.length()) == temperature_string) {
        has_temperature = true;
        temperature = get_double_from_token(
          token.substr(temperature_string.length(), token.length()),
          xyz_filename.c_str(),
          line_number);
      }
    }
    if (model_parameters.model_type == 3 && !has_temperature) {
      PRINT_INPUT_ERROR("'temperature' is missing in the second line of a frame.");
    }
    temperatures.push_back(temperature);
  }
};
} // namespace

NEP::NEP(const ParameterInput& input)
  : model_parameters(new NEP_Potential_Parameters)
{
  auto& para = *model_parameters;
  para.model_type = input.model_type;
  para.read(input);
  configure(para);
  para.finish_reading(input, para.model_type == 3 ? "temperature-dependent free energy" : "potential");
}

int NEP::prepare_data()
{
  NEP_Potential_Parameters& para = *model_parameters;
  NEP_Reader reader(para, energy_difference, temperature_data);
  training.reset(new NEP_Training(para, reader));
  initialize(para, training->N, training->device_count);
  if (para.prediction == 0) {
    energy_loss.initialize(para);
    training->open_loss_file(para,
      " rmse_energy_train rmse_force_train rmse_virial_train"
      " rmse_energy_test rmse_force_test rmse_virial_test", para.has_ediff_combinations);
  }
  return training->num_batches;
}

void NEP::prepare_training()
{
  NEP_Potential_Parameters& para = *model_parameters;
  int devices = training->device_count;
  std::vector<float> dummy_solution(para.number_of_variables * devices, para.initial_para);
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, dummy_solution.data(), training->train_set[batch],
      (para.fine_tune || para.import_q_scaler) ? false : true, devices);
  }
}

void NEP::evaluate(int batch, const float* population)
{
  NEP_Potential_Parameters& para = *model_parameters;
  int devices = training->device_count;
  for (int first = 0; first < para.population_size; first += devices) {
    int count = std::min(devices, para.population_size - first);
    find_force(para, population + first * para.number_of_variables,
      training->train_set[batch], false, count);
    for (int device = 0; device < count; ++device) {
      Dataset& data = training->train_set[batch][device];
      energy_loss.evaluate(para, data, energy_difference, batch, first + device, device);
    }
  }
}

void NEP::accumulate_fitness(float* fitness_total)
{
  NEP_Potential_Parameters& para = *model_parameters;
  energy_loss.accumulate_fitness(para, fitness_total);
  energy_loss.accumulate_energy_difference(para, fitness_total);
}

void NEP::print_training_header()
{
  print_nep_energy_header(model_parameters->has_ediff_combinations);
}

void NEP::report_error(const TrainingReport& report)
{
  NEP_Potential_Parameters& para = *model_parameters;
  find_force(para, report.parameters, training->train_set[report.batch], false, 1);
  Dataset& train_data = training->train_set[report.batch][0];
  NEP_Energy_Error train = energy_loss.get_error(
    para, train_data, energy_difference, true, report.batch);
  // Correct the last bias parameter in the energy network before evaluating the test set.
  report.parameters[para.number_of_variables_ann - 1] += train.energy_shift;
  NEP_Energy_Error test;
  if (training->has_test_set) {
    find_force(para, report.parameters, training->test_set, false, 1);
    test = energy_loss.get_error(para, training->test_set[0], energy_difference, false, 0);
  }
  training->write_model(para, report);
  write_nep_energy_error(stdout, report, train, test, para.has_ediff_combinations);
  write_nep_energy_error(training->fid_loss_out, report, train, test, para.has_ediff_combinations);
  if (training->has_test_set) {
    Dataset& data = training->test_set[0];
    copy_nep_energy_predictions(data);
    std::vector<std::pair<Dataset*, int>> structures;
    for (int nc = 0; nc < data.Nc; ++nc) {
      structures.emplace_back(&data, nc);
    }
    write_nep_energy_predictions("test", structures);
  }
}

void NEP::predict(float* parameters)
{
  NEP_Potential_Parameters& para = *model_parameters;
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, parameters, training->train_set[batch], false, 1);
    Dataset& data = training->train_set[batch][0];
    copy_nep_energy_predictions(data);
  }
  std::vector<std::pair<Dataset*, int>> structures;
  training->get_prediction_order(structures);
  write_nep_energy_predictions("train", structures);
}
