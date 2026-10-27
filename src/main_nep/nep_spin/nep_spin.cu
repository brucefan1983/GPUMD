/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "main_nep/common/nep_init.cuh"
#include "main_nep/common/nep_io.cuh"
#include "main_nep/common/nep_model.cuh"
#include "main_nep/common/nep_output.cuh"
#include "nep_spin.cuh"
#include "spin_io.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include <algorithm>
void NEP_Spin_Model::initialize(Spin_Parameters& para, int N, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    initialize_device(para, N, device_id);
  }
  initialize_compiled_kernel(para);
  initialize_spin(para, N, deviceCount);
}

void NEP_Spin_Model::initialize_model(const Spin_Parameters& para)
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
  paramb.num_c_radial = paramb.num_types * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#else
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#endif

  initialize_nep_zbl(para, zbl);
}

void NEP_Spin_Model::initialize_device(const Spin_Parameters& para, int N, int device_id)
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

void NEP_Spin_Model::initialize_compiled_kernel(const Spin_Parameters& para)
{
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    auto config = make_nep_compile_config(para);
    config.model_name = "Spin3";
    config.source_file = "main_nep/nep/nep_specialized.cu";
    compiled_kernel_.reset(new NEP_Compiled_Evaluation(config));
    if (!compiled_kernel_->is_valid())
      compiled_kernel_.reset();
    initialize_spin_compiled(para);
  }
}
static void __global__ find_max_min(
  const int N,
  const float* g_q,
  float* g_q_scaler,
  float* g_q_scaler_max,
  float* g_q_scaler_min,
  float* g_q_square_sum,
  unsigned long long* g_q_count)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  __shared__ float s_square_sum[1024];
  s_max[tid] = -1000000.0f; // a small number
  s_min[tid] = +1000000.0f; // a large number
  s_square_sum[tid] = 0.0f;
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      s_square_sum[tid] += q * q;
      if (q > s_max[tid]) {
        s_max[tid] = q;
      }
      if (q < s_min[tid]) {
        s_min[tid] = q;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) {
        s_max[tid] = s_max[tid + offset];
      }
      if (s_min[tid] > s_min[tid + offset]) {
        s_min[tid] = s_min[tid + offset];
      }
      s_square_sum[tid] += s_square_sum[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler_max[bid] = max(g_q_scaler_max[bid], s_max[0]);
    g_q_scaler_min[bid] = min(g_q_scaler_min[bid], s_min[0]);
    g_q_square_sum[bid] += s_square_sum[0];
    g_q_count[bid] += static_cast<unsigned long long>(N);
    g_q_scaler[bid] = 1.0f / (g_q_scaler_max[bid] - g_q_scaler_min[bid]);
  }
}

template <int max_dim>
static __global__ void apply_ann_one_layer_warp(
  const int N,
  const NEP_ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  constexpr int warp_size = 32;
  constexpr int values_per_lane = (max_dim + warp_size - 1) / warp_size;
  const int lane = threadIdx.x & (warp_size - 1);
  const int atom = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
  if (atom >= N)
    return;

  const int dim = annmb.dim;
  const int num_neurons = annmb.num_neurons1;
  const int type = g_type[atom];
  const float* w0 = annmb.wb[type];
  const float* b0 = w0 + num_neurons * dim;
  const float* w1 = b0 + num_neurons;
  float q[values_per_lane] = {};
  float Fp[values_per_lane] = {};
  for (int k = 0; k < values_per_lane; ++k) {
    const int d = lane + k * warp_size;
    if (d < dim) {
      q[k] = g_descriptors[atom + d * N] * g_q_scaler[d];
    }
  }

  float energy = 0.0f;
  for (int n = 0; n < num_neurons; ++n) {
    float w0_times_q = 0.0f;
    for (int k = 0; k < values_per_lane; ++k) {
      const int d = lane + k * warp_size;
      if (d < dim)
        w0_times_q += w0[n * dim + d] * q[k];
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      w0_times_q += __shfl_down_sync(0xffffffff, w0_times_q, offset);
    }
    float derivative_scale = 0.0f;
    if (lane == 0) {
      const float x1 = tanh(w0_times_q - b0[n]);
      derivative_scale = w1[n] * (1.0f - x1 * x1);
      energy += w1[n] * x1;
    }
    derivative_scale = __shfl_sync(0xffffffff, derivative_scale, 0);
    for (int k = 0; k < values_per_lane; ++k) {
      const int d = lane + k * warp_size;
      if (d < dim)
        Fp[k] += derivative_scale * w0[n * dim + d];
    }
  }

  if (lane == 0)
    g_pe[atom] = energy - annmb.b[0];
  for (int k = 0; k < values_per_lane; ++k) {
    const int d = lane + k * warp_size;
    if (d < dim)
      g_Fp[atom + d * N] = Fp[k] * g_q_scaler[d];
  }
}

Spin_Data& NEP_Spin_Model::magnetic(Dataset& dataset)
{
  if (!response_set.empty() && &dataset == &response_set[dataset.device_id])
    return *response_data[dataset.device_id];
  return magnetic_data.get(dataset);
}
void NEP_Spin_Model::find_force(
  Spin_Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{
  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    bind_nep_parameters(paramb.num_types, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    neighbor[device_id].prepare(para, dataset[device_id], device_id);

    compute_nep_descriptors(
      paramb,
      annmb[device_id].c,
      dataset[device_id],
      neighbor[device_id],
      nep_data[device_id],
      compiled_kernel_.get());

    find_additional_descriptors(para, dataset[device_id], device_id);
    if (para.prediction == 1 && para.output_descriptor >= 1) {
      output_nep_descriptors(
        dataset[device_id],
        annmb[device_id].dim,
        para.output_descriptor,
        para.q_scaler_cpu,
        nep_data[device_id].descriptors);
    }

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        dataset[device_id].N,
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        para.q_scaler_max[device_id].data(),
        para.q_scaler_min[device_id].data(),
        para.q_scaler_square_sum[device_id].data(),
        para.q_scaler_count[device_id].data());
      GPU_CHECK_KERNEL
    }

    zero_nep_outputs(dataset[device_id]);

    initialize_additional_outputs(dataset[device_id], device_id);
    auto& data = dataset[device_id];
    auto& work = nep_data[device_id];
    if (compiled_kernel_) {
      compiled_kernel_->launch_ann(
        data.N,
        data.type.data(),
        work.descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        work.parameters.data(),
        data.energy.data(),
        work.Fp.data());
    } else {
      apply_ann_one_layer_warp<NEP_Spin_Model_File::MAX_DIM_SPIN><<<(data.N + 3) / 4, 128>>>(
        data.N,
        annmb[device_id],
        data.type.data(),
        work.descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        data.energy.data(),
        work.Fp.data());
      GPU_CHECK_KERNEL
    }
    compute_nep_forces(
      paramb,
      annmb[device_id].c,
      dataset[device_id],
      neighbor[device_id],
      nep_data[device_id],
      compiled_kernel_.get());

    if (zbl.enabled) {
      add_nep_zbl(zbl, dataset[device_id], neighbor[device_id]);
    }
    find_additional_force(para, dataset[device_id], device_id);
  }
}

NEP_Spin_Model::NEP_Spin_Model(const ParameterInput& input)
  : model_parameters(new Spin_Parameters(input))
{
}

int NEP_Spin_Model::prepare_data()
{
  auto& para = *model_parameters;
  Spin_Reader reader(para, energy_difference, magnetic_data);
  training.reset(new NEP_Training(para, reader));
  int capacity = training->N;
  if (para.lambda_spin_response > 0 && !para.prediction) {
    std::vector<Structure> frames;
    std::vector<Spin_Data::Labels> labels;
    reader.read_response(frames, labels);
    if (frames.empty())
      PRINT_INPUT_ERROR("response.xyz must not be empty.");
    response_set.resize(training->device_count);
    response_data.resize(training->device_count);
    for (int device = 0; device < training->device_count; ++device) {
      CHECK(gpuSetDevice(device));
      response_set[device].construct(para, frames, 0, frames.size(), device, -1);
      response_data[device].reset(new Spin_Data);
      response_data[device]->construct(para, response_set[device], labels);
      capacity = std::max(capacity, response_set[device].N);
    }
  }
  initialize(para, capacity, training->device_count);
  if (!para.prediction) {
    energy_loss.initialize(para);
    const int size = (para.num_types + 1) * para.population_size;
    fitness_mforce.assign(size, 0.0f);
    fitness_tau.assign(size, 0.0f);
    fitness_response.assign(size, 0.0f);
    std::string columns =
      " rmse_energy_train rmse_force_train rmse_virial_train rmse_mforce_train rmse_tau_train"
      " rmse_energy_test rmse_force_test rmse_virial_test rmse_mforce_test rmse_tau_test";
    if (para.split_mforce_loss) {
      columns += " rmse_m_parallel_train rmse_m_perp_train rmse_m_parallel_test rmse_m_perp_test";
    }
    training->open_loss_file(para, columns.c_str(), para.has_ediff_combinations);
  }
  return training->num_batches;
}

void NEP_Spin_Model::prepare_training()
{
  auto& para = *model_parameters;
  if (para.prediction || para.spin_restart || para.import_q_scaler)
    return;
  // A fixed reference descriptor model makes scaling independent of population and seed.
  std::mt19937 reference_rng(12345678);
  std::vector<float> mu(para.number_of_variables), sigma(para.number_of_variables);
  initialize_search(reference_rng, mu, sigma);
  // Scale radial coefficients only: constant/scaled projection matrices would
  // change the relative scaling of the different magnetic contraction orders.
  const int radial_end = para.number_of_variables - para.number_of_variables_spin_projection;
  for (int n = para.number_of_variables_ann; n < radial_end; ++n)
    mu[n] *= para.initial_para;
  printf("Spin scaler reference: initial_para = %g\n", para.initial_para);
  std::vector<float> parameters;
  for (int device = 0; device < training->device_count; ++device)
    parameters.insert(parameters.end(), mu.begin(), mu.end());
  for (int batch = 0; batch < training->num_batches; ++batch)
    find_force(para, parameters.data(), training->train_set[batch], true, training->device_count);
  finalize_q_scaler();
}

void NEP_Spin_Model::initialize_parameters(
  std::mt19937& rng, std::vector<float>& mu, std::vector<float>& sigma)
{
  auto& para = *model_parameters;
  if (!read_nep_restart("nep.restart", para, mu, sigma, &para.energy_reference))
    initialize_search(rng, mu, sigma);
}
void NEP_Spin_Model::parameter_groups(std::vector<int>& groups)
{
  auto& para = *model_parameters;
  find_nep_type_of_variable(para, groups, 1, 1);
  const int offset = para.number_of_variables - para.number_of_variables_descriptor_spin -
                     para.number_of_variables_spin_projection;
  for (int channel = 0; channel < para.spin_compress; ++channel) {
    for (int basis = 0; basis <= para.spin_basis_size[0]; ++basis) {
      for (int t1 = 0; t1 < para.num_types; ++t1) {
        for (int t2 = 0; t2 < para.num_types; ++t2) {
          const int pair = t1 * para.num_types + t2;
          const int coefficient =
            (channel * (para.spin_basis_size[0] + 1) + basis) * para.num_types * para.num_types +
            pair;
          groups[offset + coefficient] = t1;
        }
      }
    }
  }
}
void NEP_Spin_Model::read_model(float* parameters)
{
  read_spin_model(*model_parameters, parameters);
}
void NEP_Spin_Model::write_restart(
  const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma)
{
  write_nep_restart(filename, *model_parameters, mu, sigma);
}
float NEP_Spin_Model::response_loss(int device)
{
  if (response_set.empty())
    return 0.0f;
  ResponseLoss response;
  response.append(device, response_set[device], *response_data[device]);
  return response.value();
}
void NEP_Spin_Model::evaluate(int batch, const float* population)
{
  auto& para = *model_parameters;
  const int devices = training->device_count;
  for (int first = 0; first < para.population_size; first += devices) {
    const int count = std::min(devices, para.population_size - first);
    const float* parameters = population + first * para.number_of_variables;
    find_force(para, parameters, training->train_set[batch], false, count);
    for (int device = 0; device < count; ++device) {
      Dataset& data = training->train_set[batch][device];
      energy_loss.evaluate(para, data, energy_difference, batch, first + device, device);
      auto m = para.split_mforce_loss
                 ? magnetic(data).get_rmse_mforce_parallel(para, data, true, device)
                 : magnetic(data).get_rmse_mforce(para, data, true, device);
      std::vector<float> perpendicular;
      if (para.split_mforce_loss)
        perpendicular = magnetic(data).get_rmse_mforce_perp(para, data, true, device);
      auto t = magnetic(data).get_rmse_tau(para, data, true, device);
      for (int type = 0; type <= para.num_types; ++type) {
        const int index = first + device + type * para.population_size;
        fitness_mforce[index] = para.split_mforce_loss ? para.lambda_m_parallel * m[type] +
                                                           para.lambda_m_perp * perpendicular[type]
                                                       : para.lambda_m * m[type];
        fitness_tau[index] = para.lambda_tau * t[type];
      }
    }
    if (!response_set.empty()) {
      find_force(para, parameters, response_set, false, count);
      for (int device = 0; device < count; ++device) {
        const float loss = para.lambda_spin_response * response_loss(device);
        for (int type = 0; type <= para.num_types; ++type)
          fitness_response[first + device + type * para.population_size] = loss;
      }
    }
  }
}
void NEP_Spin_Model::accumulate_fitness(float* total)
{
  auto& para = *model_parameters;
  energy_loss.accumulate_fitness(para, total);
  for (int i = 0; i < fitness_mforce.size(); ++i)
    total[i] = total[i] + fitness_mforce[i] + fitness_tau[i] + fitness_response[i];
  energy_loss.accumulate_energy_difference(para, total);
}
void NEP_Spin_Model::print_training_header()
{
  printf(
    "%-8s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s%-11s",
    "Step",
    "Total",
    "L1Reg",
    "L2Reg",
    "E-Train",
    "F-Train",
    "V-Train",
    "M-Train",
    "T-Train",
    "E-Test",
    "F-Test",
    "V-Test",
    "M-Test",
    "T-Test");
  if (model_parameters->split_mforce_loss)
    printf(" M-parallel-Train M-perp-Train M-parallel-Test M-perp-Test");
  printf("\n");
}
void NEP_Spin_Model::write_magnetic_predictions(
  const char* prefix, const std::vector<std::pair<Dataset*, int>>& structures)
{
  const std::string filename = std::string("mforce_") + prefix + ".out";
  FILE* file = my_fopen(filename.c_str(), "w");
  for (const auto& entry : structures) {
    Dataset& data = *entry.first;
    auto& spin = magnetic(data);
    const int nc = entry.second;
    if (!model_parameters->prediction && !spin.labels[nc].has_mforce)
      continue;
    for (int a = 0; a < data.Na_cpu[nc]; ++a) {
      const int atom = data.Na_sum_cpu[nc] + a;
      fprintf(
        file,
        "%g %g %g %g %g %g\n",
        spin.mforce_cpu[atom],
        spin.mforce_cpu[data.N + atom],
        spin.mforce_cpu[2 * data.N + atom],
        spin.mforce_ref_cpu[atom],
        spin.mforce_ref_cpu[data.N + atom],
        spin.mforce_ref_cpu[2 * data.N + atom]);
    }
  }
  fclose(file);
}
void NEP_Spin_Model::report_error(const TrainingReport& report)
{
  auto& para = *model_parameters;
  find_force(para, report.parameters, training->train_set[report.batch], false, 1);
  Dataset& data = training->train_set[report.batch][0];
  auto train = energy_loss.get_error(para, data, energy_difference, true, report.batch);
  const float mt = magnetic(data).get_rmse_mforce(para, data, false, 0).back();
  const float tt = magnetic(data).get_rmse_tau(para, data, false, 0).back();
  float mp_train = 0, mt_train = 0, mp_test = 0, mt_test = 0;
  if (para.split_mforce_loss) {
    mp_train = magnetic(data).get_rmse_mforce_parallel(para, data, false, 0).back();
    mt_train = magnetic(data).get_rmse_mforce_perp(para, data, false, 0).back();
  }
  report.parameters[para.number_of_variables_ann - 1] += train.energy_shift;
  NEP_Energy_Error test;
  float mv = 0, tv = 0;
  if (training->has_test_set) {
    find_force(para, report.parameters, training->test_set, false, 1);
    Dataset& data = training->test_set[0];
    test = energy_loss.get_error(para, data, energy_difference, false, 0);
    mv = magnetic(data).get_rmse_mforce(para, data, false, 0).back();
    tv = magnetic(data).get_rmse_tau(para, data, false, 0).back();
    if (para.split_mforce_loss) {
      mp_test = magnetic(data).get_rmse_mforce_parallel(para, data, false, 0).back();
      mt_test = magnetic(data).get_rmse_mforce_perp(para, data, false, 0).back();
    }
  }
  int max_radial = training->max_NN_radial;
  for (auto& batch : training->train_set)
    max_radial = std::max(max_radial, magnetic(batch[0]).max_NN_spin);
  if (training->has_test_set)
    max_radial = std::max(max_radial, magnetic(training->test_set[0]).max_NN_spin);
  int max_angular = training->max_NN_angular;
  if (!response_set.empty()) {
    max_radial =
      std::max(max_radial, std::max(response_set[0].max_NN_radial, response_data[0]->max_NN_spin));
    max_angular = std::max(max_angular, response_set[0].max_NN_angular);
  }
  for (const auto& filename : {std::string("nep.txt"), report.checkpoint}) {
    if (filename.empty())
      continue;
    FILE* file = my_fopen(filename.c_str(), "w");
    write_spin_model(file, report.parameters, max_radial, max_angular);
    fclose(file);
  }
  for (FILE* file : {stdout, training->fid_loss_out}) {
    fprintf(
      file,
      "%d %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f",
      report.generation + 1,
      report.total,
      report.l1,
      report.l2,
      train.energy,
      train.force,
      train.virial,
      mt,
      tt,
      test.energy,
      test.force,
      test.virial,
      mv,
      tv);
    if (para.split_mforce_loss)
      fprintf(file, " %.8e %.8e %.8e %.8e", mp_train, mt_train, mp_test, mt_test);
    if (para.has_ediff_combinations)
      fprintf(file, " %.5f %.5f", train.ediff, test.ediff);
    fprintf(file, "\n");
    fflush(file);
  }
  if (training->has_test_set) {
    Dataset& data = training->test_set[0];
    copy_nep_energy_predictions(data, para.energy_reference);
    magnetic(data).mforce.copy_to_host(magnetic(data).mforce_cpu.data());
    std::vector<std::pair<Dataset*, int>> structures;
    for (int nc = 0; nc < data.Nc; ++nc)
      structures.emplace_back(&data, nc);
    write_nep_energy_predictions("test", structures);
    write_magnetic_predictions("test", structures);
  }
}
void NEP_Spin_Model::predict(float* parameters)
{
  auto& para = *model_parameters;
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, parameters, training->train_set[batch], false, 1);
    Dataset& data = training->train_set[batch][0];
    copy_nep_energy_predictions(data, para.energy_reference);
    magnetic(data).mforce.copy_to_host(magnetic(data).mforce_cpu.data());
  }
  std::vector<std::pair<Dataset*, int>> structures;
  training->get_prediction_order(structures);
  write_nep_energy_predictions("train", structures);
  write_magnetic_predictions("train", structures);
}
