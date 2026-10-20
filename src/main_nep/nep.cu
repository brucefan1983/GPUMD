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
#include "nep_model.cuh"
#include "nep_init.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"

void NEP::initialize(Parameters& para, int N, int Nc, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    initialize_device(para, N, device_id);
  }
  initialize_compiled_kernel(para);
}

void NEP::initialize_model(const Parameters& para)
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

void NEP::initialize_device(const Parameters& para, int N, int device_id)
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

void NEP::initialize_compiled_kernel(const Parameters& para)
{
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    compiled_kernel_.reset(new NEP_Compile(
      make_nep_compile_config(para, NEP_Compile_Mode::NEP)));
    if (!compiled_kernel_->is_valid()) {
      compiled_kernel_.reset();
    }
  }
}

void NEP::find_force(
  Parameters& para,
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
      annmb[device_id], para.model_type == 3, dataset[device_id], nep_data[device_id],
      para.q_scaler_gpu[device_id].data(), compiled_kernel_.get());

    compute_nep_forces(
      paramb, annmb[device_id].c, dataset[device_id], neighbor[device_id],
      nep_data[device_id], compiled_kernel_.get());

    if (zbl.enabled) {
      add_nep_zbl(zbl, dataset[device_id], neighbor[device_id]);
    }
  }
}

void NEP::configure(Parameters& para)
{
  calculate_nep_model_parameters(para, para.model_type == 3 ? 1 : 0, 0, 1, 1);
}

void NEP::initialize_parameters(
  const Parameters& para, std::mt19937& rng,
  std::vector<float>& mu, std::vector<float>& sigma)
{
  initialize_nep_mu_and_sigma(para, rng, mu, sigma, 1);
}

void NEP::parameter_groups(const Parameters& para, std::vector<int>& groups)
{
  find_nep_type_of_variable(para, groups, 1, 1);
}
