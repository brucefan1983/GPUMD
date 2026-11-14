/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#include "nep_spin.cuh"
#include <algorithm>
#include <cmath>
void NEP_Spin_Model::initialize_search(
  std::mt19937& rng, std::vector<float>& mu, std::vector<float>& sigma) const
{
  const auto& para = *model_parameters;
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float input_scale = 1.0f / std::sqrt(float(para.dim + para.num_neurons1));
  const float output_scale = 1.0f / std::sqrt(float(para.num_neurons1 + 1));
  const float descriptor_scale = 0.1f;
  const float spin_noise = 0.01f;

  std::fill(mu.begin(), mu.end(), 0.0f);
  std::fill(sigma.begin(), sigma.end(), para.sigma0 * descriptor_scale);
  for (int type = 0; type < para.num_types; ++type) {
    const int ann_offset = type * para.number_of_variables_ann_1;
    for (int neuron = 0; neuron < para.num_neurons1; ++neuron) {
      for (int descriptor = 0; descriptor < para.dim; ++descriptor) {
        const int index = ann_offset + neuron * para.dim + descriptor;
        mu[index] = normal(rng) * input_scale;
        sigma[index] = para.sigma0 * input_scale;
      }
    }
    const int bias_offset = ann_offset + para.num_neurons1 * para.dim;
    const int output_offset = bias_offset + para.num_neurons1;
    for (int neuron = 0; neuron < para.num_neurons1; ++neuron) {
      mu[bias_offset + neuron] = 0.0f;
      sigma[bias_offset + neuron] = para.sigma0 * input_scale;
      mu[output_offset + neuron] = normal(rng) * output_scale;
      sigma[output_offset + neuron] = para.sigma0 * output_scale;
    }
  }
  const int descriptor_offset = para.number_of_variables_ann;
  const int type_pairs = para.num_types * para.num_types;
  const int radial_basis_count = para.basis_size_radial + 1;
  const int radial_channel_count = para.n_max_radial + 1;
  const int radial_count = type_pairs * radial_channel_count * radial_basis_count;
  for (int channel = 0; channel < radial_channel_count; ++channel) {
    for (int basis = 0; basis < radial_basis_count; ++basis) {
      for (int pair = 0; pair < type_pairs; ++pair) {
        const int index = descriptor_offset + pair * radial_channel_count * radial_basis_count +
                          channel * radial_basis_count + basis;
        mu[index] = normal(rng) * spin_noise;
        if (basis == channel % radial_basis_count) {
          mu[index] += 1.0f;
        }
      }
    }
  }
  const int angular_basis_count = para.basis_size_angular + 1;
  const int angular_channel_count = para.n_max_angular + 1;
  const int angular_offset = descriptor_offset + radial_count;
  for (int channel = 0; channel < angular_channel_count; ++channel) {
    for (int basis = 0; basis < angular_basis_count; ++basis) {
      for (int pair = 0; pair < type_pairs; ++pair) {
        const int index = angular_offset + pair * angular_channel_count * angular_basis_count +
                          channel * angular_basis_count + basis;
        mu[index] = normal(rng) * spin_noise;
        if (basis == channel % angular_basis_count) {
          mu[index] += 1.0f;
        }
      }
    }
  }

  const int structural_count =
    radial_count + type_pairs * angular_channel_count * angular_basis_count;

  const int spin_offset = descriptor_offset + structural_count;

  const int radial_mix_end = spin_offset + angular_channel_count * para.spin_compress;
  for (int index = spin_offset; index < para.number_of_variables; ++index) {
    const float scale = index < radial_mix_end ? 1.0f / std::sqrt(float(angular_channel_count)) : 0.2f;
    mu[index] = normal(rng) * scale;
    sigma[index] = para.sigma0 * scale;
  }
}
