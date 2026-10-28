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
  const int basis_count = para.spin_basis_size[0] + 1;
  // Least-squares coefficients of x^c f_c(x) in the fixed B8 magnetic
  // Chebyshev basis, sampled exactly as the Spin3 TorchNEP initializer.
  static constexpr float spin3_radial_frame[9][9] = {
    {1.0000000000e+00f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {7.2002207902e-01f,
     -8.3611353484e-01f,
     1.5734662283e-01f,
     -6.0533065445e-02f,
     2.8744838362e-02f,
     -1.4475653098e-02f,
     7.3741983730e-03f,
     -3.2703655593e-03f,
     1.3101771678e-03f},
    {4.4004415805e-01f,
     -6.7222706969e-01f,
     3.1469324567e-01f,
     -1.2106613089e-01f,
     5.7489676724e-02f,
     -2.8951306196e-02f,
     1.4748396746e-02f,
     -6.5407311186e-03f,
     2.6203543356e-03f},
    {3.1140768621e-01f,
     -5.2723264595e-01f,
     3.2681622705e-01f,
     -1.6571616135e-01f,
     8.2091070549e-02f,
     -4.1985522505e-02f,
     2.1558021488e-02f,
     -9.6011245317e-03f,
     3.8591478579e-03f},
    {2.3545411266e-01f,
     -4.2002230503e-01f,
     2.9849192552e-01f,
     -1.7860012185e-01f,
     9.8405575301e-02f,
     -5.2136865236e-02f,
     2.7238498970e-02f,
     -1.2241573652e-02f,
     4.9551740892e-03f},
    {1.8445053632e-01f,
     -3.3955276527e-01f,
     2.6111274825e-01f,
     -1.7321025355e-01f,
     1.0453147344e-01f,
     -5.8397411709e-02f,
     3.1342038309e-02f,
     -1.4286182365e-02f,
     5.8465512142e-03f},
    {1.4751535571e-01f,
     -2.7741009878e-01f,
     2.2430319966e-01f,
     -1.5928934988e-01f,
     1.0305050145e-01f,
     -6.0626267462e-02f,
     3.3654414123e-02f,
     -1.5627206618e-02f,
     6.4904010783e-03f},
    {1.1949382843e-01f,
     -2.2829289243e-01f,
     1.9098724002e-01f,
     -1.4218308534e-01f,
     9.6941146019e-02f,
     -5.9613663992e-02f,
     3.4255504783e-02f,
     -1.6247499316e-02f,
     6.8698231435e-03f},
    {9.7608450047e-02f,
     -1.8883237673e-01f,
     1.6185826384e-01f,
     -1.2464644955e-01f,
     8.8460078738e-02f,
     -5.6406223768e-02f,
     3.3467341677e-02f,
     -1.6219692953e-02f,
     6.9958126950e-03f},
  };
  for (int channel = 0; channel < para.spin_compress; ++channel) {
    for (int basis = 0; basis < basis_count; ++basis) {
      for (int pair = 0; pair < type_pairs; ++pair) {
        const int index = spin_offset + (channel * basis_count + basis) * type_pairs + pair;
        mu[index] = spin3_radial_frame[channel][basis];
        sigma[index] = para.sigma0 * descriptor_scale;
      }
    }
  }

  const int projection_offset = spin_offset + para.number_of_variables_descriptor_spin;
  const int channels = para.spin_compress;
  for (int leg = 0; leg < 4; ++leg) {
    for (int row = 0; row < channels; ++row) {
      for (int source = 0; source < channels; ++source) {
        const int index = projection_offset + (leg * channels + row) * channels + source;
        mu[index] = normal(rng) * spin_noise;
        if (source == (row + leg) % channels) {
          mu[index] += 1.0f;
        }
        sigma[index] = para.sigma0 * descriptor_scale;
      }
    }
  }
}
