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

#pragma once

#include <vector>

// Potential files store descriptor parameters as [basis][type_pair].
// GPU force kernels read one type pair at a time, so use [type_pair][basis].
inline std::vector<float> get_descriptor_parameters_type_pair(
  const std::vector<float>& parameters,
  const int num_para_ann,
  const int num_types,
  const int n_max_radial,
  const int n_max_angular,
  const int basis_size_radial,
  const int basis_size_angular)
{
  const int num_types_sq = num_types * num_types;
  const int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  const int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  const int num_c_radial = num_types_sq * num_radial_basis;
  std::vector<float> descriptor_parameters(
    num_c_radial + num_types_sq * num_angular_basis);

  for (int type_pair = 0; type_pair < num_types_sq; ++type_pair) {
    for (int basis = 0; basis < num_radial_basis; ++basis) {
      descriptor_parameters[type_pair * num_radial_basis + basis] =
        parameters[num_para_ann + basis * num_types_sq + type_pair];
    }
    for (int basis = 0; basis < num_angular_basis; ++basis) {
      descriptor_parameters[num_c_radial + type_pair * num_angular_basis + basis] =
        parameters[num_para_ann + num_c_radial + basis * num_types_sq + type_pair];
    }
  }

  return descriptor_parameters;
}
// Training keeps descriptor parameters as [channel][basis] internally while
// nep.txt and nep.restart keep the historical [basis][channel] order.
inline void descriptor_parameters_to_channel_major(
  float* parameters,
  const int descriptor_offset,
  const int num_channels,
  const int n_max_radial,
  const int n_max_angular,
  const int basis_size_radial,
  const int basis_size_angular)
{
  const int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  const int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  const int num_c_radial = num_channels * num_radial_basis;
  const int num_descriptor_parameters = num_c_radial + num_channels * num_angular_basis;
  std::vector<float> descriptor_parameters(num_descriptor_parameters);

  for (int channel = 0; channel < num_channels; ++channel) {
    for (int basis = 0; basis < num_radial_basis; ++basis) {
      descriptor_parameters[channel * num_radial_basis + basis] =
        parameters[descriptor_offset + basis * num_channels + channel];
    }
    for (int basis = 0; basis < num_angular_basis; ++basis) {
      descriptor_parameters[num_c_radial + channel * num_angular_basis + basis] =
        parameters[descriptor_offset + num_c_radial + basis * num_channels + channel];
    }
  }

  for (int n = 0; n < num_descriptor_parameters; ++n) {
    parameters[descriptor_offset + n] = descriptor_parameters[n];
  }
}

inline void descriptor_parameters_to_basis_major(
  float* parameters,
  const int descriptor_offset,
  const int num_channels,
  const int n_max_radial,
  const int n_max_angular,
  const int basis_size_radial,
  const int basis_size_angular)
{
  const int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  const int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  const int num_c_radial = num_channels * num_radial_basis;
  const int num_descriptor_parameters = num_c_radial + num_channels * num_angular_basis;
  std::vector<float> descriptor_parameters(num_descriptor_parameters);

  for (int channel = 0; channel < num_channels; ++channel) {
    for (int basis = 0; basis < num_radial_basis; ++basis) {
      descriptor_parameters[basis * num_channels + channel] =
        parameters[descriptor_offset + channel * num_radial_basis + basis];
    }
    for (int basis = 0; basis < num_angular_basis; ++basis) {
      descriptor_parameters[num_c_radial + basis * num_channels + channel] =
        parameters[descriptor_offset + num_c_radial + channel * num_angular_basis + basis];
    }
  }

  for (int n = 0; n < num_descriptor_parameters; ++n) {
    parameters[descriptor_offset + n] = descriptor_parameters[n];
  }
}
