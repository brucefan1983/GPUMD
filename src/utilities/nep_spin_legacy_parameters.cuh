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

#pragma once

#include <vector>

// Convert between legacy Spin3 [basis][channel] file order and the
// [channel][basis] order used by training and the new block format.
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
