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
#include <memory>
#include <random>
#include <cstdio>
#include <vector>
class Dataset;
class Parameters;

class Potential
{
public:
  virtual ~Potential() = default;
  virtual void initialize_parameters(
    const Parameters& para, std::mt19937& rng,
    std::vector<float>& mu, std::vector<float>& sigma) = 0;
  virtual void parameter_groups(const Parameters& para, std::vector<int>& groups) = 0;
  virtual void read_model(Parameters& para, float* parameters) = 0;
  virtual void write_model(
    FILE* file, Parameters& para, const float* parameters, int max_radial, int max_angular) = 0;
  virtual void write_restart(
    const char* filename, const Parameters& para,
    const std::vector<float>& mu, const std::vector<float>& sigma) = 0;
  virtual void configure(Parameters& para) = 0;
  virtual void initialize(Parameters& para, int N, int Nc, int deviceCount) = 0;
  virtual void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int DeviceCount) = 0;
};

std::unique_ptr<Potential> create_potential(Parameters& para);
