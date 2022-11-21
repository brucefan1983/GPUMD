/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
#include "utilities/gpu_vector.cuh"

class HNEMDEC
{
public:
  int compute = -1; // 0 for heat flow algorithm, i(0<i<number_of_types+1) means producing (i-1)th
                    // element's mass flow as dissipation flux in color conductivity algorithm
  int output_interval; // average the data every so many time steps

  // the driving force vector
  double fe_x = 0.0;
  double fe_y = 0.0;
  double fe_z = 0.0;
  double fe = 0.0; // magnitude of the driving force vector

  int number_of_types;
  int NUM_OF_DIFFUSION_COMPONENTS;
  std::vector<double> cpu_mass_type; // atom types' mass
  GPU_Vector<double> mass_type;
  double FACTOR;

  GPU_Vector<double> heat_all;
  GPU_Vector<double> diffusion_all;

  void preprocess(
    const std::vector<double>& mass,
    const std::vector<int>& type,
    const std::vector<int>& type_size);

  void process(
    int step,
    const double temperature,
    const double volume,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential,
    GPU_Vector<double>& heat_per_atom);

  void postprocess();

  void parse(const char** param, int num_param);
};
