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
  int compute = 0;     //1 for heat flow algorithm, 2 for color conductivity algorithm
  int output_interval; // average the data every so many time steps

  // the driving "force" vector (in units of 1/A)
  double fe_x = 0.0;
  double fe_y = 0.0;
  double fe_z = 0.0;
  double fe = 0.0; // magnitude of the driving "force" vector

  std::vector<double> cpu_mass_type; // 2 atom types' mass
  GPU_Vector<double> mass_type;
  std::vector<double>cpu_fraction; // 2 atom types' factor, cpu_fraction[0]=N2/N, cpu_fraction[1]=-N1/N
  GPU_Vector<double> fraction;
  double scale; // (x2/m1+x1/m2)^-1 for compute=1, (x2/m1+x1/m2)^-2 for compute=2

  GPU_Vector<double> heat_all;
  GPU_Vector<double> diffusive_all;

  void preprocess(
    const std::vector<double>& mass,
    const std::vector<int>& type,
    const std::vector<int>& type_size);

  void process(
    int step,
    const char* input_dir,
    const double temperature,
    const double volume,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential,
    GPU_Vector<double>& heat_per_atom);

  void postprocess();

  void parse(char** param, int num_param);
};
