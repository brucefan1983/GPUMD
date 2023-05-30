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
#include <vector>

class Atom;

class Electron_Stop
{
public:
  bool do_electron_stop = false;
  void parse(const char** param, int num_param, const int num_atoms, const int num_types);
  void compute(const double time_step, Atom& atom);
  void finalize();

private:
  int num_points = 0;
  double energy_min;
  double energy_max;
  double energy_interval;
  std::vector<double> stopping_power_cpu;
  GPU_Vector<double> stopping_power_gpu;
  GPU_Vector<double> stopping_force;
};
