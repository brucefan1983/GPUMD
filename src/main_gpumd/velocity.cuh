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

class Velocity
{
public:
  bool do_velocity_correction;
  int velocity_correction_interval;

  void initialize(
    const bool has_velocity_in_xyz,
    const double initial_temperature,
    const std::vector<double>& cpu_mass,
    const std::vector<double>& cpu_position_per_atom,
    std::vector<double>& cpu_velocity_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void correct_velocity(
    const int step,
    const double temperature,
    const std::vector<double>& cpu_mass,
    const GPU_Vector<double>& position_per_atom,
    std::vector<double>& cpu_position_per_atom,
    std::vector<double>& cpu_velocity_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void finalize();

private:
  void correct_velocity(
    const double initial_temperature,
    const std::vector<double>& cpu_mass,
    const std::vector<double>& cpu_position_per_atom,
    std::vector<double>& cpu_velocity_per_atom);

  void scale(
    const double initial_temperature,
    const std::vector<double>& cpu_mass,
    double* cpu_vx,
    double* cpu_vy,
    double* cpu_vz);
};
