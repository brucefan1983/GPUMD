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

#ifdef USE_PLUMED

#pragma once
#include <plumed/wrapper/Plumed.h>
#include "force/potential.cuh"
#include <stdio.h>
#include <vector>

class PLUMED
{
public:
  int step = 0;
  int interval = 1;
  int use_plumed = 0;
  void parse(char **param, int num_param);
  void preprocess(const std::vector<double>& cpu_mass);
  void init(const double ts, const double T);
  void process(
    Box& box,
    GPU_Vector<double>& thermo,
    GPU_Vector<double>& position,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void postprocess(void);
protected:
  int n_atom;
  int restart;
  int stop_flag;
  double time_step;
  double bias_energy;
  double total_energy;
  char input_file[1024];
  char output_file[1024];
  GPU_Vector<double>  gpu_v_vector; // Total Virial (GPU)
  GPU_Vector<double>  gpu_v_factor; // Scaling factor of the virial (GPU)
  std::vector<double> cpu_m_vector; // Mass
  std::vector<double> cpu_b_vector; // Box
  std::vector<double> cpu_f_vector; // Forces
  std::vector<double> cpu_q_vector; // Positions
  std::vector<double> cpu_v_vector; // Total Virial (CPU)
  std::vector<double> cpu_v_factor; // Scaling factor of the virial (CPU)
  plumed plumed_main;
};

#endif