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
#include "structure.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>
class Parameters;

class Dataset
{
public:
  int Nc;                       // number of configurations
  int N;                        // total number of atoms (sum of Na[])
  int max_Na;                   // number of atoms in the largest configuration
  int num_types;                // number of atom types
  int max_NN_radial;            // radial neighbor list size
  int max_NN_angular;           // angular neighbor list size
  GPU_Vector<int> Na;           // number of atoms in each configuration
  GPU_Vector<int> Na_sum;       // prefix sum of Na
  GPU_Vector<int> type;         // atom type (0, 1, 2, 3, ...)
  GPU_Vector<float> r;          // position
  GPU_Vector<float> force;      // force
  GPU_Vector<float> pe;         // potential energy
  GPU_Vector<float> virial;     // per-atom virial tensor
  GPU_Vector<float> h;          // box and inverse box
  GPU_Vector<float> pe_ref;     // reference energy for the whole box
  GPU_Vector<float> virial_ref; // reference virial for the whole box
  GPU_Vector<float> force_ref;  // reference force
  std::vector<float> error_cpu; // error in energy, virial, or force
  GPU_Vector<float> error_gpu;  // error in energy, virial, or force
  GPU_Vector<int> NN_radial;    // radial neighbor number
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor number
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;

  std::vector<Structure> structures;

  void construct(char*, Parameters& para, std::vector<Structure>& structures);
  float get_rmse_force();
  float get_rmse_energy();
  float get_rmse_virial();

private:
  void copy_structures(std::vector<Structure>& structures_input);
  void find_Na();
  void initialize_gpu_data(Parameters& para);
  void check_types(Parameters& para);
  void find_neighbor(Parameters& para);
};
