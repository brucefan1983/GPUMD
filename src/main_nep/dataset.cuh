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
class Parameters;

class Dataset
{
public:
  int Nc;                          // number of configurations
  int N;                           // total number of atoms (sum of Na[])
  int max_Na;                      // number of atoms in the largest configuration
  int num_types;                   // number of atom types
  int max_NN_radial;               // radial neighbor list size
  int max_NN_angular;              // angular neighbor list size
  GPU_Vector<int> Na;              // number of atoms in each configuration
  GPU_Vector<int> Na_sum;          // prefix sum of Na
  std::vector<int> has_virial;     // 1 if has virial for a configuration, 0 otherwise
  GPU_Vector<float> atomic_number; // atomic number (number of protons)
  GPU_Vector<float> r;             // position
  GPU_Vector<float> force;         // force
  GPU_Vector<float> pe;            // potential energy
  GPU_Vector<float> virial;        // per-atom virial tensor
  GPU_Vector<float> h;             // box and inverse box
  GPU_Vector<float> pe_ref;        // reference energy for the whole box
  GPU_Vector<float> virial_ref;    // reference virial for the whole box
  GPU_Vector<float> force_ref;     // reference force
  std::vector<float> error_cpu;    // error in energy, virial, or force
  GPU_Vector<float> error_gpu;     // error in energy, virial, or force
  GPU_Vector<int> NN_radial;       // radial neighbor number
  GPU_Vector<int> NL_radial;       // radial neighbor list
  GPU_Vector<int> NN_angular;      // angular neighbor number
  GPU_Vector<int> NL_angular;      // angular neighbor list

  struct Structure {
    int num_atom;
    int has_virial;
    float energy;
    float virial[6];
    float box[18];
    std::vector<int> atomic_number;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> fx;
    std::vector<float> fy;
    std::vector<float> fz;
  };
  std::vector<Structure> structures;

  // functions related to initialization
  void read_Nc(FILE*);
  void read_Na(FILE*);
  void copy_structures();
  void report_Na();
  void read_train_in(char*, Parameters& para);
  float get_rmse_force(const int, const int);
  float get_rmse_energy(const int, const int);
  float get_rmse_virial(const int, const int);
  void find_neighbor(Parameters& para);
  void make_train_or_test_set(
    Parameters& para, int num, int offset, std::vector<int>& configuration_id, Dataset& train_set);
};
