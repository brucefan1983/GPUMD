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
  std::vector<int> Na_original;    // number of atoms before possible box replication
  GPU_Vector<int> Na_sum;          // prefix sum of Na
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
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;

  struct Structure {
    int num_cell_a;
    int num_cell_b;
    int num_cell_c;
    int num_atom;
    int has_virial;
    float energy;
    float virial[6];
    float box_original[9];
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

  void construct(char*, Parameters& para);
  float get_rmse_force();
  float get_rmse_energy();
  float get_rmse_virial();

private:
  // functions called by construct:
  void read_train_in(char*, Parameters& para);
  void reorder(char* input_dir);
  void find_Na();
  void initialize_gpu_data();
  void calculate_types();
  void find_neighbor(Parameters& para);
  // functions called by read_train_in:
  void read_Nc(FILE*);
  void read_Na(FILE*);
  void read_box(FILE* fid, int nc, Parameters& para);
  void read_energy_virial(FILE* fid, int nc);
  void read_force(FILE* fid, int nc);
};
