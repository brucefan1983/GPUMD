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
#include "structure.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>
class NEP_Parameters;

class Dataset
{
public:
  int batch_index; // training batch index; -1 denotes the test set
  int device_id;
  int Nc;             // number of configurations
  int N;              // total number of atoms (sum of Na[])
  int max_Na;         // number of atoms in the largest configuration
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size
  int total_NN_radial;  // total number of radial neighbors
  int total_NN_angular; // total number of angular neighbors

  GPU_Vector<int> Na;          // number of atoms in each configuration
  GPU_Vector<int> Na_sum;      // prefix sum of Na
  std::vector<int> Na_cpu;     // number of atoms in each configuration
  std::vector<int> Na_sum_cpu; // prefix sum of Na_cpu

  GPU_Vector<int> type;           // atom type (0, 1, 2, 3, ...)
  GPU_Vector<float> r;            // position
  GPU_Vector<float> box;          // (expanded) box and inverse box (18 components)
  GPU_Vector<float> box_original; // (original) box (9 components)
  GPU_Vector<int> num_cell;       // number of cells in the expanded box (3 components)
  GPU_Vector<int> pbc;            // 1 for PPP and 0 for FFF (one value per configuration)

  GPU_Vector<int> NN_radial;      // radial neighbor number
  GPU_Vector<int> NN_angular;     // angular neighbor number
  GPU_Vector<int> NN_radial_sum;  // prefix sum of NN_radial
  GPU_Vector<int> NN_angular_sum; // prefix sum of NN_angular

  GPU_Vector<double> energy;      // calculated energy in GPU
  GPU_Vector<float> virial;       // calculated virial in GPU
  GPU_Vector<float> force;        // calculated force in GPU
  std::vector<double> energy_cpu; // calculated energy in CPU
  std::vector<float> virial_cpu;  // calculated virial in CPU
  std::vector<float> force_cpu;   // calculated force in CPU

  GPU_Vector<float> energy_weight_gpu;  // energy weight in GPU
  GPU_Vector<double> energy_ref_gpu;    // reference energy in GPU
  GPU_Vector<float> virial_ref_gpu;     // reference virial in GPU
  GPU_Vector<float> force_ref_gpu;      // reference force in GPU
  std::vector<float> energy_weight_cpu; // energy weight in CPU
  std::vector<double> energy_ref_cpu;   // reference energy in CPU
  std::vector<float> virial_ref_cpu;    // reference virial in CPU
  std::vector<float> force_ref_cpu;     // reference force in CPU
  std::vector<float> weight_cpu;        // configuration weight in CPU

  GPU_Vector<float> type_weight_gpu; // relative force weight for different atom types (GPU)

  std::vector<double> energy_error_cpu;
  GPU_Vector<double> energy_error_gpu;
  std::vector<float> error_cpu; // error in virial or force
  GPU_Vector<float> error_gpu;  // error in virial or force

  std::vector<bool> has_type;

  std::vector<Structure> structures;

  void construct(NEP_Parameters& para, std::vector<Structure>& structures,
    int n1, int n2, int device_id, int batch_index);
  std::vector<float> get_rmse_force(NEP_Parameters& para, const bool use_weight, int device_id);
  std::vector<double> get_rmse_energy(
    NEP_Parameters& para,
    double& energy_shift_per_structure,
    const bool use_weight,
    const bool do_shift,
    int device_id);
  std::vector<float> get_rmse_virial(
    NEP_Parameters& para, const bool use_weight, int device_id, int num_components = 6);

private:
  void copy_structures(std::vector<Structure>& structures_input, int n1, int n2);
  void find_has_type(NEP_Parameters& para);
  void find_Na(NEP_Parameters& para);
  void initialize_gpu_data(NEP_Parameters& para);
  void find_neighbor(NEP_Parameters& para);
};

class NEP_Neighbor
{
public:
  GPU_Vector<int> NL_radial;
  GPU_Vector<int> NL_angular;
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;

  void prepare(NEP_Parameters& para, Dataset& dataset, int device_id);

private:
  GPU_Vector<float> rc_radial;
  GPU_Vector<float> rc_angular;
  const Dataset* neighbor_dataset = nullptr;
};
