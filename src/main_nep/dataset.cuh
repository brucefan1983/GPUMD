/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
  int Nc;             // number of configurations
  int N;              // total number of atoms (sum of Na[])
  int max_Na;         // number of atoms in the largest configuration
  int max_NN_radial;  // radial neighbor list size
  int max_NN_angular; // angular neighbor list size

  GPU_Vector<int> Na;          // number of atoms in each configuration
  GPU_Vector<int> Na_sum;      // prefix sum of Na
  std::vector<int> Na_cpu;     // number of atoms in each configuration
  std::vector<int> Na_sum_cpu; // prefix sum of Na_cpu

  GPU_Vector<int> type;           // atom type (0, 1, 2, 3, ...)
  GPU_Vector<float> r;            // position
  GPU_Vector<float> box;          // (expanded) box and inverse box (18 components)
  GPU_Vector<float> box_original; // (original) box (9 components)
  GPU_Vector<int> num_cell;       // number of cells in the expanded box (3 components)

  GPU_Vector<float> charge;      // calculated charge in GPU
  GPU_Vector<float> charge_shifted;      // shifted charge in GPU
  GPU_Vector<float> bec;         // Born effective charge in GPU
  GPU_Vector<float> energy;      // calculated energy in GPU
  GPU_Vector<float> virial;      // calculated virial in GPU
  GPU_Vector<float> force;       // calculated force in GPU
  GPU_Vector<float> avirial; // calculated atomic virial in GPU
  std::vector<float> charge_cpu;  // calculated charge in CPU
  std::vector<float> bec_cpu;     // calculated BEC in CPU
  std::vector<float> energy_cpu; // calculated energy in CPU
  std::vector<float> virial_cpu; // calculated virial in CPU
  std::vector<float> force_cpu;  // calculated force in CPU
  std::vector<float> avirial_cpu;   // calculated atomic virial in CPU

  GPU_Vector<float> energy_weight_gpu;    // energy weight in GPU
  GPU_Vector<float> charge_ref_gpu;       // reference charge in GPU
  GPU_Vector<float> energy_ref_gpu;       // reference energy in GPU
  GPU_Vector<float> virial_ref_gpu;       // reference virial in GPU
  GPU_Vector<float> force_ref_gpu;        // reference force in GPU
  GPU_Vector<float> bec_ref_gpu;          // reference BEC in GPU
  GPU_Vector<float> avirial_ref_gpu;     // reference atomic virial in GPU
  GPU_Vector<float> temperature_ref_gpu;  // reference temperature in GPU
  std::vector<float> energy_weight_cpu;   // energy weight in CPU
  std::vector<float> charge_ref_cpu;      // reference charge in CPU
  std::vector<float> energy_ref_cpu;      // reference energy in CPU
  std::vector<float> virial_ref_cpu;      // reference virial in CPU
  std::vector<float> force_ref_cpu;       // reference force in CPU
  std::vector<float> bec_ref_cpu;         // reference BEC in CPU
  std::vector<float> avirial_ref_cpu;      // reference atomic virial in CPU
  std::vector<float> weight_cpu;          // configuration weight in CPU
  std::vector<float> temperature_ref_cpu; // reference temeprature in CPU

  GPU_Vector<float> type_weight_gpu; // relative force weight for different atom types (GPU)

  std::vector<float> error_cpu; // error in energy, virial, or force
  GPU_Vector<float> error_gpu;  // error in energy, virial, or force

  std::vector<bool> has_type;

  std::vector<Structure> structures;

  void
  construct(Parameters& para, std::vector<Structure>& structures, int n1, int n2, int device_id);
  std::vector<float> get_rmse_force(Parameters& para, const bool use_weight, int device_id);
  std::vector<float> get_rmse_energy(
    Parameters& para,
    float& energy_shift_per_structure,
    const bool use_weight,
    const bool do_shift,
    int device_id);
  std::vector<float> get_rmse_virial(Parameters& para, const bool use_weight, int device_id);
  std::vector<float> get_rmse_avirial(Parameters& para, const bool use_weight, int device_id);
  std::vector<float> get_rmse_charge(Parameters& para, int device_id);

private:
  void copy_structures(std::vector<Structure>& structures_input, int n1, int n2);
  void find_has_type(Parameters& para);
  void find_Na(Parameters& para);
  void initialize_gpu_data(Parameters& para);
  void find_neighbor(Parameters& para);
};
