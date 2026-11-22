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
#include "main_nep/dataset.cuh"
#include "spin_parameters.cuh"
#include <array>
struct Spin_Data {
  struct Geometry {
    std::array<double, 9> cell;
    std::vector<std::array<double, 3>> position;
  };
  struct Labels {
    bool has_full_virial = false;
    std::array<float, 3> antisymmetric_virial{}; // half differences: xy, yz, zx
    int has_mforce = 0, has_spin_response_metadata = 0, has_spin_response = 0;
    double spin_response_coordinate = 0.0;
    std::string spin_response_probe, spin_response_group;
    std::vector<double> sx, sy, sz;
    std::vector<float> mfx, mfy, mfz;
    std::vector<float> spin_tangent_x, spin_tangent_y, spin_tangent_z;
  };
  struct FrameInput : Labels {
    Geometry geometry;
  };
  int max_NN_spin = 0;
  std::vector<Labels> labels;
  GPU_Vector<double> r_spin, spin, r12_spin;
  GPU_Vector<int> NN_spin, NL_spin, has_mforce_gpu, spin_type_active_gpu;
  GPU_Vector<float> mforce, mforce_ref_gpu;
  std::vector<float> mforce_cpu, mforce_ref_cpu;
  std::vector<int> has_mforce_cpu;
  // Per-frame loss counts, fixed with labels and the active mask at construction.
  std::vector<int> active_count_cpu, torque_active_count_cpu;
  void construct(Spin_Parameters&, Dataset&, const std::vector<FrameInput>&);
  std::vector<float> get_rmse_mforce(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_rmse_mforce_parallel(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_rmse_mforce_perp(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_rmse_tau(Spin_Parameters&, Dataset&, bool, int);
  std::vector<float> get_full_virial_rmse(Spin_Parameters&, Dataset&, bool, int);

private:
  GPU_Vector<int> full_virial_mask_gpu;
  GPU_Vector<float> antisymmetric_virial_ref_gpu;
};
