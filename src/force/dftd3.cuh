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
#include <string>
#include <vector>
class Box;

class DFTD3
{
public:
  struct DFTD3_Para {
    float s6 = 0.0;
    float s8 = 0.0;
    float a1 = 0.0;
    float a2 = 0.0;
    int atomic_number[94];
  };

  DFTD3_Para dftd3_para;

  bool enabled = false;
  float rc_radial = 15.0;
  float rc_angular = 10.0;
  GPU_Vector<float> cn;
  GPU_Vector<float> c6_ref;
  GPU_Vector<float> dc6_sum;
  GPU_Vector<float> dc8_sum;

  GPU_Vector<int> NN_radial;
  GPU_Vector<int> NL_radial;
  GPU_Vector<int> NN_angular;
  GPU_Vector<int> NL_angular;
  GPU_Vector<float> r12;

  GPU_Vector<int> cell_count_radial;
  GPU_Vector<int> cell_count_sum_radial;
  GPU_Vector<int> cell_contents_radial;
  GPU_Vector<int> cell_count_angular;
  GPU_Vector<int> cell_count_sum_angular;
  GPU_Vector<int> cell_contents_angular;

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  ExpandedBox ebox;

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void initialize(std::string& ex_functional, const float rc_radial, const float rc_angular);
};
