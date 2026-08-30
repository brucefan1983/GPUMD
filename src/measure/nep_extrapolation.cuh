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
#include "force/neighbor.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>

class Atom;

class NEP_Extrapolation
{
public:
  struct ParaMB {
    float rc_radial_max = 0.0f;
    float rc_radial[NUM_ELEMENTS];
    float rc_angular[NUM_ELEMENTS];
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular;
    int has_q_222 = 0;
    int has_q_1111 = 0;
    int has_q_112 = 0;
    int has_q_123 = 0;
    int has_q_233 = 0;
    int has_q_134 = 0;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    int num_para_ann = 0;
    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;
    const float* c;
    const float* c_type_pair;
    const float* q_scaler;
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  NEP_Extrapolation(const char* file_potential, const Atom& atom);

  int get_B_size_per_atom() const { return B_size_per_atom_; }
  double* get_B_projection() { return B_projection_.data(); }

  void compute(Box& box, const GPU_Vector<double>& position);

private:
  struct Data {
    GPU_Vector<float> Fp;
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> descriptor_parameters_type_pair;
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> parameters;
  } data_;

  struct Small_Box_Data {
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> r12;
  } small_box_data_;

  ParaMB paramb_;
  ANN annmb_;
  ExpandedBox ebox_;
  Neighbor neighbor_;
  GPU_Vector<int> type_;
  GPU_Vector<double> potential_per_atom_;
  GPU_Vector<double> virial_per_atom_;
  GPU_Vector<double> B_projection_;
  int num_atoms_ = 0;
  int B_size_per_atom_ = 0;
  bool is_polarizability_ = false;
  std::string atom_types_[NUM_ELEMENTS];

  void update_potential(float* parameters);
  void compute_large_box(Box& box, const GPU_Vector<double>& position);
  void compute_small_box(Box& box, const GPU_Vector<double>& position);
};
