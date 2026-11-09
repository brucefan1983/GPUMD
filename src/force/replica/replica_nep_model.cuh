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

/*----------------------------------------------------------------------------80
NEP model of replica dynamics.

A copy of the model part of NEP (force/nep/nep.cuh) with the evaluation split
into two steps (neighbor lists, then energy, force and virial) on a given
workspace and stream, so that several systems can be evaluated concurrently.
It is separate from NEP so that ordinary NEP force evaluation is not affected
by replica dynamics; keep it in sync with force/nep by hand.
------------------------------------------------------------------------------*/

#pragma once
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

struct NEP_Replica_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<double> energy_reference;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP_Replica_Model
{
public:
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    float rc_radial_max = 0.0f;
    float rc_radial[NUM_ELEMENTS];  // radial cutoff
    float rc_angular[NUM_ELEMENTS]; // angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
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
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c_type_pair;
    const float* q_scaler;
    const double* energy_reference = nullptr;
  };

  struct ZBL {
    bool enabled = false;
    bool flexible = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct Small_Box_Data {
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> r12;
  };

  double rc = 0.0;         // largest cutoff

  // Reads the model; verbose prints the model information.
  NEP_Replica_Model(const char* file_potential, const bool verbose);

  // Whether the box needs the small-box path; also fills the expanded box for it.
  static bool get_expanded_box(const double rc, const Box& box, ExpandedBox& ebox);

protected:
  // Large-box evaluation of atoms [N1, N2) of N atoms in two steps: the radial and
  // angular neighbor lists from a global candidate list, then energy, force and virial.
  // Box_View is a Box, or a Batch_Box for replicas concatenated into one system.
  // Defined in replica_nep_kernels.cuh.
  template <typename Box_View>
  void find_neighbor_lists_large_box(
    const Box_View& box,
    const int N,
    const int N1,
    const int N2,
    const int* type,
    const double* position,
    const int* NN_global,
    const int* NL_global,
    NEP_Replica_Data& data,
    gpuStream_t stream) const;

  template <typename Box_View>
  void compute_large_box_from_neighbor_lists(
    const Box_View& box,
    const int N,
    const int N1,
    const int N2,
    const int* type,
    const double* position,
    double* potential,
    double* force,
    double* virial,
    NEP_Replica_Data& data,
    gpuStream_t stream);

  // Small-box evaluation in the same two steps, using the expanded box.
  void find_neighbor_lists_small_box(
    const Box& box,
    const ExpandedBox& expanded_box,
    const int N,
    const int N1,
    const int N2,
    const int* type,
    const double* position,
    Small_Box_Data& small_box,
    gpuStream_t stream) const;

  void compute_small_box_from_neighbor_lists(
    const int N,
    const int N1,
    const int N2,
    const int* type,
    double* potential,
    double* force,
    double* virial,
    NEP_Replica_Data& data,
    Small_Box_Data& small_box,
    gpuStream_t stream);

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  NEP_Replica_Data nep_data; // model parameters

  void update_potential(float* parameters, ANN& ann);
};
