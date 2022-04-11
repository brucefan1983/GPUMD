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
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
class Neighbor;

struct NEP4_Data {
  GPU_Vector<double> f12x; // 3-body or manybody partial forces
  GPU_Vector<double> f12y; // 3-body or manybody partial forces
  GPU_Vector<double> f12z; // 3-body or manybody partial forces
  GPU_Vector<double> dq_dx;
  GPU_Vector<double> dq_dy;
  GPU_Vector<double> dq_dz;
  GPU_Vector<double> q;               // descriptors (angular only)
  GPU_Vector<double> gnn_descriptors; // temporary descriptors for use in GNN
  GPU_Vector<double>
    gnn_messages; // messages q * theta for all atoms, same shape as gnn_descriptors
  GPU_Vector<double> gnn_messages_p_x; // derivatives of messages, theta * dq_dr
  GPU_Vector<double> gnn_messages_p_y;
  GPU_Vector<double> gnn_messages_p_z;
  GPU_Vector<double> dU_dq;
  GPU_Vector<double> s;         // s in the NEP3 manuscript
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP4 : public Potential
{
public:
  struct ParaMB {
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int n_max_angular = 0;      // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;              // l = 0, 1, 2, ..., L_max
    int basis_size = 8;
    int num_types = 0;
    int num_types_sq = 0;
  };

  struct ANN {
    int dim = 0;          // dimension of the descriptor
    int num_neurons1 = 0; // number of neurons in the 1st hidden layer
    int num_para = 0;     // number of parameters
    const float* w0;      // weight from the input layer to the hidden layer
    const float* b0;      // bias for the hidden layer
    const float* w1;      // weight from the hidden layer to the output layer
    const float* b1;      // bias for the output layer
    const float* c;
  };

  struct GNN {
    int num_para = 0;
    const float* theta; // weights of size N_descriptor x F, where F = N_descriptor atm (size of
                        // output descriptor)
  };

  struct ZBL {
    bool enabled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float atomic_numbers[10];
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  NEP4(FILE* fid, char* input_dir, int num_types, bool enable_zbl, const Neighbor& neighbor);
  virtual ~NEP4(void);
  virtual void compute(
    const int type_shift,
    const Box& box,
    const Neighbor& neighbor,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  ParaMB paramb;
  ANN ann;
  GNN gnn;
  ZBL zbl;
  NEP4_Data nep_data;
  ExpandedBox ebox;
  void update_potential(FILE* fid);
  void update_potential(const float* parameters, ANN& ann, GNN& gnn);

  void compute_small_box(
    const int type_shift,
    const Box& box,
    const Neighbor& neighbor,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    const int type_shift,
    const Box& box,
    const Neighbor& neighbor,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
};
