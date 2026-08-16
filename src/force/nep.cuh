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
#include "dftd3.cuh"
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include "utilities/gpu_vector.cuh"
#include <map>
#include <memory>
#include <mutex>
#include <vector>

struct NEP_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> descriptor_parameters_type_pair;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
};

class NEP : public Potential
{
public:
  NEP_Data nep_data;
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    int version = 4; // NEP version, 3 for NEP3 and 4 for NEP4
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    float rc_radial_max = 0.0f;
    float rc_radial_max_inv = 0.0f; 
    float rc_radial[NUM_ELEMENTS];     // radial cutoff
    float rc_angular[NUM_ELEMENTS];    // angular cutoff
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
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
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
    const float* c;
    const float* c_type_pair;
    // for the scalar part of polarizability
    const float* w0_pol[10];
    const float* b0_pol[10];
    const float* w1_pol[10];
    const float* b1_pol;
    const float* q_scaler;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
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

  NEP(
    const char* file_potential,
    int num_atoms,
    bool allocate_default_workspace = true);
  virtual ~NEP(void);
  void prepare_stream(gpuStream_t stream);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    gpuStream_t stream);

  virtual void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    gpuStream_t stream);

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  Small_Box_Data small_box_data;
  DFTD3 dftd3;
  Neighbor neighbor;

  struct Compute_State {
    int large_box_calls = 0;
    int small_box_calls = 0;
  };

  struct Stream_Workspace {
    NEP_Data data;
    ExpandedBox ebox;
    Small_Box_Data small_box_data;
    Neighbor neighbor;
    Compute_State state;
    GPU_Pinned_Vector<int> cpu_NN_radial;
    GPU_Pinned_Vector<int> cpu_NN_angular;
    std::mutex submission_mutex;
  };

  struct Host_Neighbor_Data {
    int* radial;
    int* angular;
  };

  int num_atoms_ = 0;
  int device_id_ = -1;
  bool has_default_workspace_ = true;
  Compute_State default_state_;
  std::map<gpuStream_t, std::unique_ptr<Stream_Workspace>> stream_workspaces_;
  std::mutex stream_workspaces_mutex_;
  std::mutex neighbor_log_mutex_;

  void initialize_workspace(
    NEP_Data& data, Neighbor& neighbor, bool allocate_host_data) const;
  void validate_default_workspace() const;
  void validate_stream(const gpuStream_t stream) const;
  Stream_Workspace& get_stream_workspace(const gpuStream_t stream);
  void report_neighbor_information(
    GPU_Vector<int>& radial,
    GPU_Vector<int>& angular,
    Host_Neighbor_Data host_data,
    int& num_calls,
    gpuStream_t stream);

  void update_potential(float* parameters, ANN& ann);

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    NEP_Data& data,
    Small_Box_Data& small_box,
    const ExpandedBox& expanded_box,
    Host_Neighbor_Data host_data,
    int& num_calls,
    const gpuStream_t stream);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    NEP_Data& data,
    Neighbor& neighbor_list,
    Host_Neighbor_Data host_data,
    int& num_calls,
    const gpuStream_t stream);

  void compute_small_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    NEP_Data& data,
    Small_Box_Data& small_box,
    const ExpandedBox& expanded_box,
    Host_Neighbor_Data host_data,
    int& num_calls,
    const gpuStream_t stream);

  void compute_large_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    NEP_Data& data,
    Neighbor& neighbor_list,
    Host_Neighbor_Data host_data,
    int& num_calls,
    const gpuStream_t stream);

  bool has_dftd3 = false;
  void initialize_dftd3();
};
