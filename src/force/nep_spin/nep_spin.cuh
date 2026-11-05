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
NEP-spin potential interface, model parameters, and GPU workspaces.

Contributors:
Chengbing Chen, 1747193328@qq.com

References:
Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#pragma once

#include "force/neighbor.cuh"
#include "force/potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <string>
#include <vector>

class NEP_Spin : public Potential
{
public:
  // Model-specific parameter types passed by value to CUDA kernels.
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    int model_type = 0; // 0=potential, 3=temperature-dependent free energy
    float rc_radial_max = 0.0f;
    float rc_radial_max_inv = 0.0f;
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
    const float* c;
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

private:
  struct Data {
    GPU_Vector<float> parameters;
    GPU_Vector<float> descriptor_parameters_type_pair;
    GPU_Vector<float> descriptor;
    GPU_Vector<float> Fp;
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> f12x;
    GPU_Vector<float> f12y;
    GPU_Vector<float> f12z;
    GPU_Vector<double> energy_reference;
    GPU_Vector<int> spin_type_active;
    GPU_Vector<float> spin_cutoff_pair;
    GPU_Vector<float> spin_projection_parameters;
    GPU_Vector<float> spin3_moments;
    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<int> NN_spin;
    GPU_Vector<int> NL_spin;
    GPU_Vector<double> r12_radial;
    GPU_Vector<double> r12_angular;
    GPU_Vector<double> r12_spin;
    GPU_Vector<int> required_small_box_capacity;
  };

  struct Body_Channels {
    int l_max_3body = 0;
    bool has_q_222 = false;
    bool has_q_1111 = false;
    bool has_q_112 = false;
    bool has_q_123 = false;
    bool has_q_233 = false;
    bool has_q_134 = false;

    int count(void) const;
  };

  struct Model {
    int num_types = 0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int hidden_neurons = 0;
    int max_neighbors_global = 0;
    int max_neighbors_angular = 0;
    int neighbor_capacity = 0;
    int spin_basis_size[2] = {0, 0};
    int spin_l_max[3] = {0, 0, 0};
    int spin_compress = 0;
    int spin_order = 0;
    int spin_soc = 0;
    int spin_projection_size = 0;
    double cutoff_radial = 0.0;
    double cutoff_angular = 0.0;
    double spin_cutoff = 0.0;
    int struct_descriptor_dim = 0;
    int spin_descriptor_dim = 0;
    int descriptor_dim = 0;
    std::size_t ann_parameter_count = 0;
    std::size_t radial_parameter_count = 0;
    std::size_t angular_parameter_count = 0;
    std::size_t spin_parameter_count = 0;
    std::size_t spin_projection_parameter_count = 0;
    std::size_t model_parameter_count = 0;
    std::vector<std::string> elements;
    std::vector<double> energy_reference;
    std::vector<int> spin_type_active;
    std::vector<double> spin_cutoff_by_type;
    Body_Channels body;
    int spin_moment_count = 0;
  };

public:
  NEP_Spin(const char* file_potential, const int num_atoms);
  virtual ~NEP_Spin(void);

  using Potential::compute;

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial) override;

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& spin,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    GPU_Vector<double>& mforce);

  void resize_domain(int num_atoms);
  void invalidate_neighbors() { neighbor_.invalidate(); }

  // The caller owns the buffers until the next compute completes. Only that
  // evaluation is captured; later endpoint evaluations leave the buffers intact.
  void request_force_snapshot(GPU_Vector<double>& force, GPU_Vector<double>& mforce)
  {
    force_snapshot_ = &force;
    mforce_snapshot_ = &mforce;
  }

  const std::vector<int>& spin_types() const { return model_.spin_type_active; }

private:
  GPU_Vector<double>* force_snapshot_ = nullptr;
  GPU_Vector<double>* mforce_snapshot_ = nullptr;
  void save_force_snapshot(const GPU_Vector<double>&, const GPU_Vector<double>&);
  Model model_;
  Data data_;
  class Neighbors
  {
  public:
    GPU_Vector<int> NN, NL;
    void initialize(double rc, int atoms, int capacity);
    void invalidate() { valid_ = false; }
    int get_capacity() const { return capacity_; }
    double minimum_safe_box_thickness(double rc) const { return 2.5 * (rc + skin_); }
    void find_neighbor_global(
      double rc, Box& box, const GPU_Vector<int>& type, const GPU_Vector<double>& position);

  private:
    static constexpr double skin_ = 1.0;
    int capacity_ = 0;
    bool valid_ = false;
    double box_[9] = {};
    GPU_Vector<int> cell_count_, cell_sum_, cell_contents_, flag_;
    GPU_Vector<double> reference_;
  };
  Neighbors neighbor_;
  NEP_Spin::ParaMB paramb_;
  NEP_Spin::ANN ann_;
  NEP_Spin::ZBL zbl_;
  int num_atoms_ = 0;
  int input_neighbor_capacity_ = 0;
  gpuStream_t structural_descriptor_stream_ = nullptr;
  gpuStream_t spin_descriptor_stream_ = nullptr;
  bool small_box_initialized_ = false;
  int small_box_capacity_ = 0;
  NEP_Spin::ExpandedBox expanded_box_;

  template <int atoms_per_warp, int edge_lanes, bool l4_q222_only>
  void launch_force_angular_warp(
    const NEP_Spin::ParaMB& paramb,
    const NEP_Spin::ANN& ann,
    const Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const Data& data,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    const int grid_size,
    const int block_size,
    const std::size_t shared_size);

  template <int C>
  void launch_spin3_descriptors_typed(
    const Model& model,
    const Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& spin,
    Data& data,
    const double* slot_r12 = nullptr,
    int r12_plane_size = 0,
    gpuStream_t stream = nullptr);

  void launch_spin3_descriptors(
    const Model& model,
    const Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& spin,
    Data& data,
    const double* slot_r12 = nullptr,
    int r12_plane_size = 0,
    gpuStream_t stream = nullptr);

  template <int C>
  void launch_spin3_forces_typed(
    const Model& model,
    const Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& spin,
    Data& data,
    GPU_Vector<double>& force,
    GPU_Vector<double>& mforce,
    GPU_Vector<double>& virial,
    const double* slot_r12 = nullptr,
    int r12_plane_size = 0);

  void launch_spin3_forces(
    const Model& model,
    const Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& spin,
    Data& data,
    GPU_Vector<double>& force,
    GPU_Vector<double>& mforce,
    GPU_Vector<double>& virial,
    const double* slot_r12 = nullptr,
    int r12_plane_size = 0);

  void read_model(const char* file_potential);
  void initialize_small_box(const Box& box);
  void resize_small_box_storage(int capacity);
};
