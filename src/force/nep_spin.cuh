/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    Copyright 2026 NEPAdapters contributors
    This file is part of GPUMD and is distributed under GPLv3 or later.
*/

#pragma once

#include "neighbor.cuh"
#include "nep.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <string>
#include <vector>

struct NEP_Spin_Data {
  GPU_Vector<float> parameters;
  GPU_Vector<float> descriptor_parameters_type_pair;
  GPU_Vector<float> descriptor;
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> f12x;
  GPU_Vector<float> f12y;
  GPU_Vector<float> f12z;
  GPU_Vector<float> spin_baseline;
  GPU_Vector<int> spin_dof_type_active;
  GPU_Vector<int> spin_env_type_active;
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
};

class NEP_Spin : public Potential
{
public:
  struct Spin_Polynomial_Layout {
    int channels = 0;
    int pair_count = 0;
    int density_stride = 38;
    int moment_count = 0;
    int local_s2 = -1;
    int edge_l0_dot = -1;
    int edge_l0_neighbor_s2 = -1;
    int edge_l2_pair = -1;
    int center_l2_environment = -1;
    int edge_l0_dot2 = -1;
    int density_l0_self = -1;
    int density_l1_longitudinal_self = -1;
    int density_l1_axial_self = -1;
    int density_l1_stf_self = -1;
    int density_l1_product_self = -1;
    int density_l2_product_self = -1;
    int density_l0_dot_response = -1;
    int correlation_same_edge = -1;
    int correlation_distinct_neighbor = -1;
    int correlation_distinct_l1 = -1;
    int correlation_distinct_l2 = -1;
    int angular_l1_moment_offset = -1;
    int angular_l2_moment_offset = -1;
    int coupling_l11_axial = -1;
    int edge_l11_axial = -1;
    int coupling_l22_axial = -1;
    int edge_l22_axial = -1;
    int edge_l0_moment_gate = -1;
    int coupling_l11_dot_response = -1;
    int coupling_l111_p_m_x = -1;
    int coupling_l22_dot_response = -1;
    int coupling_l111_p_qs_x = -1;
    int coupling_l112_edge_response = -1;
    int coupling_l111_bulk = -1;
    int descriptor_dim = 0;
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
    int spin_mode = 3;
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
    std::vector<double> spin_baseline;
    std::vector<int> spin_dof_type_active;
    std::vector<int> spin_env_type_active;
    std::vector<double> spin_cutoff_by_type;
    Body_Channels body;
    Spin_Polynomial_Layout spin_polynomial_layout;
  };

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
    GPU_Vector<double>& mforce) override;

  void resize_domain(int num_atoms);
  void invalidate_neighbors() { neighbor_.invalidate(); }

  const Model& model(void) const { return model_; }

private:
  Model model_;
  NEP_Spin_Data data_;
  Neighbor neighbor_;
  NEP::ParaMB paramb_;
  NEP::ANN ann_;
  NEP::ZBL zbl_;
  int num_atoms_ = 0;
  int input_neighbor_capacity_ = 0;
  gpuStream_t structural_descriptor_stream_ = nullptr;
  gpuStream_t spin_descriptor_stream_ = nullptr;
  bool small_box_initialized_ = false;
  int small_box_capacity_ = 0;
  NEP::ExpandedBox expanded_box_;

  void read_model(const char* file_potential);
  void initialize_small_box(const Box& box);
};
