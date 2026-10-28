/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    Copyright 2026 NEPAdapters contributors
    This file is part of GPUMD and is distributed under LGPLv3 or later.
*/

#pragma once

#include "force/neighbor.cuh"
#include "force/nep.cuh"
#include "force/potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <string>
#include <vector>

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
    Spin_Polynomial_Layout spin_polynomial_layout;
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
    GPU_Vector<double>& mforce) override;

  void resize_domain(int num_atoms);
  void invalidate_neighbors() { neighbor_.invalidate(); }

  const std::vector<int>& spin_types() const { return model_.spin_type_active; }

private:
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

  template <int atoms_per_warp, int edge_lanes, bool l4_q222_only>
  void launch_force_angular_warp(
    const NEP::ParaMB& paramb,
    const NEP::ANN& ann,
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
};
