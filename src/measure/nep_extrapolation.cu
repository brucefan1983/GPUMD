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
NEP4 descriptor projection used by compute_extrapolation.
The descriptor and B-projection expressions are copied from the original NEP4
implementation. This class is independent of force/NEP.
------------------------------------------------------------------------------*/

#include "nep_extrapolation.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_io/nep_model_file.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

static __device__ void apply_ann_one_layer(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  float* q,
  float& energy,
  float* energy_derivative,
  double* B_projection)
{
  for (int n = 0; n < N_neu; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    float x1 = tanh(w0_times_q - b0[n]);
    float tanh_der = 1.0f - x1 * x1;

    // calculate B_projection:
    // dE/dw0
    for (int d = 0; d < N_des; ++d)
      B_projection[n * (N_des + 2) + d] = tanh_der * q[d] * w1[n];
    // dE/db0
    B_projection[n * (N_des + 2) + N_des] = -tanh_der * w1[n];
    // dE/dw1
    B_projection[n * (N_des + 2) + N_des + 1] = x1;

    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      float y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[0];
}


#include "nep_extrapolation_small_box.cuh"

static __global__ void find_neighbor_list_large_box(
  NEP_Extrapolation::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];
  int count_radial = 0;
  int count_angular = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[static_cast<size_t>(N) * i1 + n1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    int t2 = g_type[n2];
    float rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    float rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    if (d12_square >= rc_radial * rc_radial) {
      continue;
    }
    g_NL_radial[static_cast<size_t>(N) * count_radial++ + n1] = n2;
    if (d12_square < rc_angular * rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP_Extrapolation::ParaMB paramb,
  NEP_Extrapolation::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz,
  bool need_B_projection,
  double* B_projection,
  int B_projection_size)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1] = s[abc];
      }
    }

    // normalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (!need_B_projection)
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp);
    else
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        B_projection + n1 * B_projection_size);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP_Extrapolation::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }

  if (is_small_box) {
    if (thickness_x > 10 * rc || thickness_y > 10 * rc || thickness_z > 10 * rc) {
      std::cout << "Error:\n"
                << "    The box has\n"
                << "        a thickness < 2.5 radial cutoffs in a periodic direction.\n"
                << "        and a thickness > 10 radial cutoffs in another direction.\n"
                << "    Please increase the periodic direction(s).\n";
      exit(1);
    }

    ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
    ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
    ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
    ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
    ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
    ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
    ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
    ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
    ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

    ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
    ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
    ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
    ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
    ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
    ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
    ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
    ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
    ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
    double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                 ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                 ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
    for (int n = 9; n < 18; n++) {
      ebox.h[n] /= det;
    }
  }

  return is_small_box;
}


NEP_Extrapolation::NEP_Extrapolation(const char* file_potential, const Atom& atom)
  : num_atoms_(atom.number_of_atoms)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    const std::string error = "Cannot open potential file " + std::string(file_potential) + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  NEP_Model_File model;
  std::string error;
  if (!model.read(input, error) || !model.validate_md(error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (
    model.name != "nep" && model.name != "nep_dipole" && model.name != "nep_polarizability") {
    PRINT_INPUT_ERROR(
      "compute_extrapolation supports model nep, nep_dipole or nep_polarizability.");
  }
  paramb_.num_types = model.num_types;

  std::vector<int> mapped_type(num_atoms_);
  for (int n = 0; n < num_atoms_; ++n) {
    mapped_type[n] = -1;
    for (int t = 0; t < paramb_.num_types; ++t) {
      if (atom.cpu_atom_symbol[n] == model.elements[t]) {
        mapped_type[n] = t;
        break;
      }
    }
    if (mapped_type[n] < 0) {
      PRINT_INPUT_ERROR("There is atom in model.xyz that is not allowed in the extrapolation NEP.");
    }
  }
  type_.resize(num_atoms_);
  type_.copy_from_host(mapped_type.data());

  for (int n = 0; n < paramb_.num_types; ++n) {
    paramb_.rc_radial[n] = model.rc_radial[n];
    paramb_.rc_angular[n] = model.rc_angular[n];
    if (paramb_.rc_radial[n] > paramb_.rc_radial_max) {
      paramb_.rc_radial_max = paramb_.rc_radial[n];
    }
  }
  paramb_.MN_radial = int(ceil(model.max_radial * 1.25));
  paramb_.MN_angular = int(ceil(model.max_angular * 1.25));

  paramb_.n_max_radial = model.n_max_radial;
  paramb_.n_max_angular = model.n_max_angular;
  paramb_.basis_size_radial = model.basis_size_radial;
  paramb_.basis_size_angular = model.basis_size_angular;
  paramb_.L_max = model.L_max;
  paramb_.has_q_222 = 2 * model.has_q_222;
  paramb_.has_q_1111 = model.has_q_1111;
  paramb_.has_q_112 = model.has_q_112;
  paramb_.has_q_123 = model.has_q_123;
  paramb_.has_q_233 = model.has_q_233;
  paramb_.has_q_134 = model.has_q_134;

  annmb_.num_neurons1 = model.num_neurons1;
  annmb_.dim = model.layout.dim;
  paramb_.num_types_sq = paramb_.num_types * paramb_.num_types;
  // Polarizability stores two complete ANNs before the descriptor coefficients.
  annmb_.num_para_ann = model.layout.number_of_variables_ann * model.layout.ann_count;
  annmb_.num_para = model.layout.number_of_variables;
  paramb_.num_c_radial =
    paramb_.num_types_sq * (paramb_.n_max_radial + 1) * (paramb_.basis_size_radial + 1);

  // Store ANN parameters, type-pair-major descriptor coefficients and scalers in one buffer.
  data_.parameters.resize(annmb_.num_para + annmb_.dim);
  data_.parameters.copy_from_host(model.parameters.data(), model.parameters.size());
  data_.parameters.copy_from_host(
    model.q_scaler.data(), model.q_scaler.size(), annmb_.num_para);
  update_potential(data_.parameters.data());
  annmb_.q_scaler = data_.parameters.data() + annmb_.num_para;

  neighbor_.initialize(paramb_.rc_radial_max, num_atoms_, paramb_.MN_radial);
  data_.NN_radial.resize(num_atoms_);
  data_.NL_radial.resize(static_cast<size_t>(num_atoms_) * paramb_.MN_radial);
  data_.NN_angular.resize(num_atoms_);
  data_.NL_angular.resize(static_cast<size_t>(num_atoms_) * paramb_.MN_angular);
  data_.Fp.resize(static_cast<size_t>(num_atoms_) * annmb_.dim);
  data_.sum_fxyz.resize(
    static_cast<size_t>(num_atoms_) * (paramb_.n_max_angular + 1) *
    ((paramb_.L_max + 1) * (paramb_.L_max + 1) - 1));

  potential_per_atom_.resize(num_atoms_);
  virial_per_atom_.resize(static_cast<size_t>(num_atoms_) * 9);
  B_size_per_atom_ = annmb_.num_neurons1 * (annmb_.dim + 2);
  B_projection_.resize(static_cast<size_t>(num_atoms_) * B_size_per_atom_);
}

void NEP_Extrapolation::update_potential(float* parameters)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb_.num_types; ++t) {
    annmb_.w0[t] = pointer;
    pointer += annmb_.num_neurons1 * annmb_.dim;
    annmb_.b0[t] = pointer;
    pointer += annmb_.num_neurons1;
    annmb_.w1[t] = pointer;
    pointer += annmb_.num_neurons1;
  }
  annmb_.b1 = pointer;
  annmb_.c_type_pair = parameters + annmb_.num_para_ann;
}


void NEP_Extrapolation::compute_large_box(Box& box, const GPU_Vector<double>& position)
{
  const int BLOCK_SIZE = 64;
  const int N = num_atoms_;
  const int grid_size = (N - 1) / BLOCK_SIZE + 1;

  neighbor_.find_neighbor_global(paramb_.rc_radial_max, box, type_, position);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    N,
    0,
    N,
    box,
    type_.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    neighbor_.NN.data(),
    neighbor_.NL.data(),
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data());
  GPU_CHECK_KERNEL

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    0,
    N,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    type_.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    potential_per_atom_.data(),
    data_.Fp.data(),
    virial_per_atom_.data(),
    data_.sum_fxyz.data(),
    true,
    B_projection_.data(),
    B_size_per_atom_);
  GPU_CHECK_KERNEL
}

void NEP_Extrapolation::compute_small_box(Box& box, const GPU_Vector<double>& position)
{
  const int BLOCK_SIZE = 64;
  const int N = num_atoms_;
  const int grid_size = (N - 1) / BLOCK_SIZE + 1;
  const int big_neighbor_size = 2000;
  const int size_x12 = N * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    N,
    0,
    N,
    box,
    ebox_,
    type_.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    small_box_data_.NN_radial.data(),
    small_box_data_.NL_radial.data(),
    small_box_data_.NN_angular.data(),
    small_box_data_.NL_angular.data(),
    small_box_data_.r12.data(),
    small_box_data_.r12.data() + size_x12,
    small_box_data_.r12.data() + size_x12 * 2,
    small_box_data_.r12.data() + size_x12 * 3,
    small_box_data_.r12.data() + size_x12 * 4,
    small_box_data_.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    0,
    N,
    small_box_data_.NN_radial.data(),
    small_box_data_.NL_radial.data(),
    small_box_data_.NN_angular.data(),
    small_box_data_.NL_angular.data(),
    type_.data(),
    small_box_data_.r12.data(),
    small_box_data_.r12.data() + size_x12,
    small_box_data_.r12.data() + size_x12 * 2,
    small_box_data_.r12.data() + size_x12 * 3,
    small_box_data_.r12.data() + size_x12 * 4,
    small_box_data_.r12.data() + size_x12 * 5,
    potential_per_atom_.data(),
    data_.Fp.data(),
    virial_per_atom_.data(),
    data_.sum_fxyz.data(),
    true,
    B_projection_.data(),
    B_size_per_atom_);
  GPU_CHECK_KERNEL
}

void NEP_Extrapolation::compute(Box& box, const GPU_Vector<double>& position)
{
  potential_per_atom_.fill(0.0);
  virial_per_atom_.fill(0.0);

  const bool is_small_box = get_expanded_box(paramb_.rc_radial_max, box, ebox_);
  if (is_small_box) {
    if (small_box_data_.NN_radial.size() != static_cast<size_t>(num_atoms_)) {
      const int big_neighbor_size = 2000;
      const int size_x12 = num_atoms_ * big_neighbor_size;
      small_box_data_.NN_radial.resize(num_atoms_);
      small_box_data_.NL_radial.resize(size_x12);
      small_box_data_.NN_angular.resize(num_atoms_);
      small_box_data_.NL_angular.resize(size_x12);
      small_box_data_.r12.resize(static_cast<size_t>(size_x12) * 6);
    }
    compute_small_box(box, position);
  } else {
    compute_large_box(box, position);
  }
}
