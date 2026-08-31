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

/*----------------------------------------------------------------------------80
NEP4 descriptor projection used by compute_extrapolation.
The descriptor and B-projection expressions are copied from the original NEP4
implementation. This class is independent of force/NEP.
------------------------------------------------------------------------------*/

#include "nep_extrapolation.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_parameters.cuh"
#include "utilities/nep_utilities.cuh"
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
          int c_index = (t1 * paramb.num_types + t2) *
                        ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1));
          c_index += n * (paramb.basis_size_radial + 1) + k;
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
          int c_index = paramb.num_c_radial;
          c_index += (t1 * paramb.num_types + t2) *
                     ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
          c_index += n * (paramb.basis_size_angular + 1) + k;
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

    // nomalize descriptor
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
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }

  bool has_zbl = false;
  if (tokens[0] == "nep4") {
  } else if (tokens[0] == "nep4_zbl") {
    has_zbl = true;
  } else if (tokens[0] == "nep4_dipole") {
  } else if (tokens[0] == "nep4_polarizability") {
    is_polarizability_ = true;
  } else {
    PRINT_INPUT_ERROR(
      "compute_extrapolation supports nep4, nep4_zbl, nep4_dipole, and nep4_polarizability models.");
  }

  paramb_.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb_.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb_.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }
  for (int t = 0; t < paramb_.num_types; ++t) {
    atom_types_[t] = tokens[2 + t];
  }

  std::vector<int> mapped_type(num_atoms_);
  for (int n = 0; n < num_atoms_; ++n) {
    mapped_type[n] = -1;
    for (int t = 0; t < paramb_.num_types; ++t) {
      if (atom.cpu_atom_symbol[n] == atom_types_[t]) {
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

  if (has_zbl) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
  }

  // cutoff
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != paramb_.num_types * 2 + 3) {
    std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
    exit(1);
  }
  if (tokens.size() == 5) {
    paramb_.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb_.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int n = 0; n < paramb_.num_types; ++n) {
      paramb_.rc_radial[n] = paramb_.rc_radial[0];
      paramb_.rc_angular[n] = paramb_.rc_angular[0];
    }
  } else {
    for (int n = 0; n < paramb_.num_types; ++n) {
      paramb_.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb_.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
    }
  }
  for (int n = 0; n < paramb_.num_types; ++n) {
    if (paramb_.rc_radial[n] > paramb_.rc_radial_max) {
      paramb_.rc_radial_max = paramb_.rc_radial[n];
    }
  }

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb_.MN_radial = int(ceil(MN_radial * 1.25));
  paramb_.MN_angular = int(ceil(MN_angular * 1.25));

  // n_max
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb_.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // basis_size
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb_.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    std::cout << "This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] [has_q_233] [has_q_134]." << std::endl;
    exit(1);
  }
  paramb_.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.num_L = paramb_.L_max;
  paramb_.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb_.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) {
    paramb_.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  if (tokens.size() >= 6) {
    paramb_.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  }
  if (tokens.size() >= 7) {
    paramb_.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  }
  if (tokens.size() >= 8) {
    paramb_.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  }
  if (paramb_.has_q_222) {
    paramb_.num_L += 1;
  }
  if (paramb_.has_q_1111) {
    paramb_.num_L += 1;
  }
  if (paramb_.has_q_112) {
    paramb_.num_L += 1;
  }
  if (paramb_.has_q_123) {
    paramb_.num_L += 1;
  }
  if (paramb_.has_q_233) {
    paramb_.num_L += 1;
  }
  if (paramb_.has_q_134) {
    paramb_.num_L += 1;
  }
  paramb_.dim_angular = (paramb_.n_max_angular + 1) * paramb_.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb_.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb_.dim = (paramb_.n_max_radial + 1) + paramb_.dim_angular;

  paramb_.num_types_sq = paramb_.num_types * paramb_.num_types;
  int num_para_ann_base = (annmb_.dim + 2) * annmb_.num_neurons1 * paramb_.num_types + 1;
  annmb_.num_para_ann = is_polarizability_ ? num_para_ann_base * 2 : num_para_ann_base;
  int num_para_descriptor =
    paramb_.num_types_sq * ((paramb_.n_max_radial + 1) * (paramb_.basis_size_radial + 1) +
                            (paramb_.n_max_angular + 1) * (paramb_.basis_size_angular + 1));
  annmb_.num_para = annmb_.num_para_ann + num_para_descriptor;
  paramb_.num_c_radial =
    paramb_.num_types_sq * (paramb_.n_max_radial + 1) * (paramb_.basis_size_radial + 1);

  std::vector<float> parameters(annmb_.num_para + annmb_.dim);
  for (int n = 0; n < annmb_.num_para + annmb_.dim; ++n) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected end or malformed parameter line in extrapolation NEP.");
    }
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  std::vector<float> descriptor_parameters = get_descriptor_parameters_type_pair(
    parameters,
    annmb_.num_para_ann,
    paramb_.num_types,
    paramb_.n_max_radial,
    paramb_.n_max_angular,
    paramb_.basis_size_radial,
    paramb_.basis_size_angular);
  data_.parameters.resize(annmb_.num_para + annmb_.dim);
  data_.parameters.copy_from_host(parameters.data());
  data_.descriptor_parameters_type_pair.resize(num_para_descriptor);
  data_.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());
  update_potential(data_.parameters.data());
  annmb_.c_type_pair = data_.descriptor_parameters_type_pair.data();
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
  pointer += 1;

  if (is_polarizability_) {
    for (int t = 0; t < paramb_.num_types; ++t) {
      pointer += annmb_.num_neurons1 * annmb_.dim;
      pointer += annmb_.num_neurons1;
      pointer += annmb_.num_neurons1;
    }
    pointer += 1;
  }

  annmb_.c = pointer;
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
