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
NEP4 response models used by dipole and polarizability measurements.
------------------------------------------------------------------------------*/

#include "nep_response.cuh"
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

#include "nep_response_small_box.cuh"

NEP_Response::NEP_Response(const char* file_potential, const Atom& atom)
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
  if (tokens[0] == "nep4_dipole") {
    model_type_ = 1;
  } else if (tokens[0] == "nep4_polarizability") {
    model_type_ = 2;
  } else {
    PRINT_INPUT_ERROR("NEP_Response only supports nep4_dipole and nep4_polarizability models.");
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
      PRINT_INPUT_ERROR("There is atom in model.xyz that is not allowed in the response NEP.");
    }
  }
  type_.resize(num_atoms_);
  type_.copy_from_host(mapped_type.data());

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
  annmb_.num_para_ann = (annmb_.dim + 2) * annmb_.num_neurons1 * paramb_.num_types + 1;
  if (is_polarizability()) {
    annmb_.num_para_ann *= 2;
  }
  int num_para_descriptor =
    paramb_.num_types_sq * ((paramb_.n_max_radial + 1) * (paramb_.basis_size_radial + 1) +
                            (paramb_.n_max_angular + 1) * (paramb_.basis_size_angular + 1));
  annmb_.num_para = annmb_.num_para_ann + num_para_descriptor;

  paramb_.num_c_radial =
    paramb_.num_types_sq * (paramb_.n_max_radial + 1) * (paramb_.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb_.num_para + annmb_.dim);
  for (int n = 0; n < annmb_.num_para + annmb_.dim; ++n) {
    tokens = get_tokens(input);
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

  data_.f12x.resize(static_cast<size_t>(num_atoms_) * paramb_.MN_angular);
  data_.f12y.resize(static_cast<size_t>(num_atoms_) * paramb_.MN_angular);
  data_.f12z.resize(static_cast<size_t>(num_atoms_) * paramb_.MN_angular);
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
  force_per_atom_.resize(static_cast<size_t>(num_atoms_) * 3);
  virial_per_atom_.resize(static_cast<size_t>(num_atoms_) * 9);
}

bool NEP_Response::is_dipole() const { return model_type_ == 1; }

bool NEP_Response::is_polarizability() const { return model_type_ == 2; }

void NEP_Response::update_potential(float* parameters)
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

  if (is_polarizability()) {
    for (int t = 0; t < paramb_.num_types; ++t) {
      annmb_.w0_pol[t] = pointer;
      pointer += annmb_.num_neurons1 * annmb_.dim;
      annmb_.b0_pol[t] = pointer;
      pointer += annmb_.num_neurons1;
      annmb_.w1_pol[t] = pointer;
      pointer += annmb_.num_neurons1;
    }
    annmb_.b1_pol = pointer;
    pointer += 1;
  }

  annmb_.c = pointer;
}

static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    g_virial[n1 + 0 * N] = 0.0;
    g_virial[n1 + 1 * N] = 0.0;
    g_virial[n1 + 2 * N] = 0.0;
    g_virial[n1 + 3 * N] = 0.0;
    g_virial[n1 + 4 * N] = 0.0;
    g_virial[n1 + 5 * N] = 0.0;
    g_virial[n1 + 6 * N] = 0.0;
    g_virial[n1 + 7 * N] = 0.0;
    g_virial[n1 + 8 * N] = 0.0;
  }
}

static __global__ void find_neighbor_list_large_box(
  NEP_Response::ParaMB paramb,
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
  NEP_Response::ParaMB paramb,
  NEP_Response::ANN annmb,
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
  const bool is_polarizability,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
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

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (is_polarizability) {
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0_pol[t1],
        annmb.b0_pol[t1],
        annmb.w1_pol[t1],
        annmb.b1_pol,
        q,
        F,
        Fp);
      // Add the potential F for this atom to the diagonal of the virial
      g_virial[n1] = F;
      g_virial[n1 + N * 1] = F;
      g_virial[n1 + N * 2] = F;

      // Reset the potential and forces such that they
      // are zero for the next call to the model. The next call
      // is not used in the case of is_pol = True, but it doesn't
      // hurt to clean up.
      F = 0.0f;
      for (int d = 0; d < annmb.dim; ++d) {
        Fp[d] = 0.0f;
      }
    }

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
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  NEP_Response::ParaMB paramb,
  NEP_Response::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const bool is_dipole,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index_12 = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          int c_index_21 = get_c_index(
            t2 * paramb.num_types + t1, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index_12];
          gnp21 += fnp12[k] * annmb.c_type_pair[c_index_21];
        }
        float tmp12 = g_Fp[static_cast<size_t>(N) * n + n1] * gnp12 * d12inv;
        float tmp21 = g_Fp[static_cast<size_t>(N) * n + n2] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      if (is_dipole) {
        // The dipole is proportional to minus the sum of the virials times r12
        float r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_sxx -= r12_square * f21[0];
        s_syy -= r12_square * f21[1];
        s_szz -= r12_square * f21[2];
      } else {
        s_sxx += r12[0] * f21[0];
        s_syy += r12[1] * f21[1];
        s_szz += r12[2] * f21[2];
      }
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
  }
}

static __global__ void find_partial_force_angular(
  NEP_Response::ParaMB paramb,
  NEP_Response::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[static_cast<size_t>(N) * (paramb.n_max_radial + 1 + d) + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_properties_many_body_response(
  const bool is_dipole,
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f; // virial_stress_xx
  float s_sxy = 0.0f; // virial_stress_xy
  float s_sxz = 0.0f; // virial_stress_xz
  float s_syx = 0.0f; // virial_stress_yx
  float s_syy = 0.0f; // virial_stress_yy
  float s_syz = 0.0f; // virial_stress_yz
  float s_szx = 0.0f; // virial_stress_zx
  float s_szy = 0.0f; // virial_stress_zy
  float s_szz = 0.0f; // virial_stress_zz

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      // int offset = 0;

      int l = 0;
      int r = g_neighbor_number[n2];
      int m = 0;
      int tmp_value = 0;
      while (l < r) {
        m = (l + r) >> 1;
        tmp_value = g_neighbor_list[n2 + number_of_particles * m];
        if (tmp_value < n1) {
          l = m + 1;
        } else if (tmp_value > n1) {
          r = m - 1;
        } else {
          break;
        }
      }
      // for (int k = 0; k < neighbor_number_2; ++k) {
      //   if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
      //     offset = k;
      //     break;
      //   }
      // }
      index = ((l + r) >> 1) * number_of_particles + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
      if (is_dipole) {
        // Float version of the function
        // The dipole is proportional to minus the sum of the virials times r12
        float r12_square = x12 * x12 + y12 * y12 + z12 * z12;
        s_sxx -= r12_square * f21x;
        s_syy -= r12_square * f21y;
        s_szz -= r12_square * f21z;
      } else {
        s_sxx += x12 * f21x;
        s_syy += y12 * f21y;
        s_szz += z12 * f21z;
      }
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP_Response::ExpandedBox& ebox)
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

void NEP_Response::compute_large_box(Box& box, const GPU_Vector<double>& position)
{
  const int N = type_.size();
  const int BLOCK_SIZE = 64;
  const int N1 = 0;
  const int N2 = N;
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor_.find_neighbor_global(paramb_.rc_radial_max, box, type_, position);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    N,
    N1,
    N2,
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

  const bool is_polarizability_model = is_polarizability();
  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    N1,
    N2,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    type_.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    is_polarizability_model,
    potential_per_atom_.data(),
    data_.Fp.data(),
    virial_per_atom_.data(),
    data_.sum_fxyz.data());
  GPU_CHECK_KERNEL

  const bool is_dipole_model = is_dipole();
  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    N1,
    N2,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    type_.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    data_.Fp.data(),
    is_dipole_model,
    force_per_atom_.data(),
    force_per_atom_.data() + N,
    force_per_atom_.data() + N * 2,
    virial_per_atom_.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    N1,
    N2,
    box,
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    type_.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    data_.Fp.data(),
    data_.sum_fxyz.data(),
    data_.f12x.data(),
    data_.f12y.data(),
    data_.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body_response<<<grid_size, BLOCK_SIZE>>>(
    is_dipole_model,
    N,
    N1,
    N2,
    box,
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    data_.f12x.data(),
    data_.f12y.data(),
    data_.f12z.data(),
    position.data(),
    position.data() + N,
    position.data() + N * 2,
    force_per_atom_.data(),
    force_per_atom_.data() + N,
    force_per_atom_.data() + N * 2,
    virial_per_atom_.data());
  GPU_CHECK_KERNEL
}

void NEP_Response::compute_small_box(Box& box, const GPU_Vector<double>& position)
{
  const int BLOCK_SIZE = 64;
  const int N = type_.size();
  const int N1 = 0;
  const int N2 = N;
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type_.size() * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    N,
    N1,
    N2,
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

  const bool is_polarizability_model = is_polarizability();
  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    N1,
    N2,
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
    is_polarizability_model,
    potential_per_atom_.data(),
    data_.Fp.data(),
    virial_per_atom_.data(),
    data_.sum_fxyz.data());
  GPU_CHECK_KERNEL

  const bool is_dipole_model = is_dipole();
  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    N1,
    N2,
    small_box_data_.NN_radial.data(),
    small_box_data_.NL_radial.data(),
    type_.data(),
    small_box_data_.r12.data(),
    small_box_data_.r12.data() + size_x12,
    small_box_data_.r12.data() + size_x12 * 2,
    data_.Fp.data(),
    is_dipole_model,
    force_per_atom_.data(),
    force_per_atom_.data() + N,
    force_per_atom_.data() + N * 2,
    virial_per_atom_.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb_,
    annmb_,
    N,
    N1,
    N2,
    small_box_data_.NN_angular.data(),
    small_box_data_.NL_angular.data(),
    type_.data(),
    small_box_data_.r12.data() + size_x12 * 3,
    small_box_data_.r12.data() + size_x12 * 4,
    small_box_data_.r12.data() + size_x12 * 5,
    data_.Fp.data(),
    data_.sum_fxyz.data(),
    is_dipole_model,
    force_per_atom_.data(),
    force_per_atom_.data() + N,
    force_per_atom_.data() + N * 2,
    virial_per_atom_.data());
  GPU_CHECK_KERNEL
}

const GPU_Vector<double>& NEP_Response::compute(Box& box, const GPU_Vector<double>& position)
{
  const int N = type_.size();
  if (position.size() != N * 3) {
    PRINT_INPUT_ERROR("The number of atoms changed after initializing the dipole/polarizability NEP.");
  }

  initialize_properties<<<(N - 1) / 128 + 1, 128>>>(
    N,
    force_per_atom_.data(),
    force_per_atom_.data() + N,
    force_per_atom_.data() + N * 2,
    potential_per_atom_.data(),
    virial_per_atom_.data());
  GPU_CHECK_KERNEL

  const bool is_small_box = get_expanded_box(paramb_.rc_radial_max, box, ebox_);
  if (is_small_box) {
    const int current_num_atoms = type_.size();
    if (small_box_data_.NN_radial.size() != current_num_atoms) {
      const int big_neighbor_size = 2000;
      const int size_x12 = current_num_atoms * big_neighbor_size;

      small_box_data_.NN_radial.resize(current_num_atoms);
      small_box_data_.NL_radial.resize(size_x12);
      small_box_data_.NN_angular.resize(current_num_atoms);
      small_box_data_.NL_angular.resize(size_x12);
      small_box_data_.r12.resize(size_x12 * 6);
    }

    compute_small_box(box, position);
  } else {
    compute_large_box(box, position);
  }

  return virial_per_atom_;
}
