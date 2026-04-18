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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep_energy.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP_Energy::initialize(const char* file_potential)
{

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep3") {
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.version = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4") {
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep5") {
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.version = 5;
    zbl.enabled = true;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP3 and NEP4 models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("    Use the NEP%d potential with %d atom type.\n", paramb.version, paramb.num_types);
  } else {
    printf("    Use the NEP%d potential with %d atom types.\n", paramb.version, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("        type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  // cutoff
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
    std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
    exit(1);
  }
  if (tokens.size() == 5) {
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
    printf("    radial cutoff = %g A.\n", paramb.rc_radial[0]);
    printf("    angular cutoff = %g A.\n", paramb.rc_angular[0]);
  } else {
    printf("    cutoff = \n");
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
      printf("    (%g A, %g A)\n", paramb.rc_radial[n], paramb.rc_angular[n]);
    }
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
    if (paramb.rc_angular[n] > paramb.rc_angular_max) {
      paramb.rc_angular_max = paramb.rc_angular[n];
    }
  }

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  printf("    MN_radial = %d.\n", MN_radial);
  printf("    MN_angular = %d.\n", MN_angular);
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("        n_max_radial = %d.\n", paramb.n_max_radial);
  printf("        n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("        basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("        basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() != 4) {
    std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("        l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  printf("        l_max_4body = %d.\n", L_max_4body);
  printf("        l_max_5body = %d.\n", L_max_5body);
  if (L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  printf("        ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  if (paramb.version == 3) {
    annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else {
    annmb.num_para = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }

  printf("        number of neural network parameters = %d.\n", annmb.num_para);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("        number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para += num_para_descriptor;
  printf("        total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_parameters.resize(annmb.num_para);
  nep_parameters.copy_from_host(parameters.data());
  update_potential(nep_parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
}

NEP_Energy::NEP_Energy(void)
{
  // nothing
}

NEP_Energy::~NEP_Energy(void)
{
  // nothing
}

void NEP_Energy::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }
  ann.b1 = pointer;
  ann.c = ann.b1 + 1;
}

static __device__ __forceinline__ void apply_ann_one_layer_energy_only(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  const float* q,
  float& energy)
{
  for (int n = 0; n < N_neu; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    energy += w1[n] * tanh(w0_times_q - b0[n]);
  }
  energy -= b1[0];
}

static __device__ __forceinline__ void apply_ann_one_layer_nep5_energy_only(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  const float* q,
  float& energy)
{
  for (int n = 0; n < N_neu; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    energy += w1[n] * tanh(w0_times_q - b0[n]);
  }
  energy -= w1[N_neu] + b1[0];
}

static __device__ __forceinline__ void accumulate_radial_descriptor_contribution(
  const NEP_Energy::ParaMB& paramb,
  const NEP_Energy::ANN& annmb,
  const int t1,
  const int t2,
  const float d12,
  float* q_primary,
  float* q_secondary)
{
  float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
  if (d12 >= rc) {
    return;
  }

  float rcinv = 1.0f / rc;
  float fc12;
  find_fc(rc, rcinv, d12, fc12);

  float fn12[MAX_NUM_N];
  find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    float gn12 = 0.0f;
    for (int k = 0; k <= paramb.basis_size_radial; ++k) {
      int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
      c_index += t1 * paramb.num_types + t2;
      gn12 += fn12[k] * annmb.c[c_index];
    }
    q_primary[n] += gn12;
    if (q_secondary != nullptr) {
      q_secondary[n] += gn12;
    }
  }
}

template <int L_MAX>
static __device__ __forceinline__ void accumulate_angular_descriptor_contribution_lmax(
  const NEP_Energy::ParaMB& paramb,
  const NEP_Energy::ANN& annmb,
  const int n,
  const int t1,
  const int t2,
  const float x12,
  const float y12,
  const float z12,
  const float d12,
  float* s_primary,
  float* s_secondary)
{
  float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
  if (d12 >= rc) {
    return;
  }

  float rcinv = 1.0f / rc;
  float fc12;
  find_fc(rc, rcinv, d12, fc12);

  float fn12[MAX_NUM_N];
  find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
  float gn12 = 0.0f;
  for (int k = 0; k <= paramb.basis_size_angular; ++k) {
    int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
    c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
    gn12 += fn12[k] * annmb.c[c_index];
  }
  accumulate_s(L_MAX, d12, x12, y12, z12, gn12, s_primary);
  if (s_secondary != nullptr) {
    accumulate_s(L_MAX, d12, x12, y12, z12, gn12, s_secondary);
  }
}

template <int L_MAX>
static __global__ void find_energy_nep_dual_lmax(
  NEP_Energy::ParaMB paramb,
  NEP_Energy::ANN annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NN_angular,
  const int* __restrict__ g_type_before,
  const int* __restrict__ g_type_after,
  const int* __restrict__ g_t2_radial_before,
  const int* __restrict__ g_t2_radial_after,
  const int* __restrict__ g_t2_angular_before,
  const int* __restrict__ g_t2_angular_after,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  float* g_pe_before,
  float* g_pe_after)
{
  constexpr int NUM_OF_ABC_LMAX = L_MAX * (L_MAX + 2);

  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1_before = g_type_before[n1];
    int t1_after = g_type_after[n1];
    float q_before[MAX_DIM] = {0.0f};
    float q_after[MAX_DIM] = {0.0f};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      float x12 = g_x12_radial[index];
      float y12 = g_y12_radial[index];
      float z12 = g_z12_radial[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      int t2_before = g_t2_radial_before[index];
      int t2_after = g_t2_radial_after[index];

      if (t1_before == t1_after && t2_before == t2_after) {
        accumulate_radial_descriptor_contribution(
          paramb, annmb, t1_before, t2_before, d12, q_before, q_after);
      } else {
        accumulate_radial_descriptor_contribution(
          paramb, annmb, t1_before, t2_before, d12, q_before, nullptr);
        accumulate_radial_descriptor_contribution(
          paramb, annmb, t1_after, t2_after, d12, q_after, nullptr);
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s_before[NUM_OF_ABC_LMAX] = {0.0f};
      float s_after[NUM_OF_ABC_LMAX] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        float x12 = g_x12_angular[index];
        float y12 = g_y12_angular[index];
        float z12 = g_z12_angular[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        int t2_before = g_t2_angular_before[index];
        int t2_after = g_t2_angular_after[index];

        if (t1_before == t1_after && t2_before == t2_after) {
          accumulate_angular_descriptor_contribution_lmax<L_MAX>(
            paramb, annmb, n, t1_before, t2_before, x12, y12, z12, d12, s_before, s_after);
        } else {
          accumulate_angular_descriptor_contribution_lmax<L_MAX>(
            paramb, annmb, n, t1_before, t2_before, x12, y12, z12, d12, s_before, nullptr);
          accumulate_angular_descriptor_contribution_lmax<L_MAX>(
            paramb, annmb, n, t1_after, t2_after, x12, y12, z12, d12, s_after, nullptr);
        }
      }
      find_q(L_MAX, paramb.num_L, paramb.n_max_angular + 1, n, s_before, q_before + (paramb.n_max_radial + 1));
      find_q(L_MAX, paramb.num_L, paramb.n_max_angular + 1, n, s_after, q_after + (paramb.n_max_radial + 1));
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q_before[d] = q_before[d] * paramb.q_scaler[d];
      q_after[d] = q_after[d] * paramb.q_scaler[d];
    }

    float energy_before = 0.0f;
    float energy_after = 0.0f;
    if (paramb.version == 5) {
      apply_ann_one_layer_nep5_energy_only(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1_before],
        annmb.b0[t1_before],
        annmb.w1[t1_before],
        annmb.b1,
        q_before,
        energy_before);
      apply_ann_one_layer_nep5_energy_only(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1_after],
        annmb.b0[t1_after],
        annmb.w1[t1_after],
        annmb.b1,
        q_after,
        energy_after);
    } else {
      apply_ann_one_layer_energy_only(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1_before],
        annmb.b0[t1_before],
        annmb.w1[t1_before],
        annmb.b1,
        q_before,
        energy_before);
      apply_ann_one_layer_energy_only(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1_after],
        annmb.b0[t1_after],
        annmb.w1[t1_after],
        annmb.b1,
        q_after,
        energy_after);
    }
    g_pe_before[n1] = energy_before;
    g_pe_after[n1] = energy_after;
  }
}

static __global__ void find_energy_nep(
  NEP_Energy::ParaMB paramb,
  NEP_Energy::ANN annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NN_angular,
  const int* __restrict__ g_type,
  const int* __restrict__ g_t2_radial,
  const int* __restrict__ g_t2_angular,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  float* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12;
      int t2 = g_t2_radial[index];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);

      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fc12;
        int t2 = g_t2_angular[index];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);

        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
    } else {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
    }
    g_pe[n1] = F;
  }
}

static __global__ void find_energy_zbl(
  const int N,
  const NEP_Energy::ParaMB paramb,
  const NEP_Energy::ZBL zbl,
  const int* g_NN,
  const int* __restrict__ g_type,
  const int* g_t2_angular,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_t2_angular[index];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      s_pe += f * 0.5f;
    }
    g_pe[n1] += s_pe;
  }
}

static __global__ void find_energy_zbl_dual(
  const int N,
  const NEP_Energy::ParaMB paramb,
  const NEP_Energy::ZBL zbl,
  const int* g_NN,
  const int* __restrict__ g_type_before,
  const int* __restrict__ g_type_after,
  const int* g_t2_angular_before,
  const int* g_t2_angular_after,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_pe_before,
  float* g_pe_after)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float s_pe_before = 0.0f;
    float s_pe_after = 0.0f;
    int type1_before = g_type_before[n1];
    int type1_after = g_type_after[n1];
    int zi_before = zbl.atomic_numbers[type1_before];
    int zi_after = zbl.atomic_numbers[type1_after];
    float pow_zi_before = pow(float(zi_before), 0.23f);
    float pow_zi_after = pow(float(zi_after), 0.23f);

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float d12inv = 1.0f / d12;
      int type2_before = g_t2_angular_before[index];
      int type2_after = g_t2_angular_after[index];

      if (type1_before == type1_after && type2_before == type2_after) {
        float f, fp;
        int zj = zbl.atomic_numbers[type2_before];
        float a_inv = (pow_zi_before + pow(float(zj), 0.23f)) * 2.134563f;
        float zizj = K_C_SP * zi_before * zj;
        if (zbl.flexibled) {
          int t1, t2;
          if (type1_before < type2_before) {
            t1 = type1_before;
            t2 = type2_before;
          } else {
            t1 = type2_before;
            t2 = type1_before;
          }
          int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
          float ZBL_para[10];
          for (int i = 0; i < 10; ++i) {
            ZBL_para[i] = zbl.para[10 * zbl_index + i];
          }
          find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
        } else {
          float rc_inner = zbl.rc_inner;
          float rc_outer = zbl.rc_outer;
          if (paramb.use_typewise_cutoff_zbl) {
            rc_outer = min(
              (COVALENT_RADIUS[zi_before - 1] + COVALENT_RADIUS[zj - 1]) *
                paramb.typewise_cutoff_zbl_factor,
              rc_outer);
            rc_inner = 0.0f;
          }
          find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
        }
        float pair_energy = f * 0.5f;
        s_pe_before += pair_energy;
        s_pe_after += pair_energy;
      } else {
        float f_before, fp_before;
        int zj_before = zbl.atomic_numbers[type2_before];
        float a_inv_before = (pow_zi_before + pow(float(zj_before), 0.23f)) * 2.134563f;
        float zizj_before = K_C_SP * zi_before * zj_before;
        if (zbl.flexibled) {
          int t1, t2;
          if (type1_before < type2_before) {
            t1 = type1_before;
            t2 = type2_before;
          } else {
            t1 = type2_before;
            t2 = type1_before;
          }
          int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
          float ZBL_para[10];
          for (int i = 0; i < 10; ++i) {
            ZBL_para[i] = zbl.para[10 * zbl_index + i];
          }
          find_f_and_fp_zbl(ZBL_para, zizj_before, a_inv_before, d12, d12inv, f_before, fp_before);
        } else {
          float rc_inner = zbl.rc_inner;
          float rc_outer = zbl.rc_outer;
          if (paramb.use_typewise_cutoff_zbl) {
            rc_outer = min(
              (COVALENT_RADIUS[zi_before - 1] + COVALENT_RADIUS[zj_before - 1]) *
                paramb.typewise_cutoff_zbl_factor,
              rc_outer);
            rc_inner = 0.0f;
          }
          find_f_and_fp_zbl(
            zizj_before, a_inv_before, rc_inner, rc_outer, d12, d12inv, f_before, fp_before);
        }
        s_pe_before += f_before * 0.5f;

        float f_after, fp_after;
        int zj_after = zbl.atomic_numbers[type2_after];
        float a_inv_after = (pow_zi_after + pow(float(zj_after), 0.23f)) * 2.134563f;
        float zizj_after = K_C_SP * zi_after * zj_after;
        if (zbl.flexibled) {
          int t1, t2;
          if (type1_after < type2_after) {
            t1 = type1_after;
            t2 = type2_after;
          } else {
            t1 = type2_after;
            t2 = type1_after;
          }
          int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
          float ZBL_para[10];
          for (int i = 0; i < 10; ++i) {
            ZBL_para[i] = zbl.para[10 * zbl_index + i];
          }
          find_f_and_fp_zbl(ZBL_para, zizj_after, a_inv_after, d12, d12inv, f_after, fp_after);
        } else {
          float rc_inner = zbl.rc_inner;
          float rc_outer = zbl.rc_outer;
          if (paramb.use_typewise_cutoff_zbl) {
            rc_outer = min(
              (COVALENT_RADIUS[zi_after - 1] + COVALENT_RADIUS[zj_after - 1]) *
                paramb.typewise_cutoff_zbl_factor,
              rc_outer);
            rc_inner = 0.0f;
          }
          find_f_and_fp_zbl(
            zizj_after, a_inv_after, rc_inner, rc_outer, d12, d12inv, f_after, fp_after);
        }
        s_pe_after += f_after * 0.5f;
      }
    }

    g_pe_before[n1] += s_pe_before;
    g_pe_after[n1] += s_pe_after;
  }
}

void NEP_Energy::find_energy(
  const int N,
  const int* g_NN_radial,
  const int* g_NN_angular,
  const int* g_type,
  const int* g_t2_radial,
  const int* g_t2_angular,
  const float* g_x12_radial,
  const float* g_y12_radial,
  const float* g_z12_radial,
  const float* g_x12_angular,
  const float* g_y12_angular,
  const float* g_z12_angular,
  float* g_pe)
{
  find_energy_nep<<<(N - 1) / 64 + 1, 64>>>(
    paramb,
    annmb,
    N,
    g_NN_radial,
    g_NN_angular,
    g_type,
    g_t2_radial,
    g_t2_angular,
    g_x12_radial,
    g_y12_radial,
    g_z12_radial,
    g_x12_angular,
    g_y12_angular,
    g_z12_angular,
    g_pe);
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_energy_zbl<<<(N - 1) / 64 + 1, 64>>>(
      N,
      paramb,
      zbl,
      g_NN_angular,
      g_type,
      g_t2_angular,
      g_x12_angular,
      g_y12_angular,
      g_z12_angular,
      g_pe);
    GPU_CHECK_KERNEL
  }
}

void NEP_Energy::find_energy_dual(
  const int N,
  const int* g_NN_radial,
  const int* g_NN_angular,
  const int* g_type_before,
  const int* g_type_after,
  const int* g_t2_radial_before,
  const int* g_t2_radial_after,
  const int* g_t2_angular_before,
  const int* g_t2_angular_after,
  const float* g_x12_radial,
  const float* g_y12_radial,
  const float* g_z12_radial,
  const float* g_x12_angular,
  const float* g_y12_angular,
  const float* g_z12_angular,
  float* g_pe_before,
  float* g_pe_after)
{
  switch (paramb.L_max) {
    case 1:
      find_energy_nep_dual_lmax<1><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 2:
      find_energy_nep_dual_lmax<2><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 3:
      find_energy_nep_dual_lmax<3><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 4:
      find_energy_nep_dual_lmax<4><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 5:
      find_energy_nep_dual_lmax<5><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 6:
      find_energy_nep_dual_lmax<6><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 7:
      find_energy_nep_dual_lmax<7><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
    case 8:
    default:
      find_energy_nep_dual_lmax<8><<<(N - 1) / 64 + 1, 64>>>(
        paramb,
        annmb,
        N,
        g_NN_radial,
        g_NN_angular,
        g_type_before,
        g_type_after,
        g_t2_radial_before,
        g_t2_radial_after,
        g_t2_angular_before,
        g_t2_angular_after,
        g_x12_radial,
        g_y12_radial,
        g_z12_radial,
        g_x12_angular,
        g_y12_angular,
        g_z12_angular,
        g_pe_before,
        g_pe_after);
      break;
  }
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_energy_zbl_dual<<<(N - 1) / 64 + 1, 64>>>(
      N,
      paramb,
      zbl,
      g_NN_angular,
      g_type_before,
      g_type_after,
      g_t2_angular_before,
      g_t2_angular_after,
      g_x12_angular,
      g_y12_angular,
      g_z12_angular,
      g_pe_before,
      g_pe_after);
    GPU_CHECK_KERNEL
  }
}
