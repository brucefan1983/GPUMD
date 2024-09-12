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
#include "utilities/nep_utilities.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
    paramb.atomic_numbers[n] = atomic_number - 1;
    printf("        type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be zbl rc_inner rc_outer." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_float_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_float_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("        has the flexible ZBL potential\n");
    } else {
      printf(
        "        has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
        zbl.rc_inner,
        zbl.rc_outer);
    }
  }

  // cutoff 4.2 3.7 80 47
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != 8) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                 "[radial_factor] [angular_factor] [zbl_factor].\n";
    exit(1);
  }
  paramb.rc_radial = get_float_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_float_from_token(tokens[2], __FILE__, __LINE__);
  printf("        radial cutoff = %g A.\n", paramb.rc_radial);
  printf("        angular cutoff = %g A.\n", paramb.rc_angular);

  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  printf("        MN_radial = %d.\n", MN_radial);
  printf("        MN_angular = %d.\n", MN_angular);
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("        enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("        enlarged MN_angular = %d.\n", paramb.MN_angular);

  if (tokens.size() == 8) {
    paramb.typewise_cutoff_radial_factor = get_float_from_token(tokens[5], __FILE__, __LINE__);
    paramb.typewise_cutoff_angular_factor = get_float_from_token(tokens[6], __FILE__, __LINE__);
    paramb.typewise_cutoff_zbl_factor = get_float_from_token(tokens[7], __FILE__, __LINE__);
    if (paramb.typewise_cutoff_radial_factor > 0.0f) {
      paramb.use_typewise_cutoff = true;
    }
    if (paramb.typewise_cutoff_zbl_factor > 0.0f) {
      paramb.use_typewise_cutoff_zbl = true;
    }
  }

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
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_para =
    (annmb.dim + 2) * annmb.num_neurons1 * (paramb.version == 4 ? paramb.num_types : 1) + 1;
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
    parameters[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_parameters.resize(annmb.num_para);
  nep_parameters.copy_from_host(parameters.data());
  update_potential(nep_parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
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
    if (t > 0 && paramb.version != 4) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  ann.c = ann.b1 + 1;
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
      double rc = paramb.rc_radial;
      double rcinv = paramb.rcinv_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
        rcinv = 1.0f / rc;
      }
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
        double rc = paramb.rc_angular;
        double rcinv = paramb.rcinv_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
          rcinv = 1.0f / rc;
        }
        find_fc(rc, rcinv, d12, fc12);

        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
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
          rc_inner = rc_outer * 0.5f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      s_pe += f * 0.5f;
    }
    g_pe[n1] += s_pe;
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
  CUDA_CHECK_KERNEL

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
    CUDA_CHECK_KERNEL
  }
}
