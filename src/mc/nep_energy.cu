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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep_energy.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_io/nep_model_file.cuh"
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
    const std::string error = "Cannot open potential file " + std::string(file_potential) + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  NEP_Model_File model;
  std::string error;
  if (!model.read(input, error) || !model.validate_md(error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.name != "nep") {
    PRINT_INPUT_ERROR("NEP_Energy requires model nep.");
  }
  energy_reference = model.energy_reference;
  paramb.num_types = model.num_types;

  if (paramb.num_types == 1) {
    printf("    Use the NEP4 potential with %d atom type.\n", paramb.num_types);
  } else {
    printf("    Use the NEP4 potential with %d atom types.\n", paramb.num_types);
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (model.elements[n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("        type %d (%s with Z = %d).\n", n, model.elements[n].c_str(), zbl.atomic_numbers[n]);
  }

  zbl.enabled = model.enable_zbl;
  zbl.flexible = model.flexible_zbl;
  paramb.use_typewise_cutoff_zbl = model.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = model.typewise_cutoff_zbl_factor;
  if (zbl.enabled) {
    zbl.rc_inner = model.zbl_rc_inner;
    zbl.rc_outer = model.zbl_rc_outer;
    if (zbl.flexible) {
      printf("    has the flexible ZBL potential\n");
    } else if (paramb.use_typewise_cutoff_zbl) {
      printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
        paramb.typewise_cutoff_zbl_factor);
    } else {
      printf(
        "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
        zbl.rc_inner,
        zbl.rc_outer);
    }
  }

  if (model.has_multiple_cutoffs) {
    printf("    cutoff = \n");
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    paramb.rc_radial[n] = model.rc_radial[n];
    paramb.rc_angular[n] = model.rc_angular[n];
    if (model.has_multiple_cutoffs) {
      printf("    (%g A, %g A)\n", paramb.rc_radial[n], paramb.rc_angular[n]);
    }
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
    if (paramb.rc_angular[n] > paramb.rc_angular_max) {
      paramb.rc_angular_max = paramb.rc_angular[n];
    }
  }
  if (!model.has_multiple_cutoffs) {
    printf("    radial cutoff = %g A.\n", paramb.rc_radial[0]);
    printf("    angular cutoff = %g A.\n", paramb.rc_angular[0]);
  }

  printf("    MN_radial = %d.\n", model.max_radial);
  printf("    MN_angular = %d.\n", model.max_angular);
  paramb.MN_radial = int(ceil(model.max_radial * 1.25));
  paramb.MN_angular = int(ceil(model.max_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  paramb.n_max_radial = model.n_max_radial;
  paramb.n_max_angular = model.n_max_angular;
  paramb.basis_size_radial = model.basis_size_radial;
  paramb.basis_size_angular = model.basis_size_angular;
  printf("        n_max_radial = %d.\n", paramb.n_max_radial);
  printf("        n_max_angular = %d.\n", paramb.n_max_angular);
  printf("        basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("        basis_size_angular = %d.\n", paramb.basis_size_angular);

  paramb.L_max = model.L_max;
  paramb.has_q_222 = 2 * model.has_q_222;
  paramb.has_q_1111 = model.has_q_1111;
  paramb.has_q_112 = model.has_q_112;
  paramb.has_q_123 = model.has_q_123;
  paramb.has_q_233 = model.has_q_233;
  paramb.has_q_134 = model.has_q_134;
  printf("    l_max_3body = %d.\n", paramb.L_max);
  printf("    has_q_222 = %d.\n", paramb.has_q_222);
  printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  printf("    has_q_112 = %d.\n", paramb.has_q_112);
  printf("    has_q_123 = %d.\n", paramb.has_q_123);
  printf("    has_q_233 = %d.\n", paramb.has_q_233);
  printf("    has_q_134 = %d.\n", paramb.has_q_134);

  annmb.num_neurons1 = model.num_neurons1;
  annmb.dim = model.layout.dim;
  printf("        ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  annmb.num_para = model.layout.number_of_variables;
  printf("        number of neural network parameters = %d.\n", model.layout.number_of_variables_ann);
  printf("        number of descriptor parameters = %d.\n", model.layout.number_of_variables_descriptor);
  printf("        total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // Descriptor coefficients already use the type-pair-major layout required by MC.
  nep_parameters.resize(annmb.num_para);
  nep_parameters.copy_from_host(model.parameters.data());
  update_potential(nep_parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    paramb.q_scaler[d] = model.q_scaler[d];
  }

  if (zbl.flexible) {
    const int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      zbl.para[d] = model.zbl_parameters[d];
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
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
    }

    // normalize descriptor
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
      if (zbl.flexible) {
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
