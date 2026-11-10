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
Charge NEP (qNEP) model of replica dynamics: a copy of
force/nep_charge/nep_charge.cu with all launches on a given stream and the
candidate neighbor list provided by the derived class, kept separate from
NEP_Charge so that ordinary qNEP force evaluation is not affected by replica
dynamics. DFTD3, BEC and the per-atom PPPM virial are left out: a replica run
accepts none of the commands that use them. Keep it in sync with
force/nep_charge by hand.

Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "replica_many_body.cuh"
#include "replica_nep_charge_model.cuh"
#include "replica_nep_charge_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_io/nep_model_file.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

void NEP_Charge_Replica_Model::check_ewald_pppm(const RunInput& run_input)
{
  // With alpha = pi / rc_radial, the default spacing keeps alpha * spacing at or below pi / 6.
  pppm_spacing = std::fmin(1.0, paramb.rc_radial / 6.0);
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty() && tokens[0] == "kspace") {
      if (tokens.size() < 2 || tokens.size() > 3) {
        PRINT_INPUT_ERROR("kspace requires ewald or pppm [spacing].");
      }
      if (tokens[1] == "ewald") {
        if (tokens.size() != 2) {
          PRINT_INPUT_ERROR("kspace ewald does not accept spacing.");
        }
        use_pppm = false;
      } else if (tokens[1] == "pppm") {
        use_pppm = true;
        if (tokens.size() == 3 &&
            (!is_valid_real(tokens[2], &pppm_spacing) ||
             !(pppm_spacing >= 0.2 && pppm_spacing <= 2.0))) {
          PRINT_INPUT_ERROR("PPPM spacing must be a finite number between 0.2 and 2.0 A.");
        }
      } else {
        std::cout << "kspace method can only be ewald or pppm\n";
        exit(1);
      }
    }
  }
}

NEP_Charge_Replica_Model::NEP_Charge_Replica_Model(
  const char* file_potential,
  const int num_atoms,
  const RunInput& run_input,
  const bool verbose)
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
  if (model.name == "qnep1") {
    paramb.charge_mode = 1;
  } else if (model.name == "qnep2") {
    paramb.charge_mode = 2;
  } else if (model.name == "qnep3") {
    paramb.charge_mode = 3;
  } else {
    PRINT_INPUT_ERROR("NEP_Charge requires model qnep1, qnep2 or qnep3.");
  }
  paramb.num_types = model.num_types;

  if (paramb.num_types == 1) {
    if (verbose)
      printf("Use the NEP4-Charge%d potential with %d atom type.\n",
        paramb.charge_mode, paramb.num_types);
  } else {
    if (verbose)
      printf("Use the NEP4-Charge%d potential with %d atom types.\n",
        paramb.charge_mode, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    zbl.atomic_numbers[n] = atomic_number(model.elements[n]);
    if (verbose)
      printf(
        "    type %d (%s with Z = %d).\n", n, model.elements[n].c_str(), zbl.atomic_numbers[n]);
  }

  zbl.enabled = model.enable_zbl;
  paramb.use_typewise_cutoff_zbl = model.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = model.typewise_cutoff_zbl_factor;
  if (zbl.enabled) {
    zbl.rc_inner = model.zbl_rc_inner;
    zbl.rc_outer = model.zbl_rc_outer;
    if (paramb.use_typewise_cutoff_zbl) {
      if (verbose)
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
    } else {
      if (verbose)
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
    }
  }

  paramb.rc_radial = model.rc_radial[0];
  paramb.rc_angular = model.rc_angular[0];
  if (verbose)
    printf("    radial cutoff = %g A.\n", paramb.rc_radial);
  if (verbose)
    printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  if (verbose)
    printf("    MN_radial = %d.\n", model.max_radial);
  paramb.MN_radial = int(ceil(model.max_radial * 1.25));
  paramb.MN_angular = int(ceil(model.max_angular * 1.25));
  if (verbose)
    printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  if (verbose)
    printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  paramb.n_max_radial = model.n_max_radial;
  paramb.n_max_angular = model.n_max_angular;
  paramb.basis_size_radial = model.basis_size_radial;
  paramb.basis_size_angular = model.basis_size_angular;
  if (verbose)
    printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  if (verbose)
    printf("    n_max_angular = %d.\n", paramb.n_max_angular);
  if (verbose)
    printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  if (verbose)
    printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  paramb.L_max = model.L_max;
  paramb.has_q_222 = 2 * model.has_q_222;
  paramb.has_q_1111 = model.has_q_1111;
  paramb.has_q_112 = model.has_q_112;
  paramb.has_q_123 = model.has_q_123;
  paramb.has_q_233 = model.has_q_233;
  paramb.has_q_134 = model.has_q_134;
  paramb.num_L = paramb.L_max + model.has_q_222 + model.has_q_1111 + model.has_q_112 +
                 model.has_q_123 + model.has_q_233 + model.has_q_134;
  paramb.dim_angular = model.layout.dim_angular;
  if (verbose)
    printf("    l_max_3body = %d.\n", paramb.L_max);
  if (verbose)
    printf("    has_q_222 = %d.\n", paramb.has_q_222);
  if (verbose)
    printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  if (verbose)
    printf("    has_q_112 = %d.\n", paramb.has_q_112);
  if (verbose)
    printf("    has_q_123 = %d.\n", paramb.has_q_123);
  if (verbose)
    printf("    has_q_233 = %d.\n", paramb.has_q_233);
  if (verbose)
    printf("    has_q_134 = %d.\n", paramb.has_q_134);

  annmb.num_neurons1 = model.num_neurons1;
  annmb.dim = model.layout.dim;
  if (verbose)
    printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_para_ann = model.layout.number_of_variables_ann;
  annmb.num_para = model.layout.number_of_variables;
  const int num_para_descriptor = model.layout.number_of_variables_descriptor;
  if (verbose)
    printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  if (verbose)
    printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  if (verbose)
    printf("    total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // Store ANN parameters, type-pair-major descriptor coefficients and scalers in one buffer.
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(model.parameters.data(), model.parameters.size());
  nep_data.parameters.copy_from_host(
    model.q_scaler.data(), model.q_scaler.size(), annmb.num_para);
  update_potential(nep_data.parameters.data(), annmb);
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;
  nep_data.energy_reference.resize(model.energy_reference.size());
  nep_data.energy_reference.copy_from_host(model.energy_reference.data());
  annmb.energy_reference = nep_data.energy_reference.data();

  // charge related parameters and data
  charge_para.alpha = float(PI) / paramb.rc_radial; // a good value
  check_ewald_pppm(run_input);
  if (use_pppm) {
    pppm.initialize(charge_para.alpha, pppm_spacing);
    pppm.set_verbose(verbose);
  } else {
    ewald.initialize(charge_para.alpha);
  }
  charge_para.two_alpha_over_sqrt_pi = 2.0f * charge_para.alpha / sqrt(float(PI));
  charge_para.A = erfc(float(PI)) / (paramb.rc_radial * paramb.rc_radial);
  charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-float(PI * PI)) / paramb.rc_radial;
  charge_para.B = - erfc(float(PI)) / paramb.rc_radial - charge_para.A * paramb.rc_radial;
  nep_data.D_real.resize(num_atoms);
  nep_data.charge.resize(num_atoms);
  nep_data.charge_derivative.resize(num_atoms * annmb.dim);

  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(num_atoms * annmb.dim);
  nep_data.sum_fxyz.resize(
    num_atoms * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);
}

NEP_Charge_Replica_Model::~NEP_Charge_Replica_Model(void)
{
  // nothing
}

void NEP_Charge_Replica_Model::update_potential(float* parameters, ANN& ann)
{
  const int num_outputs = 2;
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * num_outputs;
  }
  pointer += 1; // sqrt(epsilon_inf), used only for BEC
  ann.b1 = pointer;
  pointer += 1;

  ann.c_type_pair = pointer;
}

static __global__ void find_neighbor_list_large_box(
  NEP_Charge_Replica_Model::ParaMB paramb,
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
  int count_radial = 0;
  int count_angular = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    float rc_radial = paramb.rc_radial;
    float rc_angular = paramb.rc_angular;
    if (d12_square >= rc_radial * rc_radial) {
      continue;
    }
    g_NL_radial[count_radial++ * N + n1] = n2;
    if (d12_square < rc_angular * rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP_Charge_Replica_Model::ParaMB paramb,
  NEP_Charge_Replica_Model::ANN annmb,
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
  float* g_charge,
  float* g_charge_derivative,
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
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
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
        float rc = paramb.rc_angular;
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
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // normalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

      float F = 0.0f, Fp[MAX_DIM] = {0.0f};
      float charge = 0.0f;
      float charge_derivative[MAX_DIM] = {0.0f};

      apply_ann_one_layer_charge(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        charge,
        charge_derivative);

      g_pe[n1] += double(F) + annmb.energy_reference[t1];
      g_charge[n1] = charge;

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
        g_charge_derivative[d * N + n1] = charge_derivative[d] * annmb.q_scaler[d];
      }
  }
}

static __global__ void zero_total_charge(const int N, float* g_charge)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      charge += g_charge[n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_charge[n] -= s_charge[0] / N;
    }
  }
}


// Chain rule correction: zero_total_charge shifted q by -mean(q),
// so D_real must be shifted by -mean(D_real) for consistent forces.
// Uses double accumulator for numerical precision.
static __global__ void zero_mean_D_real(const int N, float* g_D_real)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      sum += (double)g_D_real[n];
    }
  }
  s_sum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  float mean_D = (float)(s_sum[0] / N);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_D_real[n] -= mean_D;
    }
  }
}

static __global__ void find_force_radial(
  NEP_Charge_Replica_Model::ParaMB paramb,
  NEP_Charge_Replica_Model::ANN annmb,
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
  const float* g_charge_derivative,
  const float* g_D_real,
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
      int n2 = g_NL[n1 + N * i1];
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
      float rc = paramb.rc_radial;
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
        float tmp12 = g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1];
        float tmp21 = g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * g_D_real[n2];
        tmp12 *= gnp12 * d12inv;
        tmp21 *= gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx += r12[0] * f21[0];
      s_syy += r12[1] * f21[1];
      s_szz += r12[2] * f21[2];
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
  NEP_Charge_Replica_Model::ParaMB paramb,
  NEP_Charge_Replica_Model::ANN annmb,
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
  const float* g_charge_derivative,
  const float* g_D_real,
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
      float tmp = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1]
        + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
      Fp[d] = tmp;
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
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
      float rc = paramb.rc_angular;
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

static __global__ void find_force_ZBL(
  NEP_Charge_Replica_Model::ParaMB paramb,
  const int N,
  const NEP_Charge_Replica_Model::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
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
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
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
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}

static __global__ void find_force_charge_real_space(
  const int N,
  const NEP_Charge_Replica_Model::Charge_Para charge_para,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  float* g_D_real)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
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
    float q1 = g_charge[n1];
    float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1; // self energy part
    float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi; // self energy part

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float q2 = g_charge[n2];
      float qq = q1 * q2;
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      s_pe += 0.5f * qq * erfc_r;
      float f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

static __global__ void find_charge_self_energy(
  const NEP_Charge_Replica_Model::Charge_Para charge_para,
  const int N1,
  const int N2,
  const float* g_charge,
  double* g_pe,
  float* g_D_real)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const float q1 = g_charge[n1];
    const float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1;
    const float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

void NEP_Charge_Replica_Model::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int* candidate_NN = nullptr;
  const int* candidate_NL = nullptr;
  update_candidate_neighbors(box, type, position_per_atom, candidate_NN, candidate_NL);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    candidate_NN,
    candidate_NL,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  if (num_calls_large_box_++ % 1000 == 0) {
    CHECK(gpuStreamSynchronize(stream_));
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    write_neighbor_info(
      num_calls_large_box_ - 1,
      1,
      paramb.MN_radial,
      paramb.MN_angular,
      nep_data.cpu_NN_radial.data(),
      nep_data.cpu_NN_angular.data(),
      N);
  }

  find_descriptor<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    nep_data.charge.data(),
    nep_data.charge_derivative.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  // enforce charge neutrality
  zero_total_charge<<<1, 1024, 0, stream_>>>(N, nep_data.charge.data());
  GPU_CHECK_KERNEL

  if (use_pppm) {
    pppm.find_force(
      N,
      N1,
      N2,
      box,
      nep_data.charge,
      position_per_atom,
      nep_data.D_real,
      force_per_atom,
      virial_per_atom,
      potential_per_atom);
  } else {
    ewald.find_force(
      N,
      N1,
      N2,
      box.cpu_h,
      nep_data.charge,
      position_per_atom,
      nep_data.D_real,
      force_per_atom,
      virial_per_atom,
      potential_per_atom);
  }

  if (paramb.charge_mode == 1) {
    find_force_charge_real_space<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
      N,
      charge_para,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  } else if (paramb.charge_mode == 3) {
    find_charge_self_energy<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
      charge_para,
      N1,
      N2,
      nep_data.charge.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  // Chain rule correction: D_real -= mean(D_real)
  zero_mean_D_real<<<1, 1024, 0, stream_>>>(N, nep_data.D_real.data());
  GPU_CHECK_KERNEL

  find_force_radial<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body_replica(
    box,
    N,
    N1,
    N2,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    position_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    stream_);

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP_Charge_Replica_Model::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int size_x12 = N * REPLICA_SMALL_BOX_NEIGHBORS;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  if (num_calls_small_box_++ % 1000 == 0) {
    CHECK(gpuStreamSynchronize(stream_));
    std::vector<int> cpu_NN_radial(N);
    std::vector<int> cpu_NN_angular(N);
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
    write_neighbor_info(
      num_calls_small_box_ - 1,
      1,
      paramb.MN_radial,
      paramb.MN_angular,
      cpu_NN_radial.data(),
      cpu_NN_angular.data(),
      N);
  }

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    nep_data.charge.data(),
    nep_data.charge_derivative.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  // enforce charge neutrality
  zero_total_charge<<<1, 1024, 0, stream_>>>(N, nep_data.charge.data());
  GPU_CHECK_KERNEL

  if (use_pppm) {
    pppm.find_force(
      N,
      N1,
      N2,
      box,
      nep_data.charge,
      position_per_atom,
      nep_data.D_real,
      force_per_atom,
      virial_per_atom,
      potential_per_atom);
  } else {
    ewald.find_force(
      N,
      N1,
      N2,
      box.cpu_h,
      nep_data.charge,
      position_per_atom,
      nep_data.D_real,
      force_per_atom,
      virial_per_atom,
      potential_per_atom);
  }

  if (paramb.charge_mode == 1) {
    find_force_charge_real_space_small_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
      N,
      charge_para,
      N1,
      N2,
      box,
      small_box_data.NN_radial.data(),
      small_box_data.NL_radial.data(),
      nep_data.charge.data(),
      small_box_data.r12.data(),
      small_box_data.r12.data() + size_x12,
      small_box_data.r12.data() + size_x12 * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  } else if (paramb.charge_mode == 3) {
    find_charge_self_energy<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
      charge_para,
      N1,
      N2,
      nep_data.charge.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  // Chain rule correction: D_real -= mean(D_real)
  zero_mean_D_real<<<1, 1024, 0, stream_>>>(N, nep_data.D_real.data());
  GPU_CHECK_KERNEL

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.sum_fxyz.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE, 0, stream_>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_Charge_Replica_Model::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  // The replica drivers accept qNEP only with periodic boundaries in all directions.
  if (get_expanded_box(paramb.rc_radial, box, ebox)) {
    small_box_data.resize(type.size());
    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  } else {
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}
