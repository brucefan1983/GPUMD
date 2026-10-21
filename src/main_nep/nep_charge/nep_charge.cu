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

#include "main_nep/dataset.cuh"
#include "nep_charge.cuh"
#include "main_nep/common/nep_output.cuh"
#include "main_nep/common/nep_charge_reader.cuh"
#include "main_nep/common/nep_model.cuh"
#include "main_nep/common/nep_init.cuh"
#include "main_nep/common/nep_io.cuh"
#include "main_nep/common/nep_long_range.cuh"
#include "main_nep/common/nep_charge_parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstring>
#include <algorithm>

static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
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
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = g_NN_sum[n1] + i1;
        int n2 = g_NL[index];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
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
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n, s, q);
      int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
      for (int abc = 0; abc < num_abc; ++abc) {
        g_sum_fxyz[(n * num_abc + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

void NEP_Charge::initialize(NEP_Charge_Parameters& para, int N, int Nc, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    initialize_device(para, N, Nc, device_id);
  }
  initialize_compiled_kernel(para);
}

void NEP_Charge::initialize_model(const NEP_Charge_Parameters& para)
{
  paramb.charge_mode = para.charge_mode;
  paramb.version = para.version;
  paramb.rc_radial = para.rc_radial[0];
  paramb.rc_angular = para.rc_angular[0];
  paramb.num_types = para.num_types;
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.has_q_222 = para.has_q_222;
  paramb.has_q_1111 = para.has_q_1111;
  paramb.has_q_112 = para.has_q_112;
  paramb.has_q_123 = para.has_q_123;
  paramb.has_q_233 = para.has_q_233;
  paramb.has_q_134 = para.has_q_134;
  paramb.num_L = para.dim_angular / (para.n_max_angular + 1);
  paramb.dim_angular = para.dim_angular;

  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

  initialize_nep_zbl(para, zbl);

  charge_para.alpha = float(PI) / paramb.rc_radial; // a good value
  charge_para.two_alpha_over_sqrt_pi = 2.0f * charge_para.alpha / sqrt(float(PI));
  charge_para.alpha_factor = 0.25f / (charge_para.alpha * charge_para.alpha);
}

void NEP_Charge::initialize_device(const NEP_Charge_Parameters& para, int N, int Nc, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  annmb[device_id].dim = para.dim;
  annmb[device_id].num_neurons1 = para.num_neurons1;
  annmb[device_id].num_para = para.number_of_variables;

  nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
  nep_data[device_id].charge_derivative.resize(N * annmb[device_id].dim);
  nep_data[device_id].Fp.resize(N * annmb[device_id].dim);
  nep_data[device_id].sum_fxyz.resize(
    N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data[device_id].parameters.resize(annmb[device_id].num_para);
  nep_data[device_id].D_real.resize(N);
  nep_data[device_id].kpoint_offset.resize(Nc + 1);
}

namespace
{
NEP_Compile_Config make_compile_config(
  const NEP_Charge_Parameters& para)
{
#ifdef USE_CJ
  warning_nep_compile(
    "qNEP, charge-vdW, and TNEP specialization is not available in "
    "this USE_CJ configuration.");
  return NEP_Compile_Config();
#endif
  NEP_Compile_Config config = make_nep_compile_config(para);
  config.model_name = "qNEP";
  config.source_file = "main_nep/nep_charge/nep_charge_specialized.cu";
  for (int t = 0; t < para.num_types; ++t) {
    config.rc_radial[t] = para.rc_radial[0];
    config.rc_angular[t] = para.rc_angular[0];
  }
  return config;
}
} // namespace

NEP_Charge::Compiled_Kernel::Compiled_Kernel(const NEP_Compile_Config& config)
  : library(config)
{
  if (!library.is_valid()) {
    return;
  }
  valid = common.load(library);
  valid &= bec.load(library);
  valid &= library.load_symbol("nep_train_launch_ann_charge", ann_charge_);
}

void NEP_Charge::Compiled_Kernel::launch_ann_charge(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
  float* Fp,
  float* charge,
  float* charge_derivative)
{
  check_nep_launch(
    ann_charge_(
      N, type, descriptors, q_scaler, parameters,
      pe, Fp, charge, charge_derivative),
    "ann_charge_jit");
}

void NEP_Charge::Compiled_Kernel::launch_force_charge_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* charge_derivative, const float* D_real,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      charge_derivative, D_real, nullptr, nullptr,
      0, fx, fy, fz, virial),
    "force_charge_radial_jit");
}

void NEP_Charge::Compiled_Kernel::launch_force_charge_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* charge_derivative, const float* D_real,
  const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      charge_derivative, D_real, nullptr, nullptr,
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_charge_angular_jit");
}

void NEP_Charge::initialize_compiled_kernel(const NEP_Charge_Parameters& para)
{
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    compiled_kernel_.reset(new Compiled_Kernel(
      make_compile_config(para)));
    if (!compiled_kernel_->is_valid()) {
      compiled_kernel_.reset();
    }
  }
}

void NEP_Charge::bind_parameters(const float* parameters, ANN& ann)
{
  const int num_outputs = 2;
  const float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * num_outputs;
  }
  ann.sqrt_epsilon_inf = pointer;
  pointer += 1;
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
}

static __global__ void apply_ann(
  const int N,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp,
  float* g_charge,
  float* g_charge_derivative)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];

    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    float charge = 0.0f;
    float charge_derivative[MAX_DIM] = {0.0f};

    apply_ann_one_layer_charge(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type],
      annmb.b0[type],
      annmb.w1[type],
      annmb.b1,
      q,
      F,
      Fp,
      charge,
      charge_derivative);

    g_pe[n1] = static_cast<double>(F);
    g_charge[n1] = charge;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
      g_charge_derivative[n1 + d * N] = charge_derivative[d] * g_q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_Fp,
  const float* g_charge_derivative,
  const float* g_D_real,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1];
        tmp12 *= gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

static __global__ void find_force_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_Fp,
  const float* g_charge_derivative,
  const float* g_D_real,
  const float* g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {

    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      float tmp = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1] 
        + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
      Fp[d] = tmp;
    }
    int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int abc = 0; abc < num_abc; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * num_abc + abc) * N + n1];
      }
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};

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
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, 
          paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

static __global__ void find_bec_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_charge_derivative,
  float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      float bec_xx = 0.5f* (r12[0] * f12[0]);
      float bec_xy = 0.5f* (r12[0] * f12[1]);
      float bec_xz = 0.5f* (r12[0] * f12[2]);
      float bec_yx = 0.5f* (r12[1] * f12[0]);
      float bec_yy = 0.5f* (r12[1] * f12[1]);
      float bec_yz = 0.5f* (r12[1] * f12[2]);
      float bec_zx = 0.5f* (r12[2] * f12[0]);
      float bec_zy = 0.5f* (r12[2] * f12[1]);
      float bec_zz = 0.5f* (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_bec_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Charge::ParaMB paramb,
  const NEP_Charge::ANN annmb,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    int num_abc = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int abc = 0; abc < num_abc; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * num_abc + abc) * N + n1];
      }
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};

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
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, 
          paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }

      float bec_xx = 0.5f* (r12[0] * f12[0]);
      float bec_xy = 0.5f* (r12[0] * f12[1]);
      float bec_xz = 0.5f* (r12[0] * f12[2]);
      float bec_yx = 0.5f* (r12[1] * f12[0]);
      float bec_yy = 0.5f* (r12[1] * f12[1]);
      float bec_yz = 0.5f* (r12[1] * f12[2]);
      float bec_zx = 0.5f* (r12[2] * f12[0]);
      float bec_zy = 0.5f* (r12[2] * f12[1]);
      float bec_zz = 0.5f* (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_force_charge_long_range(
  const int N,
  const float alpha,
  const float alpha_factor,
  const float two_alpha_over_sqrt_pi,
  const int* Na,
  const int* Na_sum,
  const int* g_pbc,
  const float* g_charge,
  const float* g_x,
  const float* g_y,
  const float* g_z,
  const int* g_kpoint_offset,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_real,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  double* g_pe)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  const int is_periodic = g_pbc[blockIdx.x];
  const int kpoint_begin = g_kpoint_offset[blockIdx.x];
  const int num_kpoints = g_kpoint_offset[blockIdx.x + 1] - kpoint_begin;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = threadIdx.x + batch * 1024 + N1;
    if (n < N2) {
      if (is_periodic) {
        double temp_energy_sum = 0.0;
        float temp_virial_sum[6] = {0.0f};
        float temp_force_sum[3] = {0.0f};
        float temp_D_real_sum = 0.0f;
        for (int nk = 0; nk < num_kpoints; ++nk) {
          const int nc_nk = kpoint_begin + nk;
          const float kx = g_kx[nc_nk];
          const float ky = g_ky[nc_nk];
          const float kz = g_kz[nc_nk];
          const float kr = kx * g_x[n] + ky * g_y[n] + kz * g_z[n];
          const float G = g_G[nc_nk];
          const float S_real = g_S_real[nc_nk];
          const float S_imag = g_S_imag[nc_nk];
          float sin_kr = sin(kr);
          float cos_kr = cos(kr);
          const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
          const float GSS = G * (S_real * S_real + S_imag * S_imag);
          temp_energy_sum += GSS;
          const float alpha_k_factor = 2.0f * alpha_factor + 2.0f / (kx * kx + ky * ky + kz * kz);
          temp_virial_sum[0] += GSS * (1.0f - alpha_k_factor * kx * kx); // xx
          temp_virial_sum[1] += GSS * (1.0f - alpha_k_factor * ky * ky); // yy
          temp_virial_sum[2] += GSS * (1.0f - alpha_k_factor * kz * kz); // zz
          temp_virial_sum[3] -= GSS * (alpha_k_factor * kx * ky); // xy
          temp_virial_sum[4] -= GSS * (alpha_k_factor * ky * kz); // yz
          temp_virial_sum[5] -= GSS * (alpha_k_factor * kz * kx); // zx
          temp_D_real_sum += G * (S_real * cos_kr - S_imag * sin_kr);
          temp_force_sum[0] += kx * imag_term;
          temp_force_sum[1] += ky * imag_term;
          temp_force_sum[2] += kz * imag_term;
        }
        g_pe[n] += K_C_SP * temp_energy_sum / (N2 - N1);
        for (int d = 0; d < 6; ++d) {
          g_virial[n + N * d] += K_C_SP * temp_virial_sum[d] / (N2 - N1);
        }
        g_D_real[n] = 2.0f * K_C_SP * temp_D_real_sum;
        const float charge_factor = K_C_SP * 2.0f * g_charge[n];
        g_fx[n] += charge_factor * temp_force_sum[0];
        g_fy[n] += charge_factor * temp_force_sum[1];
        g_fz[n] += charge_factor * temp_force_sum[2];
      } else {
        const float q1 = g_charge[n];
        const float alpha_squared = alpha * alpha;
        // The r=0 limit of erf(alpha*r)/r gives the positive self term.
        double temp_energy_sum = 0.5 * q1 * q1 * two_alpha_over_sqrt_pi;
        float temp_D_real_sum = q1 * two_alpha_over_sqrt_pi;
        float temp_force_sum[3] = {0.0f};
        float temp_virial_sum[6] = {0.0f};
        for (int n2 = N1; n2 < N2; ++n2) {
          if (n2 == n) {
            continue;
          }
          const float x12 = g_x[n2] - g_x[n];
          const float y12 = g_y[n2] - g_y[n];
          const float z12 = g_z[n2] - g_z[n];
          const float d12sq = x12 * x12 + y12 * y12 + z12 * z12;
          const float d12 = sqrt(d12sq);
          const float d12inv = 1.0f / d12;
          const float q2 = g_charge[n2];
          const float qq = q1 * q2;
          const float exp_alpha = exp(-alpha_squared * d12sq);
          const float erf_r = erf(alpha * d12) * d12inv;
          const float f2 = -K_C_SP * qq *
            (erf_r - two_alpha_over_sqrt_pi * exp_alpha) / d12sq;
          const float f12[3] = {x12 * f2, y12 * f2, z12 * f2};

          temp_energy_sum += 0.5 * qq * erf_r;
          temp_D_real_sum += q2 * erf_r;
          temp_force_sum[0] += f12[0];
          temp_force_sum[1] += f12[1];
          temp_force_sum[2] += f12[2];
          temp_virial_sum[0] -= 0.5f * x12 * f12[0];
          temp_virial_sum[1] -= 0.5f * y12 * f12[1];
          temp_virial_sum[2] -= 0.5f * z12 * f12[2];
          temp_virial_sum[3] -= 0.5f * x12 * f12[1];
          temp_virial_sum[4] -= 0.5f * y12 * f12[2];
          temp_virial_sum[5] -= 0.5f * z12 * f12[0];
        }
        g_pe[n] += K_C_SP * temp_energy_sum;
        for (int d = 0; d < 6; ++d) {
          g_virial[n + N * d] += temp_virial_sum[d];
        }
        g_D_real[n] = K_C_SP * temp_D_real_sum;
        g_fx[n] += temp_force_sum[0];
        g_fy[n] += temp_force_sum[1];
        g_fz[n] += temp_force_sum[2];
      }
    }
  }
}

static __global__ void find_force_charge_real_space(
  const int N,
  const float alpha,
  const float two_alpha_over_sqrt_pi,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const float* __restrict__ g_charge,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  double* g_pe,
  float* g_D_real)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    float q1 = g_charge[n1];
    double s_pe = -two_alpha_over_sqrt_pi * 0.5 * q1 * q1; // self energy part
    float D_real = -q1 * two_alpha_over_sqrt_pi; // self energy part

    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float q2 = g_charge[n2];
      float qq = q1 * q2;
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float erfc_r = erfc(alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      float f2 = erfc_r + two_alpha_over_sqrt_pi * exp(-alpha * alpha * d12 * d12);
      f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};

      s_pe += 0.5 * qq * erfc_r;
      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_D_real[n1] += K_C_SP * D_real;
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

static __device__ void cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] =  a[1] * b [2] - a[2] * b [1];
  c[1] =  a[2] * b [0] - a[0] * b [2];
  c[2] =  a[0] * b [1] - a[1] * b [0];
}

static __device__ float get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

static __global__ void find_k_and_G(
  const int Nc,
  const float alpha,
  const float alpha_factor,
  const float* g_box,
  const int* g_pbc,
  const int* g_kpoint_offset,
  int* g_kpoint_count,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G)
{
  int nc = threadIdx.x + blockIdx.x * blockDim.x; // structure index
  if (nc < Nc) {
    if (!g_pbc[nc]) {
      if (g_kpoint_count != nullptr) {
        g_kpoint_count[nc] = 0;
      }
      return;
    }
    const float* box = g_box + 9 * nc;
    const float det = box[0] * (box[4] * box[8] - box[5] * box[7]) +
                      box[1] * (box[5] * box[6] - box[3] * box[8]) +
                      box[2] * (box[3] * box[7] - box[4] * box[6]);
    const float a1[3] = {box[0], box[3], box[6]};
    const float a2[3] = {box[1], box[4], box[7]};
    const float a3[3] = {box[2], box[5], box[8]};
    float b1[3] = {0.0f};
    float b2[3] = {0.0f};
    float b3[3] = {0.0f};
    cross_product(a2, a3, b1);
    cross_product(a3, a1, b2);
    cross_product(a1, a2, b3);
    
    const float two_pi = 6.2831853f;
    const float two_pi_over_det = two_pi / det;
    for (int d = 0; d < 3; ++d) {
      b1[d] *= two_pi_over_det;
      b2[d] *= two_pi_over_det;
      b3[d] *= two_pi_over_det;
    }

    const float volume_k = two_pi * two_pi * two_pi / abs(det);
    int n1_max = alpha * two_pi * get_area(b2, b3) / volume_k;
    int n2_max = alpha * two_pi * get_area(b3, b1) / volume_k;
    int n3_max = alpha * two_pi * get_area(b1, b2) / volume_k;
    float ksq_max = two_pi * two_pi * alpha * alpha;

    int nk = 0;
    for (int n1 = 0; n1 <= n1_max; ++n1) {
      for (int n2 = - n2_max; n2 <= n2_max; ++n2) {
        for (int n3 = - n3_max; n3 <= n3_max; ++n3) {
          if (n1 == 0 && (n2 < 0 || (n2 == 0 && n3 <= 0))) continue;
          const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
          const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
          const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
          const float ksq = kx * kx + ky * ky + kz * kz;
          if (ksq < ksq_max) {
            if (g_kx != nullptr) {
              const int nc_nk = g_kpoint_offset[nc] + nk;
              g_kx[nc_nk] = kx;
              g_ky[nc_nk] = ky;
              g_kz[nc_nk] = kz;
              g_G[nc_nk] =
                2.0f * abs(two_pi_over_det) / ksq * exp(-ksq * alpha_factor);
            }
            ++nk;
          }
        }
      }
    }
    if (g_kpoint_count != nullptr) {
      g_kpoint_count[nc] = nk;
    }
  }
}

void NEP_Charge::prepare_kpoints(Dataset& dataset, int device_id)
{
  NEP_Charge_Data& data = nep_data[device_id];

  // Training datasets and their boxes are immutable after construction.
  if (data.kpoint_dataset == &dataset) {
    return;
  }

  const int grid_size = (dataset.Nc - 1) / 64 + 1;
  find_k_and_G<<<grid_size, 64>>>(
    dataset.Nc,
    charge_para.alpha,
    charge_para.alpha_factor,
    dataset.box_original.data(),
    dataset.pbc.data(),
    nullptr,
    data.kpoint_offset.data(),
    nullptr,
    nullptr,
    nullptr,
    nullptr);
  GPU_CHECK_KERNEL

  const int total_num_kpoints = update_nep_kpoint_offsets(dataset.Nc, data.kpoint_offset);

  const int required_capacity =
    total_num_kpoints > 0 ? total_num_kpoints : 1;
  if (required_capacity > static_cast<int>(data.kx.size())) {
    data.kx.resize(required_capacity);
    data.ky.resize(required_capacity);
    data.kz.resize(required_capacity);
    data.G.resize(required_capacity);
    data.S_real.resize(required_capacity);
    data.S_imag.resize(required_capacity);
  }

  find_k_and_G<<<grid_size, 64>>>(
    dataset.Nc,
    charge_para.alpha,
    charge_para.alpha_factor,
    dataset.box_original.data(),
    dataset.pbc.data(),
    data.kpoint_offset.data(),
    nullptr,
    data.kx.data(),
    data.ky.data(),
    data.kz.data(),
    data.G.data());
  GPU_CHECK_KERNEL
  data.kpoint_dataset = &dataset;
}

void NEP_Charge::compute_descriptors(
  const ANN& ann, const Dataset& data, const NEP_Neighbor& neighbors,
  NEP_Charge_Data& storage)
{
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->common.launch_descriptor_radial(
      data.N,
      data.NN_radial_sum.data(),
      data.NN_radial.data(),
      neighbors.NL_radial.data(),
      data.type.data(),
      neighbors.x12_radial.data(),
      neighbors.y12_radial.data(),
      neighbors.z12_radial.data(),
      storage.parameters.data(),
      storage.descriptors.data());
  } else {
    find_descriptors_radial<<<grid_size, block_size>>>(
      data.N,
      data.NN_radial_sum.data(),
      data.NN_radial.data(),
      neighbors.NL_radial.data(),
      paramb,
      ann,
      data.type.data(),
      neighbors.x12_radial.data(),
      neighbors.y12_radial.data(),
      neighbors.z12_radial.data(),
      storage.descriptors.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel_) {
    compiled_kernel_->common.launch_descriptor_angular(
      data.N,
      data.NN_angular_sum.data(),
      data.NN_angular.data(),
      neighbors.NL_angular.data(),
      data.type.data(),
      neighbors.x12_angular.data(),
      neighbors.y12_angular.data(),
      neighbors.z12_angular.data(),
      storage.parameters.data(),
      storage.descriptors.data(),
      storage.sum_fxyz.data());
  } else {
    find_descriptors_angular<<<grid_size, block_size>>>(
      data.N,
      data.NN_angular_sum.data(),
      data.NN_angular.data(),
      neighbors.NL_angular.data(),
      paramb,
      ann,
      data.type.data(),
      neighbors.x12_angular.data(),
      neighbors.y12_angular.data(),
      neighbors.z12_angular.data(),
      storage.descriptors.data(),
      storage.sum_fxyz.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_Charge::evaluate_ann(
  const ANN& ann, Dataset& data,
  NEP_Charge_Data& storage, const float* q_scaler)
{
  auto& charge_dataset = charge_data.get(data);
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->launch_ann_charge(
      data.N,
      data.type.data(),
      storage.descriptors.data(),
      q_scaler,
      storage.parameters.data(),
      data.energy.data(),
      storage.Fp.data(),
      charge_dataset.charge.data(),
      storage.charge_derivative.data());
  } else {
    apply_ann<<<grid_size, block_size>>>(
      data.N,
      paramb,
      ann,
      data.type.data(),
      storage.descriptors.data(),
      q_scaler,
      data.energy.data(),
      storage.Fp.data(),
      charge_dataset.charge.data(),
      storage.charge_derivative.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_Charge::compute_bec(
  const ANN& ann, Dataset& data, const NEP_Neighbor& neighbors,
  NEP_Charge_Data& storage)
{
  auto& charge_dataset = charge_data.get(data);
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  // get BEC (the diagonal part)
  initialize_nep_bec(data, charge_dataset);

  // get BEC (radial descriptor part)
  if (compiled_kernel_) {
    compiled_kernel_->bec.launch_bec_radial(
      data.N,
      data.NN_radial_sum.data(),
      data.NN_radial.data(),
      neighbors.NL_radial.data(),
      data.type.data(),
      neighbors.x12_radial.data(),
      neighbors.y12_radial.data(),
      neighbors.z12_radial.data(),
      storage.parameters.data(),
      storage.charge_derivative.data(),
      charge_dataset.bec.data());
  } else {
    find_bec_radial<<<grid_size, block_size>>>(
      data.N,
      data.NN_radial_sum.data(),
      data.NN_radial.data(),
      neighbors.NL_radial.data(),
      paramb,
      ann,
      data.type.data(),
      neighbors.x12_radial.data(),
      neighbors.y12_radial.data(),
      neighbors.z12_radial.data(),
      storage.charge_derivative.data(),
      charge_dataset.bec.data());
    GPU_CHECK_KERNEL
  }

  // get BEC (angular descriptor part)
  if (compiled_kernel_) {
    compiled_kernel_->bec.launch_bec_angular(
      data.N,
      data.NN_angular_sum.data(),
      data.NN_angular.data(),
      neighbors.NL_angular.data(),
      data.type.data(),
      neighbors.x12_angular.data(),
      neighbors.y12_angular.data(),
      neighbors.z12_angular.data(),
      storage.parameters.data(),
      storage.charge_derivative.data(),
      storage.sum_fxyz.data(),
      charge_dataset.bec.data());
  } else {
    find_bec_angular<<<grid_size, block_size>>>(
      data.N,
      data.NN_angular_sum.data(),
      data.NN_angular.data(),
      neighbors.NL_angular.data(),
      paramb,
      ann,
      data.type.data(),
      neighbors.x12_angular.data(),
      neighbors.y12_angular.data(),
      neighbors.z12_angular.data(),
      storage.charge_derivative.data(),
      storage.sum_fxyz.data(),
      charge_dataset.bec.data());
    GPU_CHECK_KERNEL
  }

  // scale q to q * sqrt(epsilon_inf)
  scale_nep_bec(data, charge_dataset, ann.sqrt_epsilon_inf);
}

void NEP_Charge::compute_charge_long_range(
  Dataset& data, NEP_Charge_Data& storage)
{
  auto& charge_dataset = charge_data.get(data);
  find_force_charge_long_range<<<data.Nc, 1024>>>(
    data.N,
    charge_para.alpha,
    charge_para.alpha_factor,
    charge_para.two_alpha_over_sqrt_pi,
    data.Na.data(),
    data.Na_sum.data(),
    data.pbc.data(),
    charge_dataset.charge_shifted.data(),
    data.r.data(),
    data.r.data() + data.N,
    data.r.data() + data.N * 2,
    storage.kpoint_offset.data(),
    storage.kx.data(),
    storage.ky.data(),
    storage.kz.data(),
    storage.G.data(),
    storage.S_real.data(),
    storage.S_imag.data(),
    storage.D_real.data(),
    data.force.data(),
    data.force.data() + data.N,
    data.force.data() + data.N * 2,
    data.virial.data(),
    data.energy.data());
  GPU_CHECK_KERNEL
}

void NEP_Charge::compute_charge_real_space(
  Dataset& data, const NEP_Neighbor& neighbors,
  NEP_Charge_Data& storage)
{
  auto& charge_dataset = charge_data.get(data);
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  find_force_charge_real_space<<<grid_size, block_size>>>(
    data.N,
    charge_para.alpha,
    charge_para.two_alpha_over_sqrt_pi,
    data.NN_radial_sum.data(),
    data.NN_radial.data(),
    neighbors.NL_radial.data(),
    charge_dataset.charge_shifted.data(),
    neighbors.x12_radial.data(),
    neighbors.y12_radial.data(),
    neighbors.z12_radial.data(),
    data.force.data(),
    data.force.data() + data.N,
    data.force.data() + data.N * 2,
    data.virial.data(),
    data.energy.data(),
    storage.D_real.data());
  GPU_CHECK_KERNEL
}

void NEP_Charge::compute_forces(
  const ANN& ann, Dataset& data, const NEP_Neighbor& neighbors,
  NEP_Charge_Data& storage)
{
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->launch_force_charge_radial(
      data.N,
      data.NN_radial_sum.data(),
      data.NN_radial.data(),
      neighbors.NL_radial.data(),
      data.type.data(),
      neighbors.x12_radial.data(),
      neighbors.y12_radial.data(),
      neighbors.z12_radial.data(),
      storage.parameters.data(),
      storage.Fp.data(),
      storage.charge_derivative.data(),
      storage.D_real.data(),
      data.force.data(),
      data.force.data() + data.N,
      data.force.data() + data.N * 2,
      data.virial.data());
  } else {
    find_force_radial<<<grid_size, block_size>>>(
      data.N,
      data.NN_radial_sum.data(),
      data.NN_radial.data(),
      neighbors.NL_radial.data(),
      paramb,
      ann,
      data.type.data(),
      neighbors.x12_radial.data(),
      neighbors.y12_radial.data(),
      neighbors.z12_radial.data(),
      storage.Fp.data(),
      storage.charge_derivative.data(),
      storage.D_real.data(),
      data.force.data(),
      data.force.data() + data.N,
      data.force.data() + data.N * 2,
      data.virial.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel_) {
    compiled_kernel_->launch_force_charge_angular(
      data.N,
      data.NN_angular_sum.data(),
      data.NN_angular.data(),
      neighbors.NL_angular.data(),
      data.type.data(),
      neighbors.x12_angular.data(),
      neighbors.y12_angular.data(),
      neighbors.z12_angular.data(),
      storage.parameters.data(),
      storage.Fp.data(),
      storage.charge_derivative.data(),
      storage.D_real.data(),
      storage.sum_fxyz.data(),
      data.force.data(),
      data.force.data() + data.N,
      data.force.data() + data.N * 2,
      data.virial.data());
  } else {
    find_force_angular<<<grid_size, block_size>>>(
      data.N,
      data.NN_angular_sum.data(),
      data.NN_angular.data(),
      neighbors.NL_angular.data(),
      paramb,
      ann,
      data.type.data(),
      neighbors.x12_angular.data(),
      neighbors.y12_angular.data(),
      neighbors.z12_angular.data(),
      storage.Fp.data(),
      storage.charge_derivative.data(),
      storage.D_real.data(),
      storage.sum_fxyz.data(),
      data.force.data(),
      data.force.data() + data.N,
      data.force.data() + data.N * 2,
      data.virial.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_Charge::find_force(
  NEP_Charge_Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{
  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    auto& storage = nep_data[device_id];
    auto& ann = annmb[device_id];
    storage.parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    bind_parameters(storage.parameters.data(), ann);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    auto& data = dataset[device_id];
    auto& charge_dataset = charge_data.get(data);
    auto& neighbors = neighbor[device_id];
    auto& storage = nep_data[device_id];
    const auto& ann = annmb[device_id];
    neighbors.prepare(para, data, device_id);

    compute_descriptors(ann, data, neighbors, storage);

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      output_nep_descriptors(
        data, ann.dim, para.output_descriptor, para.q_scaler_cpu, storage.descriptors);
    }

    if (calculate_q_scaler) {
      update_nep_q_scaler(
        data.N, ann.dim, storage.descriptors.data(),
        para.q_scaler_gpu[device_id].data(), para.q_scaler_max[device_id].data(),
        para.q_scaler_min[device_id].data());
    }

    zero_nep_long_range_outputs(data);

    evaluate_ann(ann, data, storage, para.q_scaler_gpu[device_id].data());

    // enforce total charge is the target
    enforce_nep_total_charge(data, charge_dataset);

    if (para.has_bec) {
      compute_bec(ann, data, neighbors, storage);
    }

    // Long range: reciprocal-space Ewald for PPP and direct erf summation for FFF.
    prepare_kpoints(data, device_id);

    compute_nep_structure_factor(
      data, charge_dataset.charge_shifted.data(), storage.kpoint_offset.data(),
      storage.kx.data(), storage.ky.data(), storage.kz.data(),
      storage.S_real.data(), storage.S_imag.data());

    compute_charge_long_range(data, storage);

    // mode 1 has real space
    if (paramb.charge_mode == 1) {
      compute_charge_real_space(data, neighbors, storage);
    }

    // Chain rule correction: D_real -= mean(D_real) per structure
    subtract_nep_charge_derivative_mean(data, storage.D_real.data());

    compute_forces(ann, data, neighbors, storage);

    if (zbl.enabled) {
      add_nep_zbl(zbl, data, neighbors);
    }
  }
}

void NEP_Charge::configure(NEP_Charge_Parameters& para)
{
  para.file_tag = std::string(para.enable_zbl ? "nep4_zbl" : "nep4") + ("_charge" + std::to_string(para.charge_mode));
  calculate_nep_model_parameters(para, 0, 1, 2, 1);
}

void NEP_Charge::initialize_parameters(
  std::mt19937& rng,
  std::vector<float>& mu, std::vector<float>& sigma)
{
  NEP_Charge_Parameters& para = *model_parameters;
  NEP_Initialization initialization = initialize_nep_mu_and_sigma(para, rng, mu, sigma, 2);
  if (initialization == NEP_Initialization::RANDOM) {
    // make sure the initial charges are zero
    for (int t = 0; t < para.num_types; ++t) {
      int charge_begin =
        para.number_of_variables_ann_1 * t + (para.dim + 2) * para.num_neurons1;
      int charge_end = charge_begin + para.num_neurons1;
      for (int n = charge_begin; n < charge_end; ++n) {
        mu[n] = 0.0f;
      }
    }
    mu[para.number_of_variables_ann_1 * para.num_types] = 2.0f; // initial sqrt(epsilon_inf)
  } else if (initialization == NEP_Initialization::RESTART && para.flip_charge) {
    for (int t = 0; t < para.num_types; ++t) {
      int charge_begin =
        para.number_of_variables_ann_1 * t + (para.dim + 2) * para.num_neurons1;
      int charge_end = charge_begin + para.num_neurons1;
      for (int n = charge_begin; n < charge_end; ++n) {
        mu[n] = -mu[n];
      }
    }
  }
}

void NEP_Charge::parameter_groups(std::vector<int>& groups)
{
  NEP_Charge_Parameters& para = *model_parameters;
  find_nep_type_of_variable(para, groups, 2, 1);
}

void NEP_Charge::read_model(float* parameters)
{
  read_nep_model("nep.txt", *model_parameters, parameters);
}

void NEP_Charge::write_restart(
  const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma)
{
  write_nep_restart(filename, *model_parameters, mu, sigma);
}

NEP_Charge::NEP_Charge(const ParameterInput& input)
  : model_parameters(new NEP_Charge_Parameters)
{
  auto& para = *model_parameters;
  para.charge_mode = input.charge_mode;
  para.flip_charge = input.flip_charge;
  para.read(input);
  if (para.num_hidden_layers == 2) {
    PRINT_INPUT_ERROR("Can only use one hidden layer.");
  }
  if (para.has_multiple_cutoffs) {
    PRINT_INPUT_ERROR("Can only use uniform cutoff for qNEP.");
  }
  configure(para);
  para.finish_reading(input, "potential", para.charge_mode == 1 ?
    "use NEP-Charge and include both real-space and k-space." :
    "use NEP-Charge and include k-space only.");
}

int NEP_Charge::prepare_data()
{
  NEP_Charge_Parameters& para = *model_parameters;
  NEP_Charge_Reader reader(para, energy_difference, charge_data);
  training.reset(new NEP_Training(para, reader));
  initialize(para, training->N, training->Nc, training->device_count);
  if (para.prediction == 0) {
    energy_loss.initialize(para);
    fitness_charge.resize(para.population_size * (para.num_types + 1));
    fitness_bec.resize(para.population_size * (para.num_types + 1));
    training->open_loss_file(para,
      " rmse_energy_train rmse_force_train rmse_virial_train rmse_charge_train rmse_bec_train"
      " rmse_energy_test rmse_force_test rmse_virial_test rmse_charge_test rmse_bec_test",
      para.has_ediff_combinations);
  }
  return training->num_batches;
}

void NEP_Charge::prepare_training()
{
  NEP_Charge_Parameters& para = *model_parameters;
  int devices = training->device_count;
  std::vector<float> dummy_solution(para.number_of_variables * devices, para.initial_para);
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, dummy_solution.data(), training->train_set[batch],
      (para.fine_tune || para.import_q_scaler) ? false : true, devices);
  }
}

void NEP_Charge::evaluate(int batch, const float* population)
{
  NEP_Charge_Parameters& para = *model_parameters;
  int devices = training->device_count;
  for (int first = 0; first < para.population_size; first += devices) {
    int count = std::min(devices, para.population_size - first);
    find_force(para, population + first * para.number_of_variables,
      training->train_set[batch], false, count);
    for (int device = 0; device < count; ++device) {
      Dataset& data = training->train_set[batch][device];
      energy_loss.evaluate(para, data, energy_difference, batch, first + device, device);
      auto rmse_charge_array = charge_data.get(data).get_rmse_charge(para, data, device);
      auto rmse_bec_array = charge_data.get(data).get_rmse_bec(para, data, device);
      for (int t = 0; t <= para.num_types; ++t) {
        int index = first + device + t * para.population_size;
        fitness_charge[index] = para.lambda_q * rmse_charge_array[t];
        fitness_bec[index] = para.lambda_z * rmse_bec_array[t];
      }
    }
  }
}

void NEP_Charge::accumulate_fitness(float* fitness_total)
{
  NEP_Charge_Parameters& para = *model_parameters;
  energy_loss.accumulate_fitness(para, fitness_total);
  for (int i = 0; i < para.population_size * (para.num_types + 1); ++i) {
    fitness_total[i] = fitness_total[i] + fitness_charge[i] + fitness_bec[i];
  }
  energy_loss.accumulate_energy_difference(para, fitness_total);
}

void NEP_Charge::print_training_header()
{
  print_nep_charge_header(model_parameters->has_ediff_combinations);
}

void NEP_Charge::report_error(const TrainingReport& report)
{
  NEP_Charge_Parameters& para = *model_parameters;
  find_force(para, report.parameters, training->train_set[report.batch], false, 1);
  Dataset& train_data = training->train_set[report.batch][0];
  NEP_Energy_Error train = energy_loss.get_error(
    para, train_data, energy_difference, true, report.batch);
  float charge_train = charge_data.get(train_data).get_rmse_charge(para, train_data, 0).back();
  float bec_train = charge_data.get(train_data).get_rmse_bec(para, train_data, 0).back();
  float charge_test = 0.0f;
  float bec_test = 0.0f;
  // Correct the last bias parameter in the energy network before evaluating the test set.
  report.parameters[para.number_of_variables_ann - 1] += train.energy_shift;
  NEP_Energy_Error test;
  if (training->has_test_set) {
    find_force(para, report.parameters, training->test_set, false, 1);
    Dataset& test_data = training->test_set[0];
    test = energy_loss.get_error(para, test_data, energy_difference, false, 0);
    charge_test = charge_data.get(test_data).get_rmse_charge(para, test_data, 0).back();
    bec_test = charge_data.get(test_data).get_rmse_bec(para, test_data, 0).back();
  }
  training->write_model(para, report);
  write_nep_charge_error(stdout, report, train, test,
    charge_train, bec_train, charge_test, bec_test, para.has_ediff_combinations);
  write_nep_charge_error(training->fid_loss_out, report, train, test,
    charge_train, bec_train, charge_test, bec_test, para.has_ediff_combinations);
  if (training->has_test_set) {
    Dataset& data = training->test_set[0];
    copy_nep_energy_predictions(data);
    copy_nep_charge_predictions(charge_data.get(data), para.has_bec);
    std::vector<std::pair<Dataset*, int>> structures;
    for (int nc = 0; nc < data.Nc; ++nc) {
      structures.emplace_back(&data, nc);
    }
    write_nep_energy_predictions("test", structures);
    write_nep_charge_predictions("test", structures, charge_data, para.has_bec);
  }
}

void NEP_Charge::predict(float* parameters)
{
  NEP_Charge_Parameters& para = *model_parameters;
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, parameters, training->train_set[batch], false, 1);
    Dataset& data = training->train_set[batch][0];
    copy_nep_energy_predictions(data);
    copy_nep_charge_predictions(charge_data.get(data), para.has_bec);
  }
  std::vector<std::pair<Dataset*, int>> structures;
  training->get_prediction_order(structures);
  write_nep_energy_predictions("train", structures);
  write_nep_charge_predictions("train", structures, charge_data, para.has_bec);
}
