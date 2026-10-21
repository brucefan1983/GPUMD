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
The tensorial neuroevolution potential (TNEP)
Ref: Nan Xu, Petter Rosander, Christian Schäfer, Eric Lindgren, 
Nicklas Österbacka, Mandi Fang, Wei Chen, Yi He, Zheyong Fan, Paul Erhart, 
Tensorial properties via the neuroevolution potential framework: 
Fast simulation of infrared and Raman spectra, 
J. Chem. Theory Comput. 20, 3273 (2024).
------------------------------------------------------------------------------*/

#include "main_nep/dataset.cuh"
#include "tnep.cuh"
#include "utilities/read_file.cuh"
#include "main_nep/common/nep_output.cuh"
#include "main_nep/xyz_reader.cuh"
#include "main_nep/common/nep_model.cuh"
#include "main_nep/common/nep_init.cuh"
#include "main_nep/common/nep_io.cuh"
#include "main_nep/common/nep_common.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstring>
#include <algorithm>

static void write_tnep_error(FILE* file, const TrainingReport& report, float train, float test)
{
  fprintf(file, "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f\n",
    report.generation + 1, report.total, report.l1, report.l2, train, test);
  fflush(file);
}

static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const TNEP::ParaMB paramb,
  const TNEP::ANN annmb,
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
  const TNEP::ParaMB paramb,
  const TNEP::ANN annmb,
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

void TNEP::initialize(TNEP_Parameters& para, int N, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    initialize_device(para, N, device_id);
  }
  initialize_compiled_kernel(para);
}

void TNEP::initialize_model(const TNEP_Parameters& para)
{
  paramb.version = para.version;
  paramb.num_types = para.num_types;
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial[t] = para.rc_radial[t];
    paramb.rc_angular[t] = para.rc_angular[t];
  }
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
}

void TNEP::initialize_device(const TNEP_Parameters& para, int N, int device_id)
{
  ANN& ann = annmb[device_id];
  TNEP_Data& data = nep_data[device_id];
  ann.dim = para.dim;
  ann.num_neurons1 = para.num_neurons1;
  ann.num_para = para.number_of_variables;
  ann.num_hidden_layers = para.num_hidden_layers;
  if (para.num_hidden_layers == 2) {
    ann.num_neurons2 = para.num_neurons2;
  }
  ann.one_ann_no_bias = para.number_of_variables_ann_1;

  data.descriptors.resize(N * ann.dim);
  data.Fp.resize(N * ann.dim);
  data.sum_fxyz.resize(
    N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  data.parameters.resize(ann.num_para);
}

namespace
{
NEP_Compile_Config make_compile_config(
  const TNEP_Parameters& para)
{
#ifdef USE_CJ
  warning_nep_compile(
    "qNEP, charge-vdW, and TNEP specialization is not available in "
    "this USE_CJ configuration.");
  return NEP_Compile_Config();
#endif
  NEP_Compile_Config config = make_nep_compile_config(para);
  config.model_name = "TNEP";
  config.source_file = "main_nep/tnep/tnep_specialized.cu";
  return config;
}
} // namespace

TNEP::Compiled_Kernel::Compiled_Kernel(const NEP_Compile_Config& config)
  : library(config)
{
  if (!library.is_valid()) {
    return;
  }
  valid = common.load(library);
  valid &= library.load_symbol("nep_train_launch_ann_nep", ann_nep_);
  valid &= library.load_symbol("nep_train_launch_ann_tnep_pol", ann_tnep_pol_);
}

void TNEP::Compiled_Kernel::launch_ann(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
  float* Fp)
{
  check_nep_launch(
    ann_nep_(
      N, type, descriptors, q_scaler, parameters, pe, Fp),
    "ann_nep_jit");
}

void TNEP::Compiled_Kernel::launch_ann_tnep_pol(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* virial,
  float* Fp)
{
  check_nep_launch(
    ann_tnep_pol_(
      N, type, descriptors, q_scaler, parameters, virial, Fp),
    "ann_tnep_pol_jit");
}

void TNEP::Compiled_Kernel::launch_force_tnep_radial(
  bool is_dipole,
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      nullptr, nullptr, nullptr, nullptr,
      is_dipole ? 1 : 0, fx, fy, fz, virial),
    "force_tnep_radial_jit");
}

void TNEP::Compiled_Kernel::launch_force_tnep_angular(
  bool is_dipole,
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp, const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      nullptr, nullptr, nullptr, nullptr,
      sum_fxyz, is_dipole ? 1 : 0, fx, fy, fz, virial),
    "force_tnep_angular_jit");
}

void TNEP::initialize_compiled_kernel(const TNEP_Parameters& para)
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

void TNEP::bind_parameters(const TNEP_Parameters& para, const float* parameters, ANN& ann)
{
  const float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.wb[t] = pointer;
    pointer += ann.one_ann_no_bias;
  }
  ann.b = pointer;
  pointer += 1;

  if (para.model_type == 2) {
    for (int t = 0; t < paramb.num_types; ++t) {
      ann.wb_pol[t] = pointer;
      pointer += ann.one_ann_no_bias;
    }
    ann.b_pol = pointer;
    pointer += 1;
  }

  ann.c = pointer;
}

static __global__ void apply_ann(
  const int N,
  const TNEP::ParaMB paramb,
  const TNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
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

    const int neu1 = annmb.num_neurons1;
    const int neu1_dim = neu1 * annmb.dim;
    if (annmb.num_hidden_layers == 2) {
      const int neu2 = annmb.num_neurons2;
      apply_ann_two_layers(
        annmb.dim,
        neu1,
        neu2,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2) + neu2,
        annmb.b,
        q,
        F,
        Fp);
    } else {
      apply_ann_one_layer(
        annmb.dim,
        neu1,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.b,
        q,
        F,
        Fp);
    }
    g_pe[n1] = static_cast<double>(F);

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void apply_ann_pol(
  const int N,
  const TNEP::ParaMB paramb,
  const TNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_virial,
  float* g_Fp)
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

    // scalar part
    const int neu1 = annmb.num_neurons1;
    const int neu1_dim = neu1 * annmb.dim;
    const int neu2 = annmb.num_neurons2;
    if (annmb.num_hidden_layers == 2) {
      apply_ann_two_layers(
        annmb.dim,
        neu1,
        neu2,
        annmb.wb_pol[type],
        annmb.wb_pol[type] + neu1_dim,
        annmb.wb_pol[type] + neu1 * (annmb.dim + 1),
        annmb.wb_pol[type] + neu1 * (annmb.dim + 1 + neu2),
        annmb.wb_pol[type] + neu1 * (annmb.dim + 1 + neu2) + neu2,
        annmb.b_pol,
        q,
        F,
        Fp);
    } else {
      apply_ann_one_layer(
        annmb.dim,
        neu1,
        annmb.wb_pol[type],
        annmb.wb_pol[type] + neu1_dim,
        annmb.wb_pol[type] + neu1 * (annmb.dim + 1),
        annmb.b_pol,
        q,
        F,
        Fp);
    }
    g_virial[n1] = F;
    g_virial[n1 + N] = F;
    g_virial[n1 + N * 2] = F;

    // tensor part
    for (int d = 0; d < annmb.dim; ++d) {
      Fp[d] = 0.0f;
    }
    if (annmb.num_hidden_layers == 2) {
      apply_ann_two_layers(
        annmb.dim,
        neu1,
        neu2,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2) + neu2,
        annmb.b,
        q,
        F,
        Fp);
    } else {
      apply_ann_one_layer(
        annmb.dim,
        neu1,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.b,
        q,
        F,
        Fp);
    }

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  const bool is_dipole,
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const TNEP::ParaMB paramb,
  const TNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
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
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
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
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
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

      if (is_dipole) {
        float r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_virial_xx -= r12_square * f12[0];
        s_virial_yy -= r12_square * f12[1];
        s_virial_zz -= r12_square * f12[2];
      } else {
        s_virial_xx -= r12[0] * f12[0];
        s_virial_yy -= r12[1] * f12[1];
        s_virial_zz -= r12[2] * f12[2];
      }
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] = s_virial_xy;
    g_virial[n1 + N * 4] = s_virial_yz;
    g_virial[n1 + N * 5] = s_virial_zx;
  }
}

static __global__ void find_force_angular(
  const bool is_dipole,
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const TNEP::ParaMB paramb,
  const TNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
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
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
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
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
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

      if (is_dipole) {
        float r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_virial_xx -= r12_square * f12[0];
        s_virial_yy -= r12_square * f12[1];
        s_virial_zz -= r12_square * f12[2];
      } else {
        s_virial_xx -= r12[0] * f12[0];
        s_virial_yy -= r12[1] * f12[1];
        s_virial_zz -= r12[2] * f12[2];
      }
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

void TNEP::compute_descriptors(
  const ANN& ann,
  const Dataset& dataset,
  const NEP_Neighbor& neighbor,
  TNEP_Data& data)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->common.launch_descriptor_radial(
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.parameters.data(),
      data.descriptors.data());
  } else {
    find_descriptors_radial<<<grid_size, block_size>>>(
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      paramb,
      ann,
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.descriptors.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel_) {
    compiled_kernel_->common.launch_descriptor_angular(
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.parameters.data(),
      data.descriptors.data(),
      data.sum_fxyz.data());
  } else {
    find_descriptors_angular<<<grid_size, block_size>>>(
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      paramb,
      ann,
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.descriptors.data(),
      data.sum_fxyz.data());
    GPU_CHECK_KERNEL
  }
}

void TNEP::evaluate_ann(
  const ANN& ann,
  bool is_polarizability,
  Dataset& dataset,
  TNEP_Data& data,
  const float* q_scaler)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    if (is_polarizability) {
      compiled_kernel_->launch_ann_tnep_pol(
        dataset.N,
        dataset.type.data(),
        data.descriptors.data(),
        q_scaler,
        data.parameters.data(),
        dataset.virial.data(),
        data.Fp.data());
    } else {
      compiled_kernel_->launch_ann(
        dataset.N,
        dataset.type.data(),
        data.descriptors.data(),
        q_scaler,
        data.parameters.data(),
        dataset.energy.data(),
        data.Fp.data());
    }
  } else {
    if (is_polarizability) {
      apply_ann_pol<<<grid_size, block_size>>>(
        dataset.N,
        paramb,
        ann,
        dataset.type.data(),
        data.descriptors.data(),
        q_scaler,
        dataset.virial.data(),
        data.Fp.data());
      GPU_CHECK_KERNEL
    } else {
      apply_ann<<<grid_size, block_size>>>(
        dataset.N,
        paramb,
        ann,
        dataset.type.data(),
        data.descriptors.data(),
        q_scaler,
        dataset.energy.data(),
        data.Fp.data());
      GPU_CHECK_KERNEL
    }
  }
}

void TNEP::compute_tensor(
  const ANN& ann,
  bool is_dipole,
  Dataset& dataset,
  const NEP_Neighbor& neighbor,
  TNEP_Data& data)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->launch_force_tnep_radial(
      is_dipole,
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.parameters.data(),
      data.Fp.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
  } else {
    find_force_radial<<<grid_size, block_size>>>(
      is_dipole,
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      paramb,
      ann,
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.Fp.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel_) {
    compiled_kernel_->launch_force_tnep_angular(
      is_dipole,
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.parameters.data(),
      data.Fp.data(),
      data.sum_fxyz.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
  } else {
    find_force_angular<<<grid_size, block_size>>>(
      is_dipole,
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      paramb,
      ann,
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.Fp.data(),
      data.sum_fxyz.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
    GPU_CHECK_KERNEL
  }
}

void TNEP::find_force(
  TNEP_Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{
  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    bind_parameters(para, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    neighbor[device_id].prepare(para, dataset[device_id], device_id);

    compute_descriptors(
      annmb[device_id], dataset[device_id], neighbor[device_id], nep_data[device_id]);

    if (calculate_q_scaler) {
      update_nep_q_scaler(
        dataset[device_id].N, annmb[device_id].dim, nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(), para.q_scaler_max[device_id].data(),
        para.q_scaler_min[device_id].data());
    }

    zero_nep_outputs(dataset[device_id]);

    evaluate_ann(
      annmb[device_id], para.model_type == 2, dataset[device_id], nep_data[device_id],
      para.q_scaler_gpu[device_id].data());

    compute_tensor(
      annmb[device_id], para.model_type == 1, dataset[device_id], neighbor[device_id],
      nep_data[device_id]);
  }
}

void TNEP::configure(TNEP_Parameters& para)
{
  para.file_tag = std::string(para.enable_zbl ? "nep4_zbl" : "nep4") + (para.model_type == 1 ? "_dipole" : "_polarizability");
  calculate_nep_model_parameters(para, 0, 0, 1, para.model_type == 2 ? 2 : 1);
}

void TNEP::initialize_parameters(
  std::mt19937& rng,
  std::vector<float>& mu, std::vector<float>& sigma)
{
  TNEP_Parameters& para = *model_parameters;
  initialize_nep_mu_and_sigma(para, rng, mu, sigma, 1);
}

void TNEP::parameter_groups(std::vector<int>& groups)
{
  TNEP_Parameters& para = *model_parameters;
  find_nep_type_of_variable(para, groups, 1, para.model_type == 2 ? 2 : 1);
}

void TNEP::read_model(float* parameters)
{
  read_nep_model("nep.txt", *model_parameters, parameters);
}

void TNEP::write_restart(
  const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma)
{
  write_nep_restart(filename, *model_parameters, mu, sigma);
}

namespace {
class TNEP_Reader : public NEP_Data_Reader
{
public:
  TNEP_Reader(TNEP_Parameters& para, NEP_Batch_Data<TNEP_Atomic_Dataset>& atomic_data)
    : NEP_Data_Reader(para), model_parameters(para), atomic_data(atomic_data) {}
  void prepare_dataset(Dataset& dataset) override
  {
    atomic_data.create(dataset).construct(dataset, labels);
  }

private:
  TNEP_Parameters& model_parameters;
  NEP_Batch_Data<TNEP_Atomic_Dataset>& atomic_data;
  std::vector<TNEP_Atomic_Labels> labels;
  void begin_reading() override { labels.clear(); }
  void read_structure(const XYZ_Record& record, Structure& structure) override
  {
    labels.emplace_back();
    auto& label = labels.back();
    read_geometry(record, structure);
    // Optional energy labels retain the existing ordering of mini-batches.
    read_energy(record, structure, false);
    const auto& tokens = record.tokens;
    const auto& xyz_filename = record.filename;
    int line_number = record.comment_line_number;
    // use the virial variable to keep the dipole data
    if (model_parameters.model_type == 1) {
      structure.has_virial = false;
      for (int n = 0; n < tokens.size(); ++n) {
        const std::string dipole_string = "dipole=";
        if (tokens[n].substr(0, dipole_string.length()) == dipole_string) {
          structure.has_virial = true;
          for (int m = 0; m < 6; ++m) {
            structure.virial[m] = 0.0f;
          }
          for (int m = 0; m < 3; ++m) {
            structure.virial[m] = get_double_from_token(
              tokens[n + m].substr(
                (m == 0) ? (dipole_string.length() + 1) : 0,
                (m == 2) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
              xyz_filename.c_str(),
              line_number);
            structure.virial[m] /= structure.num_atom;
          }
        }
      }
      if (!structure.has_virial) {
        if (para.prediction == 0) {
          PRINT_INPUT_ERROR("'dipole' is missing in the second line of a frame.");
        } else {
          for (int m = 0; m < 6; ++m) {
            structure.virial[m] = -1e6;
          }
        }
      }
    }
    // use the virial variable to keep the polarizability data
    if (model_parameters.model_type == 2) {
      structure.has_virial = false;
      for (int n = 0; n < tokens.size(); ++n) {
        const std::string pol_string = "pol=";
        if (tokens[n].substr(0, pol_string.length()) == pol_string) {
          structure.has_virial = true;
          const int reduced_index[9] = {0, 3, 5, 3, 1, 4, 5, 4, 2};
          for (int m = 0; m < 9; ++m) {
            structure.virial[reduced_index[m]] = get_double_from_token(
              tokens[n + m].substr(
                (m == 0) ? (pol_string.length() + 1) : 0,
                (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
              xyz_filename.c_str(),
              line_number);
            structure.virial[reduced_index[m]] /= structure.num_atom;
          }
        }
      }
      if (!structure.has_virial) {
        if (para.prediction == 0) {
          PRINT_INPUT_ERROR("'pol' is missing in the second line of a frame.");
        } else {
          for (int m = 0; m < 6; ++m) {
            structure.virial[m] = -1e6;
          }
        }
      }
    }
    // Retain the last atomic tensor property, as in the existing reader.
    int adipole_offset = record.get_property_offset("adipole", "atomic_dipole");
    int apol_offset = record.get_property_offset("apol", "atomic_polarizability");
    int avirial_offset = adipole_offset > apol_offset ? adipole_offset : apol_offset;
    label.has_atomic_virial = avirial_offset >= 0;
    label.atomic_virial_diag_only = adipole_offset > apol_offset;
    if (avirial_offset < 0 && model_parameters.atomic_v == 1) {
      if (model_parameters.model_type == 1) {
        PRINT_INPUT_ERROR("'adipole' or 'atomic_dipole' is missing in properties.");
      } else {
        PRINT_INPUT_ERROR("'apol' or 'atomic_polarizability' is missing in properties.");
      }
    }
    if (label.has_atomic_virial) {
      label.avirialxx.resize(structure.num_atom);
      label.avirialyy.resize(structure.num_atom);
      label.avirialzz.resize(structure.num_atom);
      if (!label.atomic_virial_diag_only) {
        label.avirialxy.resize(structure.num_atom);
        label.avirialyz.resize(structure.num_atom);
        label.avirialzx.resize(structure.num_atom);
      }
    }
    for (int na = 0; na < structure.num_atom; ++na) {
      const auto& tokens = record.atoms[na];
      line_number = record.comment_line_number + na + 1;
      if (label.has_atomic_virial) {
        if (label.atomic_virial_diag_only) {
          label.avirialxx[na] =
            get_double_from_token(tokens[0 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialyy[na] =
            get_double_from_token(tokens[1 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialzz[na] =
            get_double_from_token(tokens[2 + avirial_offset], xyz_filename.c_str(), line_number);
        } else {
          label.avirialxx[na] =
            get_double_from_token(tokens[0 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialyy[na] =
            get_double_from_token(tokens[4 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialzz[na] =
            get_double_from_token(tokens[8 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialxy[na] =
            get_double_from_token(tokens[3 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialyz[na] =
            get_double_from_token(tokens[7 + avirial_offset], xyz_filename.c_str(), line_number);
          label.avirialzx[na] =
            get_double_from_token(tokens[6 + avirial_offset], xyz_filename.c_str(), line_number);
        }
      }
    }
  }
};
} // namespace

TNEP::TNEP(const ParameterInput& input)
  : model_parameters(new TNEP_Parameters)
{
  auto& para = *model_parameters;
  para.model_type = input.model_type;
  para.read(input);
  para.lambda_e = para.lambda_f = 0.0f;
  para.enable_zbl = false;
  if (!para.is_lambda_v_set) {
    para.lambda_v = 1.0f;
  }
  configure(para);
  para.finish_reading(input, para.model_type == 1 ? "dipole" : "polarizability");
}

int TNEP::prepare_data()
{
  TNEP_Parameters& para = *model_parameters;
  TNEP_Reader reader(para, atomic_data);
  training.reset(new NEP_Training(para, reader));
  initialize(para, training->N, training->device_count);
  if (para.prediction == 0) {
    fitness_tensor.resize(para.population_size * (para.num_types + 1));
    training->open_loss_file(para, para.model_type == 1 ?
      " rmse_dipole_train rmse_dipole_test" :
      " rmse_polarizability_train rmse_polarizability_test", false);
  }
  return training->num_batches;
}

void TNEP::prepare_training()
{
  TNEP_Parameters& para = *model_parameters;
  int devices = training->device_count;
  std::vector<float> dummy_solution(para.number_of_variables * devices, para.initial_para);
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, dummy_solution.data(), training->train_set[batch],
      (para.fine_tune || para.import_q_scaler) ? false : true, devices);
  }
}

void TNEP::evaluate(int batch, const float* population)
{
  TNEP_Parameters& para = *model_parameters;
  int devices = training->device_count;
  for (int first = 0; first < para.population_size; first += devices) {
    int count = std::min(devices, para.population_size - first);
    find_force(para, population + first * para.number_of_variables,
      training->train_set[batch], false, count);
    for (int device = 0; device < count; ++device) {
      Dataset& data = training->train_set[batch][device];
      auto error = get_rmse(data, true, device);
      for (int t = 0; t <= para.num_types; ++t) {
        fitness_tensor[first + device + t * para.population_size] = para.lambda_v * error[t];
      }
    }
  }
}

void TNEP::accumulate_fitness(float* fitness_total)
{
  TNEP_Parameters& para = *model_parameters;
  for (int i = 0; i < para.population_size * (para.num_types + 1); ++i) {
    fitness_total[i] += fitness_tensor[i];
  }
}

void TNEP::print_training_header()
{
  printf("%-8s %-11s %-11s %-11s %-13s %-13s\n",
    "Step", "Total-Loss", "L1Reg-Loss", "L2Reg-Loss", "RMSE-P-Train", "RMSE-P-Test");
}

void TNEP::report_error(const TrainingReport& report)
{
  TNEP_Parameters& para = *model_parameters;
  find_force(para, report.parameters, training->train_set[report.batch], false, 1);
  Dataset& train_data = training->train_set[report.batch][0];
  float train = get_rmse(train_data, false, 0).back();
  float test = 0.0f;
  if (training->has_test_set) {
    find_force(para, report.parameters, training->test_set, false, 1);
    test = get_rmse(training->test_set[0], false, 0).back();
  }
  training->write_model(para, report);
  write_tnep_error(stdout, report, train, test);
  write_tnep_error(training->fid_loss_out, report, train, test);
  if (training->has_test_set) {
    Dataset& data = training->test_set[0];
    data.virial.copy_to_host(data.virial_cpu.data());
    std::vector<std::pair<Dataset*, int>> structures;
    for (int nc = 0; nc < data.Nc; ++nc) {
      structures.emplace_back(&data, nc);
    }
    write_tnep_predictions("test", structures, atomic_data, para.model_type == 1, para.atomic_v);
  }
}

void TNEP::predict(float* parameters)
{
  TNEP_Parameters& para = *model_parameters;
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, parameters, training->train_set[batch], false, 1);
    Dataset& data = training->train_set[batch][0];
    data.virial.copy_to_host(data.virial_cpu.data());
  }
  std::vector<std::pair<Dataset*, int>> structures;
  training->get_prediction_order(structures);
  write_tnep_predictions("train", structures, atomic_data, para.model_type == 1, para.atomic_v);
}

std::vector<float> TNEP::get_rmse(Dataset& dataset, bool use_weight, int device_id)
{
  TNEP_Parameters& para = *model_parameters;
  if (para.atomic_v) {
    return atomic_data.get(dataset).get_rmse(para, dataset, use_weight, device_id);
  }
  return dataset.get_rmse_virial(para, use_weight, device_id, para.model_type == 1 ? 3 : 6);
}

bool TNEP_Parameters::parse_keyword(const char** param, int num_param)
{
  if (strcmp(param[0], "atomic_v") == 0) {
    parse_atomic_v(param, num_param);
  } else {
    return NEP_Parameters::parse_keyword(param, num_param);
  }
  return true;
}

void TNEP_Parameters::parse_atomic_v(const char** param, int num_param)
{
  is_atomic_v_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("atomic_v should have 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &atomic_v)) {
    PRINT_INPUT_ERROR("atomic_v should be an integer.\n");
  }

  if (atomic_v != 0 && atomic_v != 1) {
    PRINT_INPUT_ERROR("atomic_v should = 0 or 1.");
  }
}

void TNEP_Parameters::report_model_inputs() const
{
  printf("    (%s) %satomic_v = %d.\n", is_atomic_v_set ? "input" : "default",
    is_atomic_v_set ? "  " : "", atomic_v);
}
