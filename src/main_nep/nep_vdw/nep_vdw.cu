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
#include "nep_vdw.cuh"
#include "main_nep/common/nep_output.cuh"
#include "main_nep/xyz_reader.cuh"
#include "main_nep/common/nep_energy_reader.cuh"
#include "main_nep/common/nep_model.cuh"
#include "main_nep/common/nep_init.cuh"
#include "main_nep/common/nep_io.cuh"
#include "main_nep/common/nep_long_range.cuh"
#include "main_nep/common/nep_vdw_parameters.cuh"
#include "main_nep/common/nep_parameters.cuh"
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
  const NEP_VDW::ParaMB paramb,
  const NEP_VDW::ANN annmb,
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
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index, n, k, paramb.n_max_radial, paramb.basis_size_radial);
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
  const NEP_VDW::ParaMB paramb,
  const NEP_VDW::ANN annmb,
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
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
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
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
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

void NEP_VDW::initialize(NEP_Energy_Parameters& para, int N, int Nc, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    initialize_device(para, N, Nc, device_id);
  }
  initialize_compiled_kernel(para);
}

void NEP_VDW::initialize_model(const NEP_Energy_Parameters& para)
{
  paramb.version = para.version;
  paramb.num_types = para.num_types;
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial[t] = para.rc_radial[t];
    paramb.rc_angular[t] = para.rc_angular[t];
    int z = para.atomic_numbers[t];
    if (z < 1 || z > max_elem_vdw) {
      PRINT_INPUT_ERROR("NEP-vdW currently supports elements from H to Pu.");
    }
    paramb.c6_ref_sqrt[t] = c6_ref_sqrt[z - 1];
  }
  vdw_para.alpha = float(PI) / para.rc_radial_max;
  vdw_para.alpha_factor = 0.25f / (vdw_para.alpha * vdw_para.alpha);
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
#ifdef USE_CJ
  paramb.num_c_radial =
    paramb.num_types * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#else
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#endif

  initialize_nep_zbl(para, zbl);
}

void NEP_VDW::initialize_device(const NEP_Energy_Parameters& para, int N, int Nc, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  annmb[device_id].dim = para.dim;
  annmb[device_id].num_neurons1 = para.num_neurons1;
  annmb[device_id].num_hidden_layers = para.num_hidden_layers;
  annmb[device_id].num_para = para.number_of_variables;
  if (para.num_hidden_layers == 2) {
    annmb[device_id].num_neurons2 = para.num_neurons2;
  }
  annmb[device_id].one_ann_no_bias = para.number_of_variables_ann_1;

  nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
  nep_data[device_id].Fp.resize(N * annmb[device_id].dim);
  nep_data[device_id].C6.resize(N);
  nep_data[device_id].C6_derivative.resize(N * annmb[device_id].dim);
  nep_data[device_id].D_C6.resize(N);
  nep_data[device_id].sum_fxyz.resize(N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data[device_id].parameters.resize(annmb[device_id].num_para);
  nep_data[device_id].kpoint_offset.resize(Nc + 1);
}

namespace
{
NEP_Compile_Config make_compile_config(
  const NEP_Energy_Parameters& para, const float* c6_ref_sqrt)
{
  NEP_Compile_Config config = make_nep_compile_config(para);
  config.model_name = "NEP-vdW";
  config.source_file = "main_nep/nep_vdw/nep_vdw_specialized.cu";
  config.extra_config =
    "#define C6_SCALING_FACTOR_JIT 1.000000000e-01f\n"
    "static __device__ __constant__ float C6_REF_SQRT_JIT[NUM_TYPES_JIT] = " +
    make_nep_compile_array(std::vector<float>(c6_ref_sqrt, c6_ref_sqrt + para.num_types)) + ";\n";
  return config;
}
} // namespace

NEP_VDW::Compiled_Kernel::Compiled_Kernel(const NEP_Compile_Config& config)
  : library(config)
{
  if (!library.is_valid()) {
    return;
  }
  valid = common.load(library);
  valid &= library.load_symbol("nep_train_launch_ann_vdw", ann_vdw_);
}

void NEP_VDW::Compiled_Kernel::launch_ann_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
  float* Fp,
  float* C6,
  float* C6_derivative)
{
  check_nep_launch(
    ann_vdw_(
      N, type, descriptors, q_scaler, parameters,
      pe, Fp, C6, C6_derivative),
    "ann_vdw_jit");
}

void NEP_VDW::Compiled_Kernel::launch_force_vdw_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* C6_derivative, const float* D_C6,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      nullptr, nullptr, C6_derivative, D_C6,
      0, fx, fy, fz, virial),
    "force_vdw_radial_jit");
}

void NEP_VDW::Compiled_Kernel::launch_force_vdw_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  const float* C6_derivative, const float* D_C6,
  const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      nullptr, nullptr, C6_derivative, D_C6,
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_vdw_angular_jit");
}

void NEP_VDW::initialize_compiled_kernel(const NEP_Energy_Parameters& para)
{
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    compiled_kernel_.reset(new Compiled_Kernel(
      make_compile_config(para, paramb.c6_ref_sqrt)));
    if (!compiled_kernel_->is_valid()) {
      compiled_kernel_.reset();
    }
  }
}

void NEP_VDW::bind_parameters(const float* parameters, ANN& ann)
{
  const float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.wb[t] = pointer;
    pointer += ann.one_ann_no_bias;
  }
  ann.b = pointer;
  pointer += 1;
  ann.c = pointer;
}

static __global__ void apply_ann_vdw(
  const int N,
  const NEP_VDW::ParaMB paramb,
  const NEP_VDW::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp,
  float* g_C6,
  float* g_C6_derivative)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];

    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy, C6, and their gradients
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    float C6 = 0.0f;
    float C6_derivative[MAX_DIM] = {0.0f};

    const int neu1 = annmb.num_neurons1;
    const int neu1_dim = neu1 * annmb.dim;
    apply_ann_one_layer_vdw(
      annmb.dim,
      neu1,
      annmb.wb[type],
      annmb.wb[type] + neu1_dim,
      annmb.wb[type] + neu1 * (annmb.dim + 1),
      annmb.b,
      q,
      F,
      Fp,
      C6,
      C6_derivative);

    g_pe[n1] = static_cast<double>(F);
    const float SCALING_FACTOR = 0.1f;
    float C6_exp = paramb.c6_ref_sqrt[type] * expf(C6 * SCALING_FACTOR);
    g_C6[n1] = C6_exp;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
      g_C6_derivative[n1 + d * N] =
        C6_exp * SCALING_FACTOR * C6_derivative[d] * g_q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_VDW::ParaMB paramb,
  const NEP_VDW::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_C6_derivative,
  const float* __restrict__ g_D_C6,
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
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * N];
        tmp12 += g_C6_derivative[n1 + n * N] * g_D_C6[n1];
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
  const NEP_VDW::ParaMB paramb,
  const NEP_VDW::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_C6_derivative,
  const float* __restrict__ g_D_C6,
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
      Fp[d] += g_C6_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_C6[n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] = 
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
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
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
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
  float* g_G_vdw,
  float* g_G_vdw_virial)
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
          if (n1 == 0 && (n2 < 0 || (n2 == 0 && n3 < 0))) continue;
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
              const float sqrt_pi = 1.77245385f;
              if (ksq == 0.0f) {
                // k=0 occurs once, unlike each nonzero +/-k pair.
                g_G_vdw[nc_nk] =
                  -abs(two_pi_over_det) * sqrt_pi *
                  alpha * alpha * alpha / 12.0f;
                g_G_vdw_virial[nc_nk] = 0.0f;
              } else {
                const float b2 = ksq * alpha_factor;
                const float b = sqrt(b2);
                const float exp_b2 = exp(-b2);
                const float k = sqrt(ksq);
                const float erfc_b = erfc(b);
                const float prefactor = abs(two_pi_over_det) * sqrt_pi / 24.0f;
                const float c1 = -k * ksq *
                  (sqrt_pi * erfc_b + (0.5f / b2 - 1.0f) * exp_b2 / b);
                const float c2 = 3.0f * k * (sqrt_pi * erfc_b - exp_b2 / b);
                g_G_vdw[nc_nk] = prefactor * c1;
                g_G_vdw_virial[nc_nk] = prefactor * c2;
              }
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

void NEP_VDW::prepare_kpoints(Dataset& dataset, int device_id)
{
  NEP_VDW_Data& data = nep_data[device_id];

  // Training datasets and their boxes are immutable after construction.
  if (data.kpoint_dataset == &dataset) {
    return;
  }

  const int grid_size = (dataset.Nc - 1) / 64 + 1;
  find_k_and_G<<<grid_size, 64>>>(
    dataset.Nc,
    vdw_para.alpha,
    vdw_para.alpha_factor,
    dataset.box_original.data(),
    dataset.pbc.data(),
    nullptr,
    data.kpoint_offset.data(),
    nullptr,
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
    data.G_vdw.resize(required_capacity);
    data.G_vdw_virial.resize(required_capacity);
    data.S_real.resize(required_capacity);
    data.S_imag.resize(required_capacity);
  }

  find_k_and_G<<<grid_size, 64>>>(
    dataset.Nc,
    vdw_para.alpha,
    vdw_para.alpha_factor,
    dataset.box_original.data(),
    dataset.pbc.data(),
    data.kpoint_offset.data(),
    nullptr,
    data.kx.data(),
    data.ky.data(),
    data.kz.data(),
    data.G_vdw.data(),
    data.G_vdw_virial.data());
  GPU_CHECK_KERNEL
  data.kpoint_dataset = &dataset;
}

void NEP_VDW::compute_descriptors(
  const ANN& ann, const Dataset& data, const NEP_Neighbor& neighbors,
  NEP_VDW_Data& storage)
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

void NEP_VDW::evaluate_ann(
  const ANN& ann, Dataset& data,
  NEP_VDW_Data& storage, const float* q_scaler)
{
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->launch_ann_vdw(
      data.N,
      data.type.data(),
      storage.descriptors.data(),
      q_scaler,
      storage.parameters.data(),
      data.energy.data(),
      storage.Fp.data(),
      storage.C6.data(),
      storage.C6_derivative.data());
  } else {
    apply_ann_vdw<<<grid_size, block_size>>>(
      data.N,
      paramb,
      ann,
      data.type.data(),
      storage.descriptors.data(),
      q_scaler,
      data.energy.data(),
      storage.Fp.data(),
      storage.C6.data(),
      storage.C6_derivative.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_VDW::compute_forces(
  const ANN& ann, Dataset& data, const NEP_Neighbor& neighbors,
  NEP_VDW_Data& storage)
{
  const int block_size = 32;
  const int grid_size = (data.N - 1) / block_size + 1;

  if (compiled_kernel_) {
    compiled_kernel_->launch_force_vdw_radial(
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
      storage.C6_derivative.data(),
      storage.D_C6.data(),
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
      storage.C6_derivative.data(),
      storage.D_C6.data(),
      data.force.data(),
      data.force.data() + data.N,
      data.force.data() + data.N * 2,
      data.virial.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel_) {
    compiled_kernel_->launch_force_vdw_angular(
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
      storage.C6_derivative.data(),
      storage.D_C6.data(),
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
      storage.C6_derivative.data(),
      storage.D_C6.data(),
      storage.sum_fxyz.data(),
      data.force.data(),
      data.force.data() + data.N,
      data.force.data() + data.N * 2,
      data.virial.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_VDW::find_force(
  NEP_Energy_Parameters& para,
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

    // Long range: reciprocal-space Ewald for PPP and direct summation for FFF.
    prepare_kpoints(data, device_id);

    compute_nep_structure_factor(
      data, storage.C6.data(), storage.kpoint_offset.data(),
      storage.kx.data(), storage.ky.data(), storage.kz.data(),
      storage.S_real.data(), storage.S_imag.data());

    add_nep_vdw_long_range(
      data, vdw_para.alpha, storage.C6.data(), storage.kpoint_offset.data(),
      storage.kx.data(), storage.ky.data(), storage.kz.data(),
      storage.G_vdw.data(), storage.G_vdw_virial.data(),
      storage.S_real.data(), storage.S_imag.data(), storage.D_C6.data());

    compute_forces(ann, data, neighbors, storage);

    if (zbl.enabled) {
      add_nep_zbl(zbl, data, neighbors);
    }
  }
}

void NEP_VDW::configure(NEP_Energy_Parameters& para)
{
  para.file_tag = std::string(para.enable_zbl ? "nep4_zbl" : "nep4") + "_vdw";
  calculate_nep_model_parameters(para, 0, 1, 1, 1);
}

void NEP_VDW::initialize_parameters(
  std::mt19937& rng,
  std::vector<float>& mu, std::vector<float>& sigma)
{
  NEP_Energy_Parameters& para = *model_parameters;
  initialize_nep_mu_and_sigma(para, rng, mu, sigma, 1);
}

void NEP_VDW::parameter_groups(std::vector<int>& groups)
{
  NEP_Energy_Parameters& para = *model_parameters;
  find_nep_type_of_variable(para, groups, 1, 1);
}

void NEP_VDW::read_model(float* parameters)
{
  read_nep_model("nep.txt", *model_parameters, parameters);
}

void NEP_VDW::write_restart(
  const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma)
{
  write_nep_restart(filename, *model_parameters, mu, sigma);
}

namespace {
class NEP_VDW_Reader : public NEP_Energy_Reader
{
public:
  NEP_VDW_Reader(NEP_Energy_Parameters& para, EnergyDifference& energy_difference)
    : NEP_Energy_Reader(para, energy_difference) {}

private:
  void read_structure(const XYZ_Record& record, Structure& structure) override
  {
    read_nep_boundaries(record, structure);
    NEP_Energy_Reader::read_structure(record, structure);
  }

  void print_summary(const std::vector<Structure>& structures) const override
  {
    print_nep_boundaries(structures);
    NEP_Energy_Reader::print_summary(structures);
  }
};
} // namespace

NEP_VDW::NEP_VDW(const ParameterInput& input)
  : model_parameters(new NEP_Energy_Parameters)
{
  auto& para = *model_parameters;
  para.read(input);
  if (para.num_hidden_layers == 2) {
    PRINT_INPUT_ERROR("Can only use one hidden layer.");
  }
  configure(para);
  para.finish_reading(input, "potential", "add environment-dependent vdW to ordinary NEP.");
}

int NEP_VDW::prepare_data()
{
  NEP_Energy_Parameters& para = *model_parameters;
  NEP_VDW_Reader reader(para, energy_difference);
  training.reset(new NEP_Training(para, reader));
  initialize(para, training->N, training->Nc, training->device_count);
  if (para.prediction == 0) {
    energy_loss.initialize(para);
    training->open_loss_file(para,
      " rmse_energy_train rmse_force_train rmse_virial_train"
      " rmse_energy_test rmse_force_test rmse_virial_test", para.has_ediff_combinations);
  }
  return training->num_batches;
}

void NEP_VDW::prepare_training()
{
  NEP_Energy_Parameters& para = *model_parameters;
  int devices = training->device_count;
  std::vector<float> dummy_solution(para.number_of_variables * devices, para.initial_para);
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, dummy_solution.data(), training->train_set[batch],
      (para.fine_tune || para.import_q_scaler) ? false : true, devices);
  }
}

void NEP_VDW::evaluate(int batch, const float* population)
{
  NEP_Energy_Parameters& para = *model_parameters;
  int devices = training->device_count;
  for (int first = 0; first < para.population_size; first += devices) {
    int count = std::min(devices, para.population_size - first);
    find_force(para, population + first * para.number_of_variables,
      training->train_set[batch], false, count);
    for (int device = 0; device < count; ++device) {
      Dataset& data = training->train_set[batch][device];
      energy_loss.evaluate(para, data, energy_difference, batch, first + device, device);
    }
  }
}

void NEP_VDW::accumulate_fitness(float* fitness_total)
{
  NEP_Energy_Parameters& para = *model_parameters;
  energy_loss.accumulate_fitness(para, fitness_total);
  energy_loss.accumulate_energy_difference(para, fitness_total);
}

void NEP_VDW::print_training_header()
{
  print_nep_energy_header(model_parameters->has_ediff_combinations);
}

void NEP_VDW::report_error(const TrainingReport& report)
{
  NEP_Energy_Parameters& para = *model_parameters;
  find_force(para, report.parameters, training->train_set[report.batch], false, 1);
  Dataset& train_data = training->train_set[report.batch][0];
  NEP_Energy_Error train = energy_loss.get_error(
    para, train_data, energy_difference, true, report.batch);
  // Correct the last bias parameter in the energy network before evaluating the test set.
  report.parameters[para.number_of_variables_ann - 1] += train.energy_shift;
  NEP_Energy_Error test;
  if (training->has_test_set) {
    find_force(para, report.parameters, training->test_set, false, 1);
    test = energy_loss.get_error(para, training->test_set[0], energy_difference, false, 0);
  }
  training->write_model(para, report);
  write_nep_energy_error(stdout, report, train, test, para.has_ediff_combinations);
  write_nep_energy_error(training->fid_loss_out, report, train, test, para.has_ediff_combinations);
  if (training->has_test_set) {
    Dataset& data = training->test_set[0];
    copy_nep_energy_predictions(data);
    std::vector<std::pair<Dataset*, int>> structures;
    for (int nc = 0; nc < data.Nc; ++nc) {
      structures.emplace_back(&data, nc);
    }
    write_nep_energy_predictions("test", structures);
  }
}

void NEP_VDW::predict(float* parameters)
{
  NEP_Energy_Parameters& para = *model_parameters;
  for (int batch = 0; batch < training->num_batches; ++batch) {
    find_force(para, parameters, training->train_set[batch], false, 1);
    Dataset& data = training->train_set[batch][0];
    copy_nep_energy_predictions(data);
  }
  std::vector<std::pair<Dataset*, int>> structures;
  training->get_prediction_order(structures);
  write_nep_energy_predictions("train", structures);
}
