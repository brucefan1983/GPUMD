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

#include "dataset.cuh"
#include "mic.cuh"
#include "nep.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstring>

void NEP::initialize(Parameters& para, int N, int Nc, int deviceCount)
{
  initialize_model(para);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    initialize_device(para, N, device_id);
  }
  initialize_compiled_kernel(para);
}

void NEP::initialize_model(const Parameters& para)
{
  paramb.version = para.version;
  paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
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
#ifdef USE_CJ
  paramb.num_c_radial =
    paramb.num_types * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#else
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#endif

  zbl.enabled = para.enable_zbl;
  zbl.flexible = para.flexible_zbl;
  zbl.rc_inner = para.zbl_rc_inner;
  zbl.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    zbl.atomic_numbers[n] = para.atomic_numbers[n];        // starting from 1
  }
  if (zbl.flexible) {
    zbl.num_types = para.num_types;
    int num_type_zbl = (para.num_types * (para.num_types + 1)) / 2;
    for (int n = 0; n < num_type_zbl * 10; ++n) {
      zbl.para[n] = para.zbl_para[n];
    }
  }
}

void NEP::initialize_device(const Parameters& para, int N, int device_id)
{
  NEP_ANN& ann = annmb[device_id];
  NEP_Data& data = nep_data[device_id];
  ann.dim = para.dim;
  ann.num_neurons1 = para.num_neurons1;
  ann.num_hidden_layers = para.num_hidden_layers;
  ann.num_para = para.number_of_variables;
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

void NEP::initialize_compiled_kernel(const Parameters& para)
{
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    compiled_kernel_.reset(new NEP_Compile(
      make_nep_compile_config(para, NEP_Compile_Mode::NEP)));
    if (!compiled_kernel_->is_valid()) {
      compiled_kernel_.reset();
    }
  }
}

static void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler, float* g_q_scaler_max, float* g_q_scaler_min)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f; // a small number
  s_min[tid] = +1000000.0f; // a large number
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      if (q > s_max[tid]) {
        s_max[tid] = q;
      }
      if (q < s_min[tid]) {
        s_min[tid] = q;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) {
        s_max[tid] = s_max[tid + offset];
      }
      if (s_min[tid] > s_min[tid + offset]) {
        s_min[tid] = s_min[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler_max[bid] = max(g_q_scaler_max[bid], s_max[0]);
    g_q_scaler_min[bid] = min(g_q_scaler_min[bid], s_min[0]);
    g_q_scaler[bid] = 1.0f / (g_q_scaler_max[bid] - g_q_scaler_min[bid]);
  }
}

static __global__ void zero_force(
  const int N, float* g_fx, float* g_fy, float* g_fz, float* g_vxx, float* g_vyy, float* g_vzz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    g_vxx[n1] = 0.0f;
    g_vyy[n1] = 0.0f;
    g_vzz[n1] = 0.0f;
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP_Descriptor_Parameters paramb,
  const NEP::ZBL zbl,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1]; // starting from 1
    float pow_zi = pow(float(zi), 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2]; // starting from 1
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
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};

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
      s_pe += f * 0.5f;
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
    g_pe[n1] += s_pe;
  }
}

void NEP::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    bind_nep_parameters(
      paramb.num_types, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    neighbor[device_id].prepare(para, dataset[device_id], device_id);
    const int block_size = 32;
    const int grid_size = (dataset[device_id].N - 1) / block_size + 1;

    compute_nep_descriptors(
      paramb, annmb[device_id].c, dataset[device_id], neighbor[device_id],
      nep_data[device_id], compiled_kernel_.get());

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      FILE* fid_descriptor = my_fopen("descriptor.out", "a");
      std::vector<float> descriptor_cpu(nep_data[device_id].descriptors.size());
      nep_data[device_id].descriptors.copy_to_host(descriptor_cpu.data());
      for (int nc = 0; nc < dataset[device_id].Nc; ++nc) {
        float q_structure[MAX_DIM] = {0.0f};
        for (int na = 0; na < dataset[device_id].Na_cpu[nc]; ++na) {
          int n = dataset[device_id].Na_sum_cpu[nc] + na;
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            float q = descriptor_cpu[n + d * dataset[device_id].N] * para.q_scaler_cpu[d];
            q_structure[d] += q;
            if (para.output_descriptor == 2) {
              fprintf(fid_descriptor, "%g ", q);
            }
          }
          if (para.output_descriptor == 2) {
            fprintf(fid_descriptor, "\n");
          }
        }
        if (para.output_descriptor == 1) {
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            fprintf(fid_descriptor, "%g ", q_structure[d] / dataset[device_id].Na_cpu[nc]);
          }
        }
        if (para.output_descriptor == 1) {
          fprintf(fid_descriptor, "\n");
        }
      }
      fclose(fid_descriptor);
    }

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        dataset[device_id].N,
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        para.q_scaler_max[device_id].data(),
        para.q_scaler_min[device_id].data());
      GPU_CHECK_KERNEL
    }

    zero_force<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].virial.data() + dataset[device_id].N,
      dataset[device_id].virial.data() + dataset[device_id].N * 2);
    GPU_CHECK_KERNEL

    evaluate_nep_ann(
      annmb[device_id], para.model_type == 3, dataset[device_id], nep_data[device_id],
      para.q_scaler_gpu[device_id].data(), compiled_kernel_.get());

    compute_nep_forces(
      paramb, annmb[device_id].c, dataset[device_id], neighbor[device_id],
      nep_data[device_id], compiled_kernel_.get());

    if (zbl.enabled) {
      find_force_ZBL<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        zbl,
        dataset[device_id].NN_angular_sum.data(),
        dataset[device_id].NN_angular.data(),
        neighbor[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        neighbor[device_id].x12_angular.data(),
        neighbor[device_id].y12_angular.data(),
        neighbor[device_id].z12_angular.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data(),
        dataset[device_id].energy.data());
      GPU_CHECK_KERNEL
    }
  }
}
