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

#include "model/box.cuh"
#include "nep.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"

#ifdef USE_KEPLER
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
#endif

static __device__ void apply_mic_small_box(
  const Box& box, const NEP::ExpandedBox& ebox, float& x12, float& y12, float& z12)
{
  float sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
  float sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
  float sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
  if (box.pbc_x == 1)
    sx12 -= nearbyint(sx12);
  if (box.pbc_y == 1)
    sy12 -= nearbyint(sy12);
  if (box.pbc_z == 1)
    sz12 -= nearbyint(sz12);
  x12 = ebox.h[0] * sx12 + ebox.h[1] * sy12 + ebox.h[2] * sz12;
  y12 = ebox.h[3] * sx12 + ebox.h[4] * sy12 + ebox.h[5] * sz12;
  z12 = ebox.h[6] * sx12 + ebox.h[7] * sy12 + ebox.h[8] * sz12;
}

static __global__ void find_neighbor_list_small_box(
  NEP::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const NEP::ExpandedBox ebox,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
        for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
          for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }

            float delta[3];
            delta[0] = box.float_h[0] * ia + box.float_h[1] * ib + box.float_h[2] * ic;
            delta[1] = box.float_h[3] * ia + box.float_h[4] * ib + box.float_h[5] * ic;
            delta[2] = box.float_h[6] * ia + box.float_h[7] * ib + box.float_h[8] * ic;

            float x12 = g_x[n2] + delta[0] - x1;
            float y12 = g_y[n2] + delta[1] - y1;
            float z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(box, ebox, x12, y12, z12);

            float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);

            int t2 = g_type[n2];
            float rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;

            if (distance_square < rc_radial * rc_radial) {
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = x12;
              g_y12_radial[count_radial * N + n1] = y12;
              g_z12_radial[count_radial * N + n1] = z12;
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = x12;
              g_y12_angular[count_angular * N + n1] = y12;
              g_z12_angular[count_angular * N + n1] = z12;
              count_angular++;
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

static __global__ void find_descriptor_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
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
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_radial,
            paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
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
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
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

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

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
    g_pe[n1] += double(F) + (annmb.energy_reference ? annmb.energy_reference[t1] : 0.0);

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

static __global__ void find_descriptor_small_box(
  const float temperature,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
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
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
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
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // normalize descriptor
    q[annmb.dim - 1] = temperature;
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
    g_pe[n1] += double(F) + (annmb.energy_reference ? annmb.energy_reference[t1] : 0.0);

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_one_angular_channel_in_double(
  const int n,
  const int n_max_angular_plus_1,
  const double d12inv,
  const double fn,
  const double fnp,
  const float* Fp,
  const float* sum_fxyz,
  const double* r12unit,
  double* f12)
{
  double s[2 * L + 1];
  double contribution[3] = {0.0, 0.0, 0.0};
  calculate_s_one<L>(n, n_max_angular_plus_1, Fp, sum_fxyz, s);
  accumulate_f12_one<L>(d12inv, fn, fnp, s, r12unit, contribution);
  for (int d = 0; d < 3; ++d) {
    f12[d] += static_cast<double>(contribution[d]);
  }
}

static __device__ __forceinline__ void find_fn_and_fnp_in_double(
  const int n_max,
  const double rc,
  const double distance,
  double* fn,
  double* fnp)
{
  const double rcinv = 1.0 / rc;
  const double scaled = distance * rcinv;
  const double fc =
    distance < rc ? 0.5 * cos(3.14159265358979323846 * scaled) + 0.5 : 0.0;
  const double fcp =
    distance < rc
      ? -1.57079632679489661923 * sin(3.14159265358979323846 * scaled) * rcinv
      : 0.0;
  const double x = 2.0 * (scaled - 1.0) * (scaled - 1.0) - 1.0;
  fn[0] = fc;
  fnp[0] = fcp;
  fn[1] = (x + 1.0) * 0.5 * fc;
  fnp[1] =
    2.0 * (scaled - 1.0) * rcinv * fc + (x + 1.0) * 0.5 * fcp;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double t0 = 1.0;
  double t1 = x;
  for (int m = 2; m <= n_max; ++m) {
    const double t2 = 2.0 * x * t1 - t0;
    t0 = t1;
    t1 = t2;
    const double derivative_chebyshev = m * u1;
    const double u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
    const double basis = (t2 + 1.0) * 0.5;
    fnp[m] =
      derivative_chebyshev * 2.0 * (scaled - 1.0) * rcinv * fc + basis * fcp;
    fn[m] = basis * fc;
  }
}

static __device__ __forceinline__ void accumulate_f12_small_box_in_double(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  const double fn,
  const double fnp,
  const float* Fp,
  const float* sum_fxyz,
  double* f12)
{
  const bool only_supported_extra =
    !has_q_1111 && !has_q_112 && !has_q_123 && !has_q_233 && !has_q_134 &&
    num_L == L_max + static_cast<int>(has_q_222);
  if (!only_supported_extra) {
    float fallback[3] = {0.0f};
    const float d12_float = static_cast<float>(d12);
    const float r12_float[3] = {
      static_cast<float>(r12[0]),
      static_cast<float>(r12[1]),
      static_cast<float>(r12[2])};
    accumulate_f12(
      L_max,
      has_q_222,
      has_q_1111,
      has_q_112,
      has_q_123,
      has_q_233,
      has_q_134,
      num_L,
      n,
      n_max_angular_plus_1,
      d12_float,
      r12_float,
      static_cast<float>(fn),
      static_cast<float>(fnp),
      Fp,
      sum_fxyz,
      fallback);
    for (int d = 0; d < 3; ++d) {
      f12[d] += static_cast<double>(fallback[d]);
    }
    return;
  }

  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {
    r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};
  if (L_max >= 1)
    accumulate_one_angular_channel_in_double<1>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 2)
    accumulate_one_angular_channel_in_double<2>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 3)
    accumulate_one_angular_channel_in_double<3>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 4)
    accumulate_one_angular_channel_in_double<4>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 5)
    accumulate_one_angular_channel_in_double<5>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 6)
    accumulate_one_angular_channel_in_double<6>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 7)
    accumulate_one_angular_channel_in_double<7>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 8)
    accumulate_one_angular_channel_in_double<8>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);

  if (has_q_222) {
    double radial_fn = fn * d12inv;
    double radial_fnp = fnp * d12inv - fn * d12inv * d12inv;
    const double fn2 = radial_fn * d12inv;
    const double fnp2 =
      radial_fnp * d12inv - radial_fn * d12inv * d12inv;
    const double s2[5] = {
      sum_fxyz[n * NUM_OF_ABC + 3],
      sum_fxyz[n * NUM_OF_ABC + 4],
      sum_fxyz[n * NUM_OF_ABC + 5],
      sum_fxyz[n * NUM_OF_ABC + 6],
      sum_fxyz[n * NUM_OF_ABC + 7]};
    double contribution[3] = {0.0, 0.0, 0.0};
    get_f12_4body(
      d12,
      d12inv,
      fn2,
      fnp2,
      static_cast<double>(Fp[L_max * n_max_angular_plus_1 + n]),
      s2,
      r12,
      contribution);
    for (int d = 0; d < 3; ++d) {
      f12[d] += static_cast<double>(contribution[d]);
    }
  }
}

template <bool AccumulateInDouble = false, typename R12 = float>
static __global__ void find_force_radial_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const R12* __restrict__ g_x12,
  const R12* __restrict__ g_y12,
  const R12* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12_double[3] = {
        static_cast<double>(g_x12[index]),
        static_cast<double>(g_y12[index]),
        static_cast<double>(g_z12[index])};
      float r12[3] = {
        static_cast<float>(r12_double[0]),
        static_cast<float>(r12_double[1]),
        static_cast<float>(r12_double[2])};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      const double d12_double = sqrt(
        r12_double[0] * r12_double[0] +
        r12_double[1] * r12_double[1] +
        r12_double[2] * r12_double[2]);
      float f12[3] = {0.0f};
      double f12_double[3] = {0.0, 0.0, 0.0};
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      double fn12_double[MAX_NUM_N] = {0.0};
      double fnp12_double[MAX_NUM_N] = {0.0};
      if (AccumulateInDouble) {
        find_fn_and_fnp_in_double(
          paramb.basis_size_radial,
          static_cast<double>(rc),
          d12_double,
          fn12_double,
          fnp12_double);
      }
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        double gnp12_double = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
          if (AccumulateInDouble) {
            gnp12_double +=
              fnp12_double[k] * static_cast<double>(annmb.c_type_pair[c_index]);
          }
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        const double tmp12_double =
          static_cast<double>(g_Fp[n1 + n * N]) * gnp12_double / d12_double;
        for (int d = 0; d < 3; ++d) {
          const float contribution = tmp12 * r12[d];
          if (AccumulateInDouble) {
            f12_double[d] += tmp12_double * r12_double[d];
          } else {
            f12[d] += contribution;
          }
        }
      }
      if (!AccumulateInDouble) {
        for (int d = 0; d < 3; ++d) {
          f12_double[d] = static_cast<double>(f12[d]);
        }
      }
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;
      if (AccumulateInDouble) {
        s_sxx -= r12_double[0] * f12_double[0];
        s_syy -= r12_double[1] * f12_double[1];
        s_szz -= r12_double[2] * f12_double[2];
        s_sxy -= r12_double[0] * f12_double[1];
        s_sxz -= r12_double[0] * f12_double[2];
        s_syz -= r12_double[1] * f12_double[2];
        s_syx -= r12_double[1] * f12_double[0];
        s_szx -= r12_double[2] * f12_double[0];
        s_szy -= r12_double[2] * f12_double[1];
      } else {
        s_sxx -= r12[0] * f12[0];
        s_syy -= r12[1] * f12[1];
        s_szz -= r12[2] * f12[2];
        s_sxy -= r12[0] * f12[1];
        s_sxz -= r12[0] * f12[2];
        s_syz -= r12[1] * f12[2];
        s_syx -= r12[1] * f12[0];
        s_szx -= r12[2] * f12[0];
        s_szy -= r12[2] * f12[1];
      }

      atomicAdd(&g_fx[n1], f12_double[0]);
      atomicAdd(&g_fy[n1], f12_double[1]);
      atomicAdd(&g_fz[n1], f12_double[2]);
      atomicAdd(&g_fx[n2], -f12_double[0]);
      atomicAdd(&g_fy[n2], -f12_double[1]);
      atomicAdd(&g_fz[n2], -f12_double[2]);
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);
    }
  }
}

template <bool AccumulateInDouble = false, typename R12 = float>
static __global__ void find_force_angular_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const R12* __restrict__ g_x12,
  const R12* __restrict__ g_y12,
  const R12* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] = 
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12_double[3] = {
        static_cast<double>(g_x12[index]),
        static_cast<double>(g_y12[index]),
        static_cast<double>(g_z12[index])};
      float r12[3] = {
        static_cast<float>(r12_double[0]),
        static_cast<float>(r12_double[1]),
        static_cast<float>(r12_double[2])};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      double f12_double[3] = {0.0, 0.0, 0.0};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      const double d12_double = sqrt(
        r12_double[0] * r12_double[0] +
        r12_double[1] * r12_double[1] +
        r12_double[2] * r12_double[2]);
      double fn12_double[MAX_NUM_N] = {0.0};
      double fnp12_double[MAX_NUM_N] = {0.0};
      if (AccumulateInDouble) {
        find_fn_and_fnp_in_double(
          paramb.basis_size_angular,
          static_cast<double>(rc),
          d12_double,
          fn12_double,
          fnp12_double);
      }
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        double gn12_double = 0.0;
        double gnp12_double = 0.0;
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
          if (AccumulateInDouble) {
            gn12_double +=
              fn12_double[k] * static_cast<double>(annmb.c_type_pair[c_index]);
            gnp12_double +=
              fnp12_double[k] * static_cast<double>(annmb.c_type_pair[c_index]);
          }
        }
        if (AccumulateInDouble) {
          accumulate_f12_small_box_in_double(
            paramb.L_max,
            paramb.has_q_222,
            paramb.has_q_1111,
            paramb.has_q_112,
            paramb.has_q_123,
            paramb.has_q_233,
            paramb.has_q_134,
            paramb.num_L,
            n,
            paramb.n_max_angular + 1,
            d12_double,
            r12_double,
            gn12_double,
            gnp12_double,
            Fp,
            sum_fxyz,
            f12_double);
        } else {
          accumulate_f12(
            paramb.L_max,
            paramb.has_q_222,
            paramb.has_q_1111,
            paramb.has_q_112,
            paramb.has_q_123,
            paramb.has_q_233,
            paramb.has_q_134,
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
      }
      if (!AccumulateInDouble) {
        for (int d = 0; d < 3; ++d) {
          f12_double[d] = static_cast<double>(f12[d]);
        }
      }
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;
      if (AccumulateInDouble) {
        s_sxx -= r12_double[0] * f12_double[0];
        s_syy -= r12_double[1] * f12_double[1];
        s_szz -= r12_double[2] * f12_double[2];
        s_sxy -= r12_double[0] * f12_double[1];
        s_sxz -= r12_double[0] * f12_double[2];
        s_syz -= r12_double[1] * f12_double[2];
        s_syx -= r12_double[1] * f12_double[0];
        s_szx -= r12_double[2] * f12_double[0];
        s_szy -= r12_double[2] * f12_double[1];
      } else {
        s_sxx -= r12[0] * f12[0];
        s_syy -= r12[1] * f12[1];
        s_szz -= r12[2] * f12[2];
        s_sxy -= r12[0] * f12[1];
        s_sxz -= r12[0] * f12[2];
        s_syz -= r12[1] * f12[2];
        s_syx -= r12[1] * f12[0];
        s_szx -= r12[2] * f12[0];
        s_szy -= r12[2] * f12[1];
      }

      atomicAdd(&g_fx[n1], f12_double[0]);
      atomicAdd(&g_fy[n1], f12_double[1]);
      atomicAdd(&g_fz[n1], f12_double[2]);
      atomicAdd(&g_fx[n2], -f12_double[0]);
      atomicAdd(&g_fy[n2], -f12_double[1]);
      atomicAdd(&g_fz[n2], -f12_double[2]);
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);
    }
  }
}

static __global__ void find_force_ZBL_small_box(
  NEP::ParaMB paramb,
  const int N,
  const NEP::ZBL zbl,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
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
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      atomicAdd(&g_virial[n2 + 0 * N], double(-r12[0] * f12[0]));
      atomicAdd(&g_virial[n2 + 1 * N], double(-r12[1] * f12[1]));
      atomicAdd(&g_virial[n2 + 2 * N], double(-r12[2] * f12[2]));
      atomicAdd(&g_virial[n2 + 3 * N], double(-r12[0] * f12[1]));
      atomicAdd(&g_virial[n2 + 4 * N], double(-r12[0] * f12[2]));
      atomicAdd(&g_virial[n2 + 5 * N], double(-r12[1] * f12[2]));
      atomicAdd(&g_virial[n2 + 6 * N], double(-r12[1] * f12[0]));
      atomicAdd(&g_virial[n2 + 7 * N], double(-r12[2] * f12[0]));
      atomicAdd(&g_virial[n2 + 8 * N], double(-r12[2] * f12[1]));
      s_pe += f * 0.5f;
    }
    g_pe[n1] += s_pe;
  }
}
