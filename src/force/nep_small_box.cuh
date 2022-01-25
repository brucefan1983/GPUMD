/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "model/box.cuh"
#include "nep.cuh"
#include "utilities/common.cuh"
#include "utilities/nep_utilities.cuh"

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

static __device__ void apply_mic_small_box(
  const Box& box, const NEP2::ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
  if (box.triclinic == 0) {
    if (box.pbc_x == 1 && x12 < -ebox.h[3]) {
      x12 += ebox.h[0];
    } else if (box.pbc_x == 1 && x12 > +ebox.h[3]) {
      x12 -= ebox.h[0];
    }
    if (box.pbc_y == 1 && y12 < -ebox.h[4]) {
      y12 += ebox.h[1];
    } else if (box.pbc_y == 1 && y12 > +ebox.h[4]) {
      y12 -= ebox.h[1];
    }
    if (box.pbc_z == 1 && z12 < -ebox.h[5]) {
      z12 += ebox.h[2];
    } else if (box.pbc_z == 1 && z12 > +ebox.h[5]) {
      z12 -= ebox.h[2];
    }
  } else {
    double sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
    double sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
    double sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
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
}

static __global__ void find_neighbor_list_small_box(
  NEP2::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const NEP2::ExpandedBox ebox,
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
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
        for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
          for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }

            double delta[3];
            if (box.triclinic) {
              delta[0] = box.cpu_h[0] * ia + box.cpu_h[1] * ib + box.cpu_h[2] * ic;
              delta[1] = box.cpu_h[3] * ia + box.cpu_h[4] * ib + box.cpu_h[5] * ic;
              delta[2] = box.cpu_h[6] * ia + box.cpu_h[7] * ib + box.cpu_h[8] * ic;
            } else {
              delta[0] = box.cpu_h[0] * ia;
              delta[1] = box.cpu_h[1] * ib;
              delta[2] = box.cpu_h[2] * ic;
            }

            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(box, ebox, x12, y12, z12);

            float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
            if (distance_square < paramb.rc_radial * paramb.rc_radial) {
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = float(x12);
              g_y12_radial[count_radial * N + n1] = float(y12);
              g_z12_radial[count_radial * N + n1] = float(z12);
              count_radial++;
            }
            if (distance_square < paramb.rc_angular * paramb.rc_angular) {
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = float(x12);
              g_y12_angular[count_angular * N + n1] = float(y12);
              g_z12_angular[count_angular * N + n1] = float(z12);
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
  NEP2::ParaMB paramb,
  NEP2::ANN annmb,
  NEP2::ZBL zbl,
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
      if (zbl.enabled) {
        find_fc_nep_with_zbl(zbl.rc_inner, zbl.rc_outer, paramb.rc_radial, d12, fc12);
      } else {
        find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      }
      int t2 = g_type[n2];
      float fn12[MAX_NUM_N];
      find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float c = (paramb.num_types == 1)
                    ? 1.0f
                    : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
        q[n] += fn12[n] * c;
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
        if (zbl.enabled) {
          find_fc_nep_with_zbl(zbl.rc_inner, zbl.rc_outer, paramb.rc_angular, d12, fc12);
        } else {
          find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        }
        int t2 = g_type[n2];
        float fn;
        find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
        fn *=
          (paramb.num_types == 1)
            ? 1.0f
            : annmb
                .c[((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
        accumulate_s(d12, r12[0], r12[1], r12[2], fn, s);
      }
      find_q(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc] * YLM[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0, annmb.b0, annmb.w1, annmb.b1, q, F, Fp);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial_small_box(
  NEP2::ParaMB paramb,
  NEP2::ANN annmb,
  NEP2::ZBL zbl,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      if (zbl.enabled) {
        find_fc_and_fcp_nep_with_zbl(
          zbl.rc_inner, zbl.rc_outer, paramb.rc_radial, d12, fc12, fcp12);
      } else {
        find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      }
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float tmp12 = g_Fp[n1 + n * N] * fnp12[n] * d12inv;
        tmp12 *= (paramb.num_types == 1)
                   ? 1.0f
                   : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
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

static __global__ void find_force_angular_small_box(
  NEP2::ParaMB paramb,
  NEP2::ANN annmb,
  NEP2::ZBL zbl,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
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
    for (int d = 0; d < (paramb.n_max_angular + 1) * paramb.L_max; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      if (zbl.enabled) {
        find_fc_and_fcp_nep_with_zbl(
          zbl.rc_inner, zbl.rc_outer, paramb.rc_angular, d12, fc12, fcp12);
      } else {
        find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      }
      int t2 = g_type[n2];
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float fn;
        float fnp;
        find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);
        const float c =
          (paramb.num_types == 1)
            ? 1.0f
            : annmb
                .c[((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
        fn *= c;
        fnp *= c;
        accumulate_f12(
          n, n1, paramb.n_max_radial + 1, paramb.n_max_angular + 1, d12, r12, fn, fnp, Fp, sum_fxyz,
          f12);
      }
      f12[0] *= 2.0f;
      f12[1] *= 2.0f;
      f12[2] *= 2.0f;
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
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
  }
}

static __global__ void find_force_ZBL_small_box(
  const int N,
  const NEP2::ZBL zbl,
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
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    float zi = zbl.atomic_numbers[g_type[n1]];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      float zj = zbl.atomic_numbers[g_type[n2]];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
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
