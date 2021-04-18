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

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., in preparison.
------------------------------------------------------------------------------*/

#include "nep2.cuh"
#include "utilities/error.cuh"
#include <vector>

// set by me:
const int MAX_NUM_NEURONS_PER_LAYER = 40; // largest ANN: input-40-40-output
const int MAX_NUM_N = 9;                  // n_max+1 = 8+1
const int MAX_NUM_L = 9;                  // L_max+1 = 8+1
// calculated:
const int MAX_DIM = MAX_NUM_N * MAX_NUM_L;
const int MAX_2B_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + 1) + 1;
const int MAX_3B_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + 3) + 1;
const int MAX_MB_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + MAX_DIM) + 1;
const int MAX_ANN_SIZE = MAX_2B_SIZE + MAX_3B_SIZE + MAX_MB_SIZE;
// constant memory
__constant__ float c_parameters[MAX_ANN_SIZE];

NEP2::NEP2(FILE* fid, const Neighbor& neighbor)
{
  printf("Use the NEP potential.\n");
  char name[20];

  int count = fscanf(fid, "%s%d%f", name, &ann2b.num_neurons_per_layer, &para2b.rc);
  PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
  printf("two_body: %d neurons, %g A.\n", ann2b.num_neurons_per_layer, para2b.rc);

  count = fscanf(fid, "%s%d%f", name, &ann3b.num_neurons_per_layer, &para3b.rc);
  PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
  printf("three_body: %d neurons, %g A.\n", ann3b.num_neurons_per_layer, para3b.rc);

  count = fscanf(fid, "%s%d%d%d", name, &annmb.num_neurons_per_layer, &paramb.n_max, &paramb.L_max);
  PRINT_SCANF_ERROR(count, 4, "reading error for NEP potential.");
  printf(
    "many_body: %d neurons, n_max = %d, l_max = %d.\n", annmb.num_neurons_per_layer, paramb.n_max,
    paramb.L_max);

  ann2b.num_neurons1 = ann2b.num_neurons_per_layer;
  ann2b.num_neurons2 = ann2b.num_neurons_per_layer;
  ann3b.num_neurons1 = ann3b.num_neurons_per_layer;
  ann3b.num_neurons2 = ann3b.num_neurons_per_layer;
  annmb.num_neurons1 = annmb.num_neurons_per_layer;
  annmb.num_neurons2 = annmb.num_neurons_per_layer;

  rc = para2b.rc; // largest cutoff

  // 2body
  ann2b.dim = 1;
  ann2b.num_para =
    ann2b.num_neurons_per_layer > 0
      ? ann2b.num_neurons_per_layer * (ann2b.num_neurons_per_layer + ann2b.dim + 3) + 1
      : 0;
  para2b.rcinv = 1.0f / para2b.rc;
  // 3body
  ann3b.dim = 3;
  ann3b.num_para =
    ann3b.num_neurons_per_layer > 0
      ? ann3b.num_neurons_per_layer * (ann3b.num_neurons_per_layer + ann3b.dim + 3) + 1
      : 0;
  para3b.rcinv = 1.0f / para3b.rc;
  // manybody
  paramb.rc = para2b.rc; // manybody has the same cutoff as twobody
  paramb.rcinv = 1.0f / paramb.rc;
  annmb.dim = (paramb.n_max + 1) * (paramb.L_max + 1);
  annmb.num_para =
    annmb.num_neurons_per_layer > 0
      ? annmb.num_neurons_per_layer * (annmb.num_neurons_per_layer + annmb.dim + 3) + 1
      : 0;

  if (ann3b.num_neurons_per_layer > 0) {
    nep_data.NN3b.resize(neighbor.NN.size());
    nep_data.NL3b.resize(neighbor.NN.size() * neighbor.MN);
  }
  if (ann3b.num_neurons_per_layer > 0 || annmb.num_neurons_per_layer > 0) {
    nep_data.f12x.resize(neighbor.NN.size() * neighbor.MN);
    nep_data.f12y.resize(neighbor.NN.size() * neighbor.MN);
    nep_data.f12z.resize(neighbor.NN.size() * neighbor.MN);
  }
  update_potential(fid);
}

NEP2::~NEP2(void)
{
  // nothing
}

void NEP2::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons_per_layer * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons_per_layer;
  ann.b1 = ann.w1 + ann.num_neurons_per_layer * ann.num_neurons_per_layer;
  ann.w2 = ann.b1 + ann.num_neurons_per_layer;
  ann.b2 = ann.w2 + ann.num_neurons_per_layer;
}

void NEP2::update_potential(FILE* fid)
{
  const int num_para = ann2b.num_para + ann3b.num_para + annmb.num_para;
  std::vector<float> parameters(num_para);
  for (int n = 0; n < num_para; ++n) {
    int count = fscanf(fid, "%f", &parameters[n]);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
  }
  CHECK(cudaMemcpyToSymbol(c_parameters, parameters.data(), sizeof(float) * num_para));
  float* address_c_parameters;
  CHECK(cudaGetSymbolAddress((void**)&address_c_parameters, c_parameters));
  if (ann2b.num_neurons_per_layer > 0) {
    update_potential(address_c_parameters, ann2b);
  }
  if (ann3b.num_neurons_per_layer > 0) {
    update_potential(address_c_parameters + ann2b.num_para, ann3b);
  }
  if (annmb.num_neurons_per_layer > 0) {
    update_potential(address_c_parameters + ann2b.num_para + ann3b.num_para, annmb);
  }
}

static __device__ void
apply_ann(const NEP2::ANN& ann, float* q, float& energy, float* energy_derivative)
{
  // energy
  float x1[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 1st hidden layer neurons
  float x2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 2nd hidden layer neurons
  for (int n = 0; n < ann.num_neurons1; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < ann.dim; ++d) {
      w0_times_q += ann.w0[n * ann.dim + d] * q[d];
    }
    x1[n] = tanh(w0_times_q - ann.b0[n]);
  }
  for (int n = 0; n < ann.num_neurons2; ++n) {
    for (int m = 0; m < ann.num_neurons1; ++m) {
      x2[n] += ann.w1[n * ann.num_neurons1 + m] * x1[m];
    }
    x2[n] = tanh(x2[n] - ann.b1[n]);
    energy += ann.w2[n] * x2[n];
  }
  energy -= ann.b2[0];
  // energy gradient (compute it component by component)
  for (int d = 0; d < ann.dim; ++d) {
    float y2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f};
    for (int n1 = 0; n1 < ann.num_neurons1; ++n1) {
      float y1 = (1.0f - x1[n1] * x1[n1]) * ann.w0[n1 * ann.dim + d];
      for (int n2 = 0; n2 < ann.num_neurons2; ++n2) {
        y2[n2] += ann.w1[n2 * ann.num_neurons1 + n1] * y1;
      }
    }
    for (int n2 = 0; n2 < ann.num_neurons2; ++n2) {
      energy_derivative[d] += ann.w2[n2] * (y2[n2] * (1.0f - x2[n2] * x2[n2]));
    }
  }
}

static __device__ void find_fc(float rc, float rcinv, float d12, float& fc)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    float y = 1.0f + x * x * (2.0f * x - 3.0f);
    fc = y * y;
  } else {
    fc = 0.0f;
  }
}

static __device__ void find_fc_and_fcp(float rc, float rcinv, float d12, float& fc, float& fcp)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    float y = 1.0f + x * x * (2.0f * x - 3.0f);
    fc = y * y;
    fcp = 12.0f * y * x * (x - 1.0f);
    fcp *= rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static __global__ void find_force_2body(
  const NEP2::Para2B para2b,
  const NEP2::ANN ann2b,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
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
    int neighbor_number = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float pe = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    float virial_xx = 0.0f;
    float virial_xy = 0.0f;
    float virial_xz = 0.0f;
    float virial_yx = 0.0f;
    float virial_yy = 0.0f;
    float virial_yz = 0.0f;
    float virial_zx = 0.0f;
    float virial_zy = 0.0f;
    float virial_zz = 0.0f;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float p2 = 0.0f, f2[1] = {0.0f};
      float q[1] = {d12 * para2b.rcinv};
      apply_ann(ann2b, q, p2, f2);
      f2[0] *= para2b.rcinv;
      float fc, fcp;
      find_fc_and_fcp(para2b.rc, para2b.rcinv, d12, fc, fcp);
      f2[0] = (f2[0] * fc + p2 * fcp) / d12;
      fx += x12 * f2[0];
      fy += y12 * f2[0];
      fz += z12 * f2[0];
      virial_xx -= x12 * x12 * f2[0] * 0.5f;
      virial_xy -= x12 * y12 * f2[0] * 0.5f;
      virial_xz -= x12 * z12 * f2[0] * 0.5f;
      virial_yx -= y12 * x12 * f2[0] * 0.5f;
      virial_yy -= y12 * y12 * f2[0] * 0.5f;
      virial_yz -= y12 * z12 * f2[0] * 0.5f;
      virial_zx -= z12 * x12 * f2[0] * 0.5f;
      virial_zy -= z12 * y12 * f2[0] * 0.5f;
      virial_zz -= z12 * z12 * f2[0] * 0.5f;
      pe += p2 * fc * 0.5f;
    }
    g_fx[n1] = fx;
    g_fy[n1] = fy;
    g_fz[n1] = fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += virial_xx;
    g_virial[n1 + 1 * N] += virial_yy;
    g_virial[n1 + 2 * N] += virial_zz;
    g_virial[n1 + 3 * N] += virial_xy;
    g_virial[n1 + 4 * N] += virial_xz;
    g_virial[n1 + 5 * N] += virial_yz;
    g_virial[n1 + 6 * N] += virial_yx;
    g_virial[n1 + 7 * N] += virial_zx;
    g_virial[n1 + 8 * N] += virial_zy;
    g_pe[n1] = pe;
  }
}

static __global__ void find_neighbor_list_3body(
  const NEP2::Para3B para3b,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN2b,
  const int* g_NL2b,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN3b,
  int* g_NL3b)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_NN2b[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count = 0;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL2b[n1 + N * i1];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12sq = float(x12 * x12 + y12 * y12 + z12 * z12);
      if (d12sq < para3b.rc * para3b.rc) {
        g_NL3b[n1 + N * (count++)] = n2;
      }
    }
    g_NN3b[n1] = count;
  }
}

static __global__ void find_partial_force_3body(
  const NEP2::Para3B para3b,
  const NEP2::ANN ann3b,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_potential,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float pot_energy = 0.0f;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_neighbor_list[index];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(para3b.rc, para3b.rcinv, d12, fc12, fcp12);
      float p12 = 0.0f, f12[3] = {0.0f, 0.0f, 0.0f};
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + N * i2];
        if (n3 == n2) {
          continue;
        }
        double x13double = g_x[n3] - x1;
        double y13double = g_y[n3] - y1;
        double z13double = g_z[n3] - z1;
        apply_mic(box, x13double, y13double, z13double);
        float x13 = float(x13double), y13 = float(y13double), z13 = float(z13double);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float fc13;
        find_fc(para3b.rc, para3b.rcinv, d13, fc13);
        float x23 = x13 - x12;
        float y23 = y13 - y12;
        float z23 = z13 - z12;
        float d23 = sqrt(x23 * x23 + y23 * y23 + z23 * z23);
        float d23inv = 1.0f / d23;
        float q[3] = {d12 + d13, (d12 - d13) * (d12 - d13), d23};
        float p123 = 0.0f, f123[3] = {0.0f};
        apply_ann(ann3b, q, p123, f123);
        p12 += p123 * fc12 * fc13;
        float tmp = p123 * fcp12 * fc13 + (f123[0] + f123[1] * (d12 - d13) * 2.0f) * fc12 * fc13;
        f12[0] += 2.0f * (tmp * x12 * d12inv - f123[2] * fc12 * fc13 * x23 * d23inv);
        f12[1] += 2.0f * (tmp * y12 * d12inv - f123[2] * fc12 * fc13 * y23 * d23inv);
        f12[2] += 2.0f * (tmp * z12 * d12inv - f123[2] * fc12 * fc13 * z23 * d23inv);
      }
      pot_energy += p12;
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
    g_potential[n1] += pot_energy;
  }
}

static __device__ __forceinline__ void
find_fn(const int n_max, const float rcinv, const float d12, const float fc12, float* fn)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f * fc12;
  }
}

static __device__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float* fn,
  float* fnp)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fnp[0] = 0.0f;
  fn[1] = x;
  fnp[1] = 1.0f;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  float u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f;
    fnp[m] *= 2.0f * (d12 * rcinv - 1.0f) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

static __device__ __forceinline__ void
find_poly_cos(const int L_max, const float x, float* poly_cos)
{
  poly_cos[0] = 0.079577471545948f;
  poly_cos[1] = 0.238732414637843f * x;
  float x2 = x * x;
  poly_cos[2] = 0.596831036594608f * x2 - 0.198943678864869f;
  float x3 = x2 * x;
  poly_cos[3] = 1.392605752054084f * x3 - 0.835563451232451f * x;
  float x4 = x3 * x;
  poly_cos[4] = 3.133362942121690f * x4 - 2.685739664675734f * x2 + 0.268573966467573f;
  float x5 = x4 * x;
  poly_cos[5] = 6.893398472667717f * x5 - 7.659331636297464f * x3 + 1.641285350635171f * x;
  float x6 = x5 * x;
  poly_cos[6] = 14.935696690780054f * x6 - 20.366859123790981f * x4 + 6.788953041263660f * x2 -
                0.323283478155412f;
}

static __device__ __forceinline__ void
find_poly_cos_der(const int L_max, const float x, float* poly_cos_der)
{
  poly_cos_der[0] = 0.0f;
  poly_cos_der[1] = 0.238732414637843f;
  poly_cos_der[2] = 1.193662073189215f * x;
  float x2 = x * x;
  poly_cos_der[3] = 4.177817256162252f * x2 - 0.835563451232451f;
  float x3 = x2 * x;
  poly_cos_der[4] = 12.533451768486758f * x3 - 5.371479329351468f * x;
  float x4 = x3 * x;
  poly_cos_der[5] = 34.466992363338584f * x4 - 22.977994908892391f * x2 + 1.641285350635171f;
  float x5 = x4 * x;
  poly_cos_der[6] = 89.614180144680319f * x5 - 81.467436495163923f * x3 + 13.577906082527321f * x;
}

static __global__ void find_partial_force_manybody(
  NEP2::ParaMB paramb,
  NEP2::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      find_fc(paramb.rc, paramb.rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.n_max, paramb.rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max; ++n) {
        q[n * (paramb.L_max + 1) + 0] += fn12[n];
      }
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_NL[n1 + N * i2];
        double x13double = g_x[n3] - x1;
        double y13double = g_y[n3] - y1;
        double z13double = g_z[n3] - z1;
        apply_mic(box, x13double, y13double, z13double);
        float x13 = float(x13double), y13 = float(y13double), z13 = float(z13double);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float fc13;
        find_fc(paramb.rc, paramb.rcinv, d13, fc13);
        float cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
        float poly_cos[MAX_NUM_L];
        find_poly_cos(paramb.L_max, cos123, poly_cos);
        for (int n = 0; n <= paramb.n_max; ++n) {
          for (int l = 1; l <= paramb.L_max; ++l) {
            q[n * (paramb.L_max + 1) + l] += fn12[n] * fc13 * poly_cos[l];
          }
        }
      }
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann(annmb, q, F, Fp);
    g_pe[n1] += F;
    // get partial force
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc, paramb.rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.n_max, paramb.rcinv, d12, fc12, fcp12, fn12, fnp12);
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max; ++n) {
        float tmp = Fp[n * (paramb.L_max + 1) + 0] * fnp12[n] * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
      }
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_NL[n1 + N * i2];
        double x13double = g_x[n3] - x1;
        double y13double = g_y[n3] - y1;
        double z13double = g_z[n3] - z1;
        apply_mic(box, x13double, y13double, z13double);
        float x13 = float(x13double), y13 = float(y13double), z13 = float(z13double);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float d13inv = 1.0f / d13;
        float fc13;
        find_fc(paramb.rc, paramb.rcinv, d13, fc13);
        float cos123 = (r12[0] * x13 + r12[1] * y13 + r12[2] * z13) / (d12 * d13);
        float fn13[MAX_NUM_N];
        find_fn(paramb.n_max, paramb.rcinv, d13, fc13, fn13);
        float poly_cos[MAX_NUM_L];
        find_poly_cos(paramb.L_max, cos123, poly_cos);
        float poly_cos_der[MAX_NUM_L];
        find_poly_cos_der(paramb.L_max, cos123, poly_cos_der);
        float cos_der[3] = {
          x13 * d13inv - r12[0] * d12inv * cos123, y13 * d13inv - r12[1] * d12inv * cos123,
          z13 * d13inv - r12[2] * d12inv * cos123};
        for (int n = 0; n <= paramb.n_max; ++n) {
          float tmp_n_a = (fnp12[n] * fn13[0] + fnp12[0] * fn13[n]) * d12inv;
          float tmp_n_b = (fn12[n] * fn13[0] + fn12[0] * fn13[n]) * d12inv;
          for (int l = 1; l <= paramb.L_max; ++l) {
            float tmp_nl_a = Fp[n * (paramb.L_max + 1) + l] * tmp_n_a * poly_cos[l];
            float tmp_nl_b = Fp[n * (paramb.L_max + 1) + l] * tmp_n_b * poly_cos_der[l];
            for (int d = 0; d < 3; ++d) {
              f12[d] += tmp_nl_a * r12[d] + tmp_nl_b * cos_der[d];
            }
          }
        }
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

void NEP2::compute(
  const int type_shift,
  const Box& box,
  const Neighbor& neighbor,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;
  if (ann2b.num_neurons_per_layer > 0) {
    find_force_2body<<<grid_size, BLOCK_SIZE>>>(
      para2b, ann2b, N, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + N * 2,
      virial_per_atom.data(), potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }
  if (ann3b.num_neurons_per_layer > 0) {
    find_neighbor_list_3body<<<grid_size, BLOCK_SIZE>>>(
      para3b, N, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      nep_data.NN3b.data(), nep_data.NL3b.data());
    CUDA_CHECK_KERNEL
    find_partial_force_3body<<<grid_size, BLOCK_SIZE>>>(
      para3b, ann3b, N, N1, N2, box, nep_data.NN3b.data(), nep_data.NL3b.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      potential_per_atom.data(), nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_properties_many_body(
      box, nep_data.NN3b.data(), nep_data.NL3b.data(), nep_data.f12x.data(), nep_data.f12y.data(),
      nep_data.f12z.data(), position_per_atom, force_per_atom, virial_per_atom);
  }
  if (annmb.num_neurons_per_layer > 0) {
    find_partial_force_manybody<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      potential_per_atom.data(), nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_properties_many_body(
      box, neighbor.NN_local.data(), neighbor.NL_local.data(), nep_data.f12x.data(),
      nep_data.f12y.data(), nep_data.f12z.data(), position_per_atom, force_per_atom,
      virial_per_atom);
    CUDA_CHECK_KERNEL
  }
}
