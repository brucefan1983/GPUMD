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

#include "nep.cuh"
#include "utilities/error.cuh"
#include <vector>

//#define USE_YLM
#ifdef USE_YLM
const int NUM_OF_ABC = 9; // 1 + 3 + 5 for L_max = 2
#else
const int NUM_OF_ABC = 10; // 1 + 3 + 6 for L_max = 2
#endif

// set by me:
const int MAX_NUM_NEURONS_PER_LAYER = 20; // largest ANN: input-20-20-output
const int MAX_NUM_N = 11;                 // n_max+1 = 10+1
const int MAX_NUM_L = 3;                  // L_max+1 = 2+1
// calculated:
const int MAX_DIM = MAX_NUM_N * MAX_NUM_L;
const int MAX_2B_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + 1) + 1;
const int MAX_3B_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + 3) + 1;
const int MAX_MB_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + MAX_DIM) + 1;
const int MAX_ANN_SIZE = MAX_2B_SIZE + MAX_3B_SIZE + MAX_MB_SIZE;
// constant memory
__constant__ float c_parameters[MAX_ANN_SIZE];

NEP::NEP(FILE* fid, const Neighbor& neighbor)
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
  paramb.delta_r = paramb.rc / paramb.n_max;
  paramb.eta = 0.5f / (paramb.delta_r * paramb.delta_r * 4.0f);
  annmb.dim = (paramb.n_max + 1) * (paramb.L_max + 1);
  annmb.num_para =
    annmb.num_neurons_per_layer > 0
      ? annmb.num_neurons_per_layer * (annmb.num_neurons_per_layer + annmb.dim + 3) + 1
      : 0;

  if (ann3b.num_neurons_per_layer > 0) {
    nep_data.NN3b.resize(neighbor.NN.size());
    nep_data.NL3b.resize(neighbor.NN.size() * neighbor.MN);
  }
  if (annmb.num_neurons_per_layer > 0) {
    nep_data.Fp.resize(neighbor.NN.size() * annmb.dim);
    nep_data.sum_fxyz.resize(neighbor.NN.size() * (paramb.n_max + 1) * NUM_OF_ABC);
  }
  if (ann3b.num_neurons_per_layer > 0 || annmb.num_neurons_per_layer > 0) {
    nep_data.f12x.resize(neighbor.NN.size() * neighbor.MN);
    nep_data.f12y.resize(neighbor.NN.size() * neighbor.MN);
    nep_data.f12z.resize(neighbor.NN.size() * neighbor.MN);
  }
  update_potential(fid);
}

NEP::~NEP(void)
{
  // nothing
}

void NEP::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons_per_layer * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons_per_layer;
  ann.b1 = ann.w1 + ann.num_neurons_per_layer * ann.num_neurons_per_layer;
  ann.w2 = ann.b1 + ann.num_neurons_per_layer;
  ann.b2 = ann.w2 + ann.num_neurons_per_layer;
}

void NEP::update_potential(FILE* fid)
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
apply_ann(const NEP::ANN& ann, float* q, float& energy, float* energy_derivative)
{
  // energy
  float x1[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 1st hidden layer neurons
  float x2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 2nd hidden layer neurons
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < ann.dim; ++d) {
      w0_times_q += ann.w0[n * ann.dim + d] * q[d];
    }
    x1[n] = tanh(w0_times_q - ann.b0[n]);
  }
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    for (int m = 0; m < ann.num_neurons_per_layer; ++m) {
      x2[n] += ann.w1[n * ann.num_neurons_per_layer + m] * x1[m];
    }
    x2[n] = tanh(x2[n] - ann.b1[n]);
  }
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    energy += ann.w2[n] * x2[n];
  }
  energy -= ann.b2[0];
  // energy gradient (compute it component by component)
  for (int d = 0; d < ann.dim; ++d) {
    float y1[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // derivatives of the 1st hidden layer neurons
    float y2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // derivatives of the 2nd hidden layer neurons
    for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
      y1[n] = (1.0f - x1[n] * x1[n]) * ann.w0[n * ann.dim + d];
    }
    for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
      for (int m = 0; m < ann.num_neurons_per_layer; ++m) {
        y2[n] += ann.w1[n * ann.num_neurons_per_layer + m] * y1[m];
      }
      y2[n] *= 1.0f - x2[n] * x2[n];
    }
    for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
      energy_derivative[d] += ann.w2[n] * y2[n];
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
  const NEP::Para2B para2b,
  const NEP::ANN ann2b,
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
  const NEP::Para3B para3b,
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
  const NEP::Para3B para3b,
  const NEP::ANN ann3b,
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

static __device__ void find_fn(const int n, const float rcinv, const float d12, float& fn)
{
  if (n == 0) {
    fn = 1.0f;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5f;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0f) * 0.5f;
  }
}

static __device__ void
find_fn_and_fnp(const int n, const float rcinv, const float d12, float& fn, float& fnp)
{
  if (n == 0) {
    fn = 1.0f;
    fnp = 0.0f;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5f;
    fnp = 2.0f * (d12 * rcinv - 1.0f) * rcinv;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    float u0 = 1.0f;
    float u1 = 2.0f * x;
    float u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0f * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0f) * 0.5f;
    fnp = n * u0 * 2.0f * (d12 * rcinv - 1.0f) * rcinv;
  }
}

#ifdef USE_YLM
__constant__ float YLM_PREFACTOR[NUM_OF_ABC] = {
  0.282094791773878f,  0.488602511902920f, -0.345494149471336f,
  -0.345494149471336f, 0.315391565252520f, -0.772548404046379f,
  -0.772548404046379f, 0.386274202023190f, 0.386274202023190f};
#endif

static __global__ void find_energy_manybody(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
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
  float* g_Fp,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};
    for (int n = 0; n <= paramb.n_max; ++n) {
      float sum_xyz[NUM_OF_ABC] = {0.0f};
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
        float fn;
        find_fn(n, paramb.rcinv, d12, fn);
        fn *= fc12;
        float d12inv = 1.0f / d12;
        x12 *= d12inv;
        y12 *= d12inv;
        z12 *= d12inv;
#ifdef USE_YLM
        sum_xyz[0] += fn;                                                // Y00 without prefactor
        sum_xyz[1] += YLM_PREFACTOR[1] * z12 * fn;                       // Y10
        sum_xyz[2] += YLM_PREFACTOR[2] * x12 * fn;                       // Y11_real
        sum_xyz[3] += YLM_PREFACTOR[3] * y12 * fn;                       // Y11_imag
        sum_xyz[4] += YLM_PREFACTOR[4] * (3.0f * z12 * z12 - 1.0f) * fn; // Y20
        sum_xyz[5] += YLM_PREFACTOR[5] * x12 * z12 * fn;                 // Y21_real
        sum_xyz[6] += YLM_PREFACTOR[6] * y12 * z12 * fn;                 // Y21_imag
        sum_xyz[7] += YLM_PREFACTOR[7] * (x12 * x12 - y12 * y12) * fn;   // Y22_real
        sum_xyz[8] += YLM_PREFACTOR[8] * 2.0f * x12 * y12 * fn;          // Y22_imag
#else
        sum_xyz[0] += fn;
        sum_xyz[1] += x12 * fn;
        sum_xyz[2] += y12 * fn;
        sum_xyz[3] += z12 * fn;
        sum_xyz[4] += x12 * x12 * fn;
        sum_xyz[5] += y12 * y12 * fn;
        sum_xyz[6] += z12 * z12 * fn;
        sum_xyz[7] += x12 * y12 * fn;
        sum_xyz[8] += x12 * z12 * fn;
        sum_xyz[9] += y12 * z12 * fn;
#endif
      }
#ifdef USE_YLM
      q[n * MAX_NUM_L + 0] = sum_xyz[0];
      q[n * MAX_NUM_L + 1] =
        sum_xyz[1] * sum_xyz[1] + 2.0f * sum_xyz[2] * sum_xyz[2] + sum_xyz[3] * sum_xyz[3];
      q[n * MAX_NUM_L + 2] =
        sum_xyz[4] * sum_xyz[4] + 2.0f * (sum_xyz[5] * sum_xyz[5] + sum_xyz[6] * sum_xyz[6] +
                                          sum_xyz[7] * sum_xyz[7] + sum_xyz[8] * sum_xyz[8]);
#else
      q[n * MAX_NUM_L + 0] = sum_xyz[0];
      q[n * MAX_NUM_L + 1] =
        sum_xyz[1] * sum_xyz[1] + sum_xyz[2] * sum_xyz[2] + sum_xyz[3] * sum_xyz[3];
      q[n * MAX_NUM_L + 2] =
        sum_xyz[7] * sum_xyz[7] + sum_xyz[8] * sum_xyz[8] + sum_xyz[9] * sum_xyz[9];
      q[n * MAX_NUM_L + 2] *= 2.0f;
      q[n * MAX_NUM_L + 2] +=
        sum_xyz[4] * sum_xyz[4] + sum_xyz[5] * sum_xyz[5] + sum_xyz[6] * sum_xyz[6];
#endif
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = sum_xyz[abc];
      }
    }
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann(annmb, q, F, Fp);
    g_pe[n1] += F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d];
    }
  }
}

static __global__ void find_partial_force_manybody(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
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
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {float(x12), float(y12), float(z12)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc, paramb.rcinv, d12, fc12, fcp12);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max; ++n) {
        float fn;
        float fnp;
        find_fn_and_fnp(n, paramb.rcinv, d12, fn, fnp);
        // l=0
        float fn0 = fn * fc12;
        float fn0p = fnp * fc12 + fn * fcp12;
        float Fp0 = g_Fp[(n * MAX_NUM_L + 0) * N + n1];
        float tmp = Fp0 * 0.5f * fn0p * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
#ifdef USE_YLM
        float sum_xyz[9] = {
          /*not used*/ 0.0f,
          g_sum_fxyz[(n * NUM_OF_ABC + 1) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 2) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 3) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 4) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 5) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 6) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 7) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 8) * N + n1]};
        // l=1
        float fn1 = fn0 * d12inv;
        float fn1p = fn0p * d12inv - fn0 * d12inv * d12inv;
        float Fp1 = g_Fp[(n * MAX_NUM_L + 1) * N + n1];
        float tmp =
          Fp1 * fn1p * d12inv *
          (sum_xyz[1] * YLM_PREFECTOR[1] * r12[2] + 2.0f * sum_xyz[2] * YLM_PREFECTOR[2] * r12[0] +
           2.0f * sum_xyz[3] * YLM_PREFECTOR[3] * r12[1]);
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
        tmp = Fp1 * fn1;
        f12[0] += tmp * 2.0f * sum_xyz[2] * YLM_PREFECTOR[2];
        f12[1] += tmp * 2.0f * sum_xyz[3] * YLM_PREFECTOR[3];
        f12[2] += tmp * sum_xyz[1] * YLM_PREFECTOR[1];
        // l=2
        float fn2 = fn1 * d12inv;
        float fn2p = fn1p * d12inv - fn1 * d12inv * d12inv;
        float Fp2 = g_Fp[(n * MAX_NUM_L + 2) * N + n1];
        tmp = Fp2 * fn2p * d12inv *
              (sum_xyz[4] * YLM_PREFECTOR[4] * (3.0f * r12[2] * r12[2] - d12 * d12) +
               2.0f * sum_xyz[5] * YLM_PREFECTOR[5] * r12[0] * r12[2] +
               2.0f * sum_xyz[6] * YLM_PREFECTOR[6] * r12[1] * r12[2] +
               2.0f * sum_xyz[7] * YLM_PREFECTOR[7] * (r12[0] * r12[0] - r12[1] * r12[1]) +
               2.0f * sum_xyz[8] * YLM_PREFECTOR[8] * 2.0f * r12[0] * r12[1]);
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
        tmp = Fp2 * fn2;
        f12[0] += tmp * (-2.0f * sum_xyz[4] * YLM_PREFECTOR[4] * r12[0] +
                         2.0f * sum_xyz[5] * YLM_PREFECTOR[5] * r12[2] +
                         4.0f * sum_xyz[7] * YLM_PREFECTOR[7] * r12[0] +
                         4.0f * sum_xyz[8] * YLM_PREFECTOR[8] * r12[1]);
        f12[1] += tmp * (-2.0f * sum_xyz[4] * YLM_PREFECTOR[4] * r12[1] +
                         2.0f * sum_xyz[6] * YLM_PREFECTOR[6] * r12[2] -
                         4.0f * sum_xyz[7] * YLM_PREFECTOR[7] * r12[1] +
                         4.0f * sum_xyz[8] * YLM_PREFECTOR[8] * r12[0]);
        f12[2] += tmp * (4.0f * sum_xyz[4] * YLM_PREFECTOR[4] * r12[2] +
                         2.0f * sum_xyz[5] * YLM_PREFECTOR[5] * r12[0] +
                         2.0f * sum_xyz[6] * YLM_PREFECTOR[6] * r12[1]);
#else
        // l=1
        float fn1 = fn0 * d12inv;
        float fn1p = fn0p * d12inv - fn0 * d12inv * d12inv;
        float Fp1 = g_Fp[(n * MAX_NUM_L + 1) * N + n1];
        float sum_f1[3] = {
          g_sum_fxyz[(n * NUM_OF_ABC + 1) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 2) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 3) * N + n1]};
        float tmp1 =
          Fp1 * fn1p * (sum_f1[0] * r12[0] + sum_f1[1] * r12[1] + sum_f1[2] * r12[2]) * d12inv;
        float tmp2 = Fp1 * fn1;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp1 * r12[d] + tmp2 * sum_f1[d];
        }
        // l=2
        float fn2 = fn1 * d12inv;
        float fn2p = fn1p * d12inv - fn1 * d12inv * d12inv;
        float Fp2 = g_Fp[(n * MAX_NUM_L + 2) * N + n1];
        float sum_f2[6] = {
          g_sum_fxyz[(n * NUM_OF_ABC + 4) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 5) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 6) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 7) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 8) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 9) * N + n1]};
        tmp1 = Fp2 * fn2p *
               (sum_f2[0] * r12[0] * r12[0] + sum_f2[1] * r12[1] * r12[1] +
                sum_f2[2] * r12[2] * r12[2] + 2.0f * sum_f2[3] * r12[0] * r12[1] +
                2.0f * sum_f2[4] * r12[0] * r12[2] + 2.0f * sum_f2[5] * r12[1] * r12[2]) *
               d12inv;
        tmp2 = 2.0f * Fp2 * fn2;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp1 * r12[d] + tmp2 * sum_f2[d] * r12[d];
        }
        f12[0] += tmp2 * (sum_f2[3] * r12[1] + sum_f2[4] * r12[2]);
        f12[1] += tmp2 * (sum_f2[3] * r12[0] + sum_f2[5] * r12[2]);
        f12[2] += tmp2 * (sum_f2[4] * r12[0] + sum_f2[5] * r12[1]);
#endif
      }
      g_f12x[index] = f12[0] * 2.0f;
      g_f12y[index] = f12[1] * 2.0f;
      g_f12z[index] = f12[2] * 2.0f;
    }
  }
}

void NEP::compute(
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
    find_energy_manybody<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      potential_per_atom.data(), nep_data.Fp.data(), nep_data.sum_fxyz.data());
    CUDA_CHECK_KERNEL
    find_partial_force_manybody<<<grid_size, BLOCK_SIZE>>>(
      paramb, annmb, N, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      nep_data.Fp.data(), nep_data.sum_fxyz.data(), nep_data.f12x.data(), nep_data.f12y.data(),
      nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_properties_many_body(
      box, neighbor.NN_local.data(), neighbor.NL_local.data(), nep_data.f12x.data(),
      nep_data.f12y.data(), nep_data.f12z.data(), position_per_atom, force_per_atom,
      virial_per_atom);
  }
}
