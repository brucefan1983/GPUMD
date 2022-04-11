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
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep4.cuh"
#include "nep4_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include <string>
#include <vector>

const int NUM_ELEMENTS = 103;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

NEP4::NEP4(FILE* fid, char* input_dir, int num_types, bool enable_zbl, const Neighbor& neighbor)
{
  if (num_types == 1) {
    printf("Use the NEP4 potential with %d atom type.\n", num_types);
  } else {
    printf("Use the NEP4 potential with %d atom types.\n", num_types);
  }

  char name[20];

  for (int n = 0; n < num_types; ++n) {
    int count = fscanf(fid, "%s", name);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
    std::string element(name);
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (element == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %g).\n", n, name, zbl.atomic_numbers[n]);
  }

  paramb.num_types = num_types;

  if (enable_zbl) {
    int count = fscanf(fid, "%s%f%f", name, &zbl.rc_inner, &zbl.rc_outer);
    PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
    zbl.enabled = true;
    printf(
      "    has ZBL with inner cutoff %g A and outer cutoff %g A.\n", zbl.rc_inner, zbl.rc_outer);
  }

  int count = fscanf(fid, "%s%f", name, &paramb.rc_angular);
  PRINT_SCANF_ERROR(count, 2, "reading error for NEP potential.");
  printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  count = fscanf(fid, "%s%d", name, &paramb.n_max_angular);
  PRINT_SCANF_ERROR(count, 2, "reading error for NEP potential.");
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  count = fscanf(fid, "%s%d", name, &paramb.L_max);
  PRINT_SCANF_ERROR(count, 2, "reading error for NEP potential.");
  printf("    l_max = %d.\n", paramb.L_max);

  int num_neurons2;
  count = fscanf(fid, "%s%d%d", name, &ann.num_neurons1, &num_neurons2);
  PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");

  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  ann.dim = (paramb.n_max_angular + 1) * paramb.L_max;

  printf("    ANN = %d-%d-1.\n", ann.dim, ann.num_neurons1);

  ann.num_para = (ann.dim + 2) * ann.num_neurons1 + 1;
  printf("    number of neural network parameters = %d.\n", ann.num_para);
  int num_para_descriptor =
    paramb.num_types * paramb.num_types * (paramb.n_max_angular + 1) * (paramb.basis_size + 1);
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  ann.num_para += num_para_descriptor;
  gnn.num_para = ann.dim * ann.dim;
  printf("    number of GNN parameters = %d.\n", gnn.num_para);
  printf("    total number of parameters = %d\n", ann.num_para + gnn.num_para);

  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  int N_times_max_neighbors = neighbor.NN.size() * neighbor.MN;
  nep_data.f12x.resize(N_times_max_neighbors);
  nep_data.f12y.resize(N_times_max_neighbors);
  nep_data.f12z.resize(N_times_max_neighbors);
  // nep_data.Fp.resize(neighbor.NN.size() * ann.dim);
  // nep_data.sum_fxyz.resize(neighbor.NN.size() * (param.n_max_angular + 1) * NUM_OF_ABC);
  // nep_data.parameters.resize(ann.num_para + gnn.num_para);
  nep_data.dq_dx.resize(N_times_max_neighbors * ann.dim);
  nep_data.dq_dy.resize(N_times_max_neighbors * ann.dim);
  nep_data.dq_dz.resize(N_times_max_neighbors * ann.dim);
  nep_data.q.resize(N2 * ann.dim);
  nep_data.gnn_descriptors.resize(N2 * ann.dim);
  nep_data.gnn_messages.resize(N2 * ann.dim);
  nep_data.gnn_messages_p_x.resize(N2 * ann.dim * neighbor.MN);
  nep_data.gnn_messages_p_y.resize(N2 * ann.dim * neighbor.MN);
  nep_data.gnn_messages_p_z.resize(N2 * ann.dim * neighbor.MN);
  nep_data.dU_dq.resize(N2 * ann.dim);
  nep_data.s.resize(N2 * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  nep_data.parameters.resize(ann.num_para + gnn.num_para);
  update_potential(fid);
}

NEP4::~NEP4(void)
{
  // nothing
}

void NEP4::update_potential(const float* parameters, ANN& ann, GNN& gnn)
{
  // ann
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  ann.b1 = ann.w1 + ann.num_neurons1;
  ann.c = ann.b1 + 1;
  // gnn
  gnn.theta = parameters + ann.num_para;
}

void NEP4::update_potential(FILE* fid)
{
  int total_num_parameters = ann.num_para + gnn.num_para;
  std::vector<float> parameters(total_num_parameters);
  for (int n = 0; n < total_num_parameters; ++n) {
    int count = fscanf(fid, "%f", &parameters[n]);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
  }
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), ann, gnn);
  // for (int d = 0; d < annmb.dim; ++d) {
  //   // Where does this read? Hasn't all the parameters already been read from the file?
  //   // Eric: Remove this?
  //   int count = fscanf(fid, "%f", &paramb.q_scaler[d]);
  //   PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
  // }
}

static __global__ void find_descriptors(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  NEP4::ParaMB paramb,
  NEP4::ANN annmb,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular,
  double* g_q,
  double* g_s)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        apply_mic(box, x12double, y12double, z12double);
        float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        // save distances - is this correct?
        g_x12_angular[n1 + i1 * N] = float(x12);
        g_y12_angular[n1 + i1 * N] = float(y12);
        g_z12_angular[n1 + i1 * N] = float(z12);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        int t2 = g_type[n2];
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size, paramb.rcinv_angular, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size; ++k) {
          int c_index = (n * (paramb.basis_size + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_s[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.L_max; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_q[n1 + ln * N] = q[ln];
      }
    }
  }
}

static __global__ void find_partial_force_angular(
  NEP4::ParaMB paramb,
  NEP4::ANN annmb,
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
  const float* __restrict__ g_sum_fxyz,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < (paramb.n_max_angular + 1) * paramb.L_max; ++d) {
      Fp[d] = g_Fp[d * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float f12[3] = {0.0f};

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size; ++k) {
          int c_index = (n * (paramb.basis_size + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP4::ZBL zbl,
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
    float zi = zbl.atomic_numbers[g_type[n1]];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      float zj = zbl.atomic_numbers[g_type[n2]];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
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

static __global__ void find_dq_dr(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP4::ParaMB para,
  const NEP4::ANN ann,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  const double* __restrict__ g_s,
  double* g_dq_dx,
  double* g_dq_dy,
  double* g_dq_dz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < (para.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      s[d] = g_s[d * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(para.rc_angular, para.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(para.basis_size, para.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= para.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= para.basis_size; ++k) {
          int c_index = (n * (para.basis_size + 1) + k) * para.num_types_sq;
          c_index += t1 * para.num_types + t2;
          gn12 += fn12[k] * ann.c[c_index];
          gnp12 += fnp12[k] * ann.c[c_index];
        }
        dq_dr_double(
          N * (i1 * ann.dim + n) + n1, N * (para.n_max_angular + 1), n, para.n_max_angular + 1, d12,
          r12, gn12, gnp12, s, g_dq_dx, g_dq_dy, g_dq_dz);
      }
    }
  }
}

// Precompute messages q*theta for all descriptors
static __global__ void apply_gnn_compute_messages(
  const int N,
  const NEP4::ANN ann,
  const NEP4::GNN gnn,
  const double* __restrict__ g_q,
  const double* __restrict__ dq_dx,
  const double* __restrict__ dq_dy,
  const double* __restrict__ dq_dz,
  const int* g_NN,
  const int* g_NL,
  double* gnn_messages,
  double* gnn_messages_p_x,
  double* gnn_messages_p_y,
  double* gnn_messages_p_z)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int num_neighbors_of_n1 = g_NN[n1];
    const int F = ann.dim; // dimension of q_out, for now dim_out = dim_in.
    for (int nu = 0; nu < F; nu++) {
      float q_theta_nu = 0.0f;
      for (int gamma = 0; gamma < ann.dim; gamma++) {
        q_theta_nu += g_q[n1 + gamma * N] * gnn.theta[gamma + ann.dim * nu];
      }
      for (int j = 0; j < num_neighbors_of_n1; ++j) {
        // int index_j = n1 + N * j;
        // int n2 = g_NL[index_j];
        float dq_drij_x = 0.0f;
        float dq_drij_y = 0.0f;
        float dq_drij_z = 0.0f;
        for (int gamma = 0; gamma < ann.dim; gamma++) {
          dq_drij_x += dq_dx[N * (j * ann.dim + gamma) + n1] * gnn.theta[gamma + ann.dim * nu];
          dq_drij_y += dq_dy[N * (j * ann.dim + gamma) + n1] * gnn.theta[gamma + ann.dim * nu];
          dq_drij_z += dq_dz[N * (j * ann.dim + gamma) + n1] * gnn.theta[gamma + ann.dim * nu];
        }
        gnn_messages_p_x[N * (nu * MAX_NEIGHBORS + j) + n1] = dq_drij_x;
        gnn_messages_p_y[N * (nu * MAX_NEIGHBORS + j) + n1] = dq_drij_y;
        gnn_messages_p_z[N * (nu * MAX_NEIGHBORS + j) + n1] = dq_drij_z;
      }
      gnn_messages[n1 + nu * N] = q_theta_nu;
    }
  }
}

static __global__ void apply_gnn_message_passing(
  const int N,
  const NEP4::ParaMB para,
  const NEP4::ANN ann,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const double* __restrict__ g_messages,
  const int* g_NN,
  const int* g_NL,
  double* gnn_descriptors,
  double* g_dU_dq)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int num_neighbors_of_n1 = g_NN[n1];
    const int F = ann.dim; // dimension of q_out, for now dim_out = dim_in.
    for (int nu = 0; nu < F; nu++) {
      float q_i_nu = g_messages[n1 + nu * N]; // fc(r_ii) = 1

      // TODO perhaps normalize weights? Compare Kipf, Welling et al. (2016)
      for (int j = 0; j < num_neighbors_of_n1; ++j) {
        int index_j = n1 + N * j;
        int n2 = g_NL[index_j];
        float r12[3] = {g_x12[index_j], g_y12[index_j], g_z12[index_j]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fcij, fcpij;
        find_fc_and_fcp(para.rc_angular, para.rcinv_angular, d12, fcij, fcpij);
        q_i_nu += fcij * g_messages[n2 + nu * N];
      }
      gnn_descriptors[n1 + nu * N] = tanh(q_i_nu);
      g_dU_dq[n1 + nu * N] =
        1 - q_i_nu * q_i_nu; // save sigma'(zi) for when computing message passing forces later
    }
  }
}

static __global__ void apply_ann(
  const int N, const NEP4::ANN ann, const double* __restrict__ g_q, double* g_pe, double* g_dU_dq)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < ann.dim; ++d) {
      q[d] = g_q[n1 + d * N];
    }
    float U = 0.0f, dU_dq[MAX_DIM] = {0.0f};
    apply_ann_one_layer(ann.dim, ann.num_neurons1, ann.w0, ann.b0, ann.w1, ann.b1, q, U, dU_dq);
    g_pe[n1] = U;
    for (int d = 0; d < ann.dim; ++d) {
      g_dU_dq[n1 + d * N] *= dU_dq[d];
    }
  }
}

static __global__ void zero_force(const int N, double* g_fx, double* g_fy, double* g_fz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
  }
}

static __global__ void find_force_gnn(
  const int N,
  const NEP4::ParaMB para,
  const NEP4::ANN ann,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const double* __restrict__ g_messages,
  const double* __restrict__ g_messages_p_x,
  const double* __restrict__ g_messages_p_y,
  const double* __restrict__ g_messages_p_z,
  const double* __restrict__ g_dU_dq,
  const int* g_NN,
  const int* g_NL,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int num_neighbors_of_n1 = g_NN[n1];
    const int F = ann.dim; // dimension of q_out, for now dim_out = dim_in.
    for (int nu = 0; nu < F; nu++) {
      float f_i_x = 0.0f;
      float f_i_y = 0.0f;
      float f_i_z = 0.0f;

      float f_j_x = 0.0f;
      float f_j_y = 0.0f;
      float f_j_z = 0.0f;

      float f_k_x = 0.0f;
      float f_k_y = 0.0f;
      float f_k_z = 0.0f;

      for (int j = 0; j < num_neighbors_of_n1; ++j) {
        int index_j = n1 + N * j;
        int n2 = g_NL[index_j];

        // Fetch index i for atom n1 as a neighbor of n2
        int num_neighbors_of_n2 = g_NN[n2];
        int n2_i = -1;
        for (int n2_j = 0; n2_j < num_neighbors_of_n2; n2_j++) {
          int n2_neighbor = g_NL[n2 + N * n2_j];
          if (n2_neighbor == n1) {
            n2_i = n2_j;
            break;
          }
        }

        float r12[3] = {g_x12[index_j], g_y12[index_j], g_z12[index_j]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fcij, fcpij;
        find_fc_and_fcp(para.rc_angular, para.rcinv_angular, d12, fcij, fcpij);
        f_i_x += g_messages_p_x[N * (j + nu * MAX_NEIGHBORS) + n1];
        f_i_y += g_messages_p_y[N * (j + nu * MAX_NEIGHBORS) + n1];
        f_i_z += g_messages_p_z[N * (j + nu * MAX_NEIGHBORS) + n1];
        f_i_x += fcij * g_messages_p_x[N * (n2_i + nu * MAX_NEIGHBORS) + n2];
        f_i_y += fcij * g_messages_p_y[N * (n2_i + nu * MAX_NEIGHBORS) + n2];
        f_i_z += fcij * g_messages_p_z[N * (n2_i + nu * MAX_NEIGHBORS) + n2];
        f_i_x += fcpij * g_messages[n2 + nu * N] * r12[0] / d12;
        f_i_y += fcpij * g_messages[n2 + nu * N] * r12[1] / d12;
        f_i_z += fcpij * g_messages[n2 + nu * N] * r12[2] / d12;

        f_j_x += g_messages_p_x[N * (n2_i + nu * MAX_NEIGHBORS) + n2];
        f_j_y += g_messages_p_y[N * (n2_i + nu * MAX_NEIGHBORS) + n2];
        f_j_z += g_messages_p_z[N * (n2_i + nu * MAX_NEIGHBORS) + n2];
        f_j_x += fcij * g_messages_p_x[N * (j + nu * MAX_NEIGHBORS) + n1]; // fcij = fcji
        f_j_y += fcij * g_messages_p_y[N * (j + nu * MAX_NEIGHBORS) + n1];
        f_j_z += fcij * g_messages_p_z[N * (j + nu * MAX_NEIGHBORS) + n1];
        f_j_x -= fcpij * g_messages[n2 + nu * N] * r12[0] / d12; // \vec{r}_ij = -\vec{r}_ji
        f_j_y -= fcpij * g_messages[n2 + nu * N] * r12[1] / d12;
        f_j_z -= fcpij * g_messages[n2 + nu * N] * r12[2] / d12;

        for (int k = 0; k < num_neighbors_of_n1; ++k) {
          if (k != j) {
            int index_k = n1 + N * k;
            int n3 = g_NL[index_k];
            // get rjk
            float r23[3] = {g_x12[index_k], g_y12[index_k], g_z12[index_k]};
            float d23 = sqrt(r23[0] * r23[0] + r23[1] * r23[1] + r23[2] * r23[2]);
            float fcjk;
            find_fc(para.rc_angular, para.rcinv_angular, d23, fcjk);
            // Fetch index i for atom n1 as a neighbor of n3
            int num_neighbors_of_n3 = g_NN[n3];
            int n3_i = -1;
            for (int n3_j = 0; n3_j < num_neighbors_of_n3; n3_j++) {
              int n3_neighbor = g_NL[n3 + N * n3_j];
              if (n3_neighbor == n1) {
                n3_i = n3_j;
                break;
              }
            }
            f_k_x += fcjk * g_messages_p_x[N * (n3_i + nu * MAX_NEIGHBORS) + n3] -
                     fcij * g_messages_p_x[N * (k + nu * MAX_NEIGHBORS) + n1];
            f_k_y += fcjk * g_messages_p_y[N * (n3_i + nu * MAX_NEIGHBORS) + n3] -
                     fcij * g_messages_p_y[N * (k + nu * MAX_NEIGHBORS) + n1];
            f_k_z += fcjk * g_messages_p_z[N * (n3_i + nu * MAX_NEIGHBORS) + n3] -
                     fcij * g_messages_p_z[N * (k + nu * MAX_NEIGHBORS) + n1];
          }
        }
        g_fx[n1] -= g_dU_dq[n2 + nu * N] * (f_j_x + f_k_x);
        g_fy[n1] -= g_dU_dq[n2 + nu * N] * (f_j_y + f_k_y);
        g_fz[n1] -= g_dU_dq[n2 + nu * N] * (f_j_z + f_k_z);
      }
      // sum forces over nu
      g_fx[n1] += g_dU_dq[n1 + nu * N] * f_i_x;
      g_fy[n1] += g_dU_dq[n1 + nu * N] * f_i_y;
      g_fz[n1] += g_dU_dq[n1 + nu * N] * f_i_z;
    }
  }
}

// large box fo MD applications
void NEP4::compute_large_box(
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
  const int size_NL = neighbor.NL_local.size();
  GPU_Vector<float> r12(size_NL * 3);

  find_descriptors<<<grid_size, BLOCK_SIZE>>>(
    N, N1, N2, box, paramb, ann, neighbor.NN_local.data(), neighbor.NL_local.data(), type.data(),
    position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
    r12.data(), r12.data() + size_NL, r12.data() + size_NL * 2, nep_data.q.data(),
    nep_data.s.data());
  CUDA_CHECK_KERNEL

  find_dq_dr<<<grid_size, BLOCK_SIZE>>>(
    N, neighbor.NN_local.data(), neighbor.NL_local.data(), paramb, ann, type.data(),
    position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
    nep_data.s.data(), nep_data.dq_dx.data(), nep_data.dq_dy.data(), nep_data.dq_dz.data());
  CUDA_CHECK_KERNEL

  apply_gnn_compute_messages<<<(N - 1) / 64 + 1, 64>>>(
    N, ann, gnn, nep_data.q.data(), nep_data.dq_dx.data(), nep_data.dq_dy.data(),
    nep_data.dq_dz.data(), neighbor.NN_local.data(), neighbor.NL_local.data(),
    nep_data.gnn_messages.data(), nep_data.gnn_messages_p_x.data(),
    nep_data.gnn_messages_p_y.data(), nep_data.gnn_messages_p_z.data());
  CUDA_CHECK_KERNEL

  apply_gnn_message_passing<<<(N - 1) / 64 + 1, 64>>>(
    N, paramb, ann, r12.data(), r12.data() + size_NL, r12.data() + size_NL * 2,
    nep_data.gnn_messages.data(), neighbor.NN_local.data(), neighbor.NL_local.data(),
    nep_data.gnn_descriptors.data(), nep_data.dU_dq.data());
  CUDA_CHECK_KERNEL

  apply_ann<<<grid_size, BLOCK_SIZE>>>(
    N, ann, nep_data.gnn_descriptors.data(), potential_per_atom.data(), nep_data.dU_dq.data());
  CUDA_CHECK_KERNEL

  zero_force<<<grid_size, BLOCK_SIZE>>>(
    N, force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + (N * 2));
  CUDA_CHECK_KERNEL

  find_force_gnn<<<(N - 1) / 64 + 1, 64>>>(
    N, paramb, ann, r12.data(), r12.data() + size_NL, r12.data() + size_NL * 2,
    nep_data.gnn_messages.data(), nep_data.gnn_messages_p_x.data(),
    nep_data.gnn_messages_p_y.data(), nep_data.gnn_messages_p_z.data(), nep_data.dU_dq.data(),
    neighbor.NN.data(), neighbor.NL.data(), force_per_atom.data(), force_per_atom.data() + N,
    force_per_atom.data() + N * 2);
  CUDA_CHECK_KERNEL

  find_properties_many_body(
    box, neighbor.NN_local.data(), neighbor.NL_local.data(), nep_data.f12x.data(),
    nep_data.f12y.data(), nep_data.f12z.data(), position_per_atom, force_per_atom, virial_per_atom);
  CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      N, zbl, N1, N2, box, neighbor.NN_local.data(), neighbor.NL_local.data(), type.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + N * 2,
      virial_per_atom.data(), potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP4::compute_small_box(
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

  const int size_NL = neighbor.NL_local.size();
  GPU_Vector<int> NN_angular(neighbor.NN_local.size());
  GPU_Vector<int> NL_angular(size_NL);
  GPU_Vector<float> r12(size_NL * 3);

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, N, N1, N2, box, ebox, position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, NN_angular.data(), NL_angular.data(), r12.data(),
    r12.data() + size_NL, r12.data() + size_NL * 2);
  CUDA_CHECK_KERNEL

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, ann, N, N1, N2, NN_angular.data(), NL_angular.data(), type.data(), r12.data(),
    r12.data() + size_NL, r12.data() + size_NL * 2, nep_data.q.data(), nep_data.s.data());
  CUDA_CHECK_KERNEL

  find_dq_dr_small_box<<<grid_size, BLOCK_SIZE>>>(
    N, neighbor.NN_local.data(), neighbor.NL_local.data(), paramb, ann, type.data(),
    position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
    nep_data.s.data(), nep_data.dq_dx.data(), nep_data.dq_dy.data(), nep_data.dq_dz.data());
  CUDA_CHECK_KERNEL

  apply_gnn_compute_messages_small_box<<<(N - 1) / 64 + 1, 64>>>(
    N, ann, gnn, nep_data.q.data(), nep_data.dq_dx.data(), nep_data.dq_dy.data(),
    nep_data.dq_dz.data(), neighbor.NN_local.data(), neighbor.NL_local.data(),
    nep_data.gnn_messages.data(), nep_data.gnn_messages_p_x.data(),
    nep_data.gnn_messages_p_y.data(), nep_data.gnn_messages_p_z.data());
  CUDA_CHECK_KERNEL

  apply_gnn_message_passing_small_box<<<(N - 1) / 64 + 1, 64>>>(
    N, paramb, ann, position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, nep_data.gnn_messages.data(), neighbor.NN_local.data(),
    neighbor.NL_local.data(), nep_data.gnn_descriptors.data(), nep_data.dU_dq.data());
  CUDA_CHECK_KERNEL

  apply_ann_small_box<<<grid_size, BLOCK_SIZE>>>(
    N, ann, nep_data.gnn_descriptors.data(), potential_per_atom.data(), nep_data.dU_dq.data());
  CUDA_CHECK_KERNEL

  zero_force_small_box<<<grid_size, BLOCK_SIZE>>>(
    N, force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + (N * 2));
  CUDA_CHECK_KERNEL

  find_force_gnn_small_box<<<(N - 1) / 64 + 1, 64>>>(
    N, paramb, ann, box, ebox, r12.data(), r12.data() + size_NL, r12.data() + size_NL * 2,
    nep_data.gnn_messages.data(), nep_data.gnn_messages_p_x.data(),
    nep_data.gnn_messages_p_y.data(), nep_data.gnn_messages_p_z.data(), nep_data.dU_dq.data(),
    neighbor.NN.data(), neighbor.NL.data(), force_per_atom.data(), force_per_atom.data() + N,
    force_per_atom.data() + N * 2);
  CUDA_CHECK_KERNEL

  // find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
  //   paramb, ann, N, N1, N2, NN_angular.data(), NL_angular.data(), type.data(), r12.data(),
  //   r12.data() + size_NL, r12.data() + size_NL * 2, nep_data.Fp.data(), nep_data.sum_fxyz.data(),
  //   force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + N * 2,
  //   virial_per_atom.data());
  // CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      N, zbl, N1, N2, NN_angular.data(), NL_angular.data(), type.data(), r12.data(),
      r12.data() + size_NL, r12.data() + size_NL * 2, force_per_atom.data(),
      force_per_atom.data() + N, force_per_atom.data() + N * 2, virial_per_atom.data(),
      potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }
}

static void get_expanded_box(const double rc, const Box& box, NEP4::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;
  if (ebox.num_cells[0] * ebox.num_cells[1] * ebox.num_cells[2] > 1) {
    if (box.triclinic) {
      ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
      ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
      ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
      ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
      ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
      ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
      ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
      ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
      ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

      ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
      ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
      ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
      ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
      ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
      ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
      ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
      ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
      ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
      double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                   ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                   ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
      for (int n = 9; n < 18; n++) {
        ebox.h[n] /= det;
      }
    } else {
      ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
      ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
      ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
      ebox.h[3] = ebox.h[0] * 0.5;
      ebox.h[4] = ebox.h[1] * 0.5;
      ebox.h[5] = ebox.h[2] * 0.5;
    }
  }
}

void NEP4::compute(
  const int type_shift,
  const Box& box,
  const Neighbor& neighbor,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  get_expanded_box(paramb.rc_angular, box, ebox);

  if (ebox.num_cells[0] * ebox.num_cells[1] * ebox.num_cells[2] > 1) {
    compute_small_box(
      type_shift, box, neighbor, type, position_per_atom, potential_per_atom, force_per_atom,
      virial_per_atom);
  } else {
    compute_large_box(
      type_shift, box, neighbor, type, position_per_atom, potential_per_atom, force_per_atom,
      virial_per_atom);
  }
}
