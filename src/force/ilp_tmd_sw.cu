/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
The class dealing with the interlayer potential(ILP) and SW.
TODO:
------------------------------------------------------------------------------*/

#include "ilp_tmd_sw.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

#define BLOCK_SIZE_FORCE 128

// there are most 6 intra-layer neighbors for TMD
#define NNEI 6


ILP_TMD_SW::ILP_TMD_SW(FILE* fid_ilp, FILE* fid_sw, int num_types, int num_atoms)
{
  // read ILP TMD potential parameter
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP_TMD_SW)) {
    PRINT_INPUT_ERROR("Incorrect type number of ILP_TMD_SW parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid_ilp, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for ILP_TMD_SW potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  // read parameters
  float beta, alpha, delta, epsilon, C, d, sR;
  float reff, C6, S, rcut_ilp, rcut_global;
  rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid_ilp, "%f%f%f%f%f%f%f%f%f%f%f%f", \
      &beta, &alpha, &delta, &epsilon, &C, &d, &sR, &reff, &C6, &S, \
      &rcut_ilp, &rcut_global);
      PRINT_SCANF_ERROR(count, 12, "Reading error for ILP_TMD_SW potential.");

      ilp_para.C[n][m] = C;
      ilp_para.C_6[n][m] = C6;
      ilp_para.d[n][m] = d;
      ilp_para.d_Seff[n][m] = d / sR / reff;
      ilp_para.epsilon[n][m] = epsilon;
      ilp_para.z0[n][m] = beta;
      ilp_para.lambda[n][m] = alpha / beta;
      ilp_para.delta2inv[n][m] = 1.0 / (delta * delta);
      ilp_para.S[n][m] = S;
      ilp_para.rcutsq_ilp[n][m] = rcut_ilp * rcut_ilp;
      ilp_para.rcut_global[n][m] = rcut_global;
      float meV = 1e-3 * S;
      ilp_para.C[n][m] *= meV;
      ilp_para.C_6[n][m] *= meV;
      ilp_para.epsilon[n][m] *= meV;

      if (rc < rcut_global)
        rc = rcut_global;
    }
  }

  // read SW potential parameter
  if (num_types == 1) {
    initialize_sw_1985_1(fid_sw);
  }
  if (num_types == 2) {
    initialize_sw_1985_2(fid_sw);
  }
  if (num_types == 3) {
    initialize_sw_1985_3(fid_sw);
  }

  // initialize neighbor lists and some temp vectors
  int max_neighbor_number = min(num_atoms, CUDA_MAX_NL_TMD);
  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * max_neighbor_number);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);

  // init ilp neighbor list
  ilp_data.ilp_NN.resize(num_atoms);
  ilp_data.ilp_NL.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.reduce_NL.resize(num_atoms * max_neighbor_number);
  ilp_data.big_ilp_NN.resize(num_atoms);
  ilp_data.big_ilp_NL.resize(num_atoms * MAX_BIG_ILP_NEIGHBOR_TMD);

  ilp_data.f12x.resize(num_atoms * max_neighbor_number);
  ilp_data.f12y.resize(num_atoms * max_neighbor_number);
  ilp_data.f12z.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.f12y_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.f12z_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);

  // intialize sw neighbor list
  sw2_data.NN.resize(num_atoms);
  sw2_data.NL.resize(num_atoms * 1024); // the largest supported by CUDA
  sw2_data.cell_count.resize(num_atoms);
  sw2_data.cell_count_sum.resize(num_atoms);
  sw2_data.cell_contents.resize(num_atoms);

  // memory for the partial forces dU_i/dr_ij
  const int num_of_neighbors = MAX_SW_NEIGHBOR_NUM * num_atoms;
  sw2_data.f12x.resize(num_of_neighbors);
  sw2_data.f12y.resize(num_of_neighbors);
  sw2_data.f12z.resize(num_of_neighbors);

  // init constant cutoff coeff
  float h_tap_coeff[8] = \
    {1.0f, 0.0f, 0.0f, 0.0f, -35.0f, 84.0f, -70.0f, 20.0f};
  CHECK(gpuMemcpyToSymbol(Tap_coeff_tmd, h_tap_coeff, 8 * sizeof(float)));

  // set ilp_flag to 1
  ilp_flag = 1;
}

ILP_TMD_SW::~ILP_TMD_SW(void)
{
  // nothing
}

void ILP_TMD_SW::initialize_sw_1985_1(FILE* fid)
{
  printf("Use single-element Stillinger-Weber potential.\n");
  int count;
  double epsilon, lambda, A, B, a, gamma, sigma, cos0;
  count =
    fscanf(fid, "%lf%lf%lf%lf%lf%lf%lf%lf", &epsilon, &lambda, &A, &B, &a, &gamma, &sigma, &cos0);
  PRINT_SCANF_ERROR(count, 8, "Reading error for SW potential.");

  sw2_para.A[0][0] = epsilon * A;
  sw2_para.B[0][0] = B;
  sw2_para.a[0][0] = a;
  sw2_para.sigma[0][0] = sigma;
  sw2_para.gamma[0][0] = gamma;
  sw2_para.rc[0][0] = sigma * a;
  rc_sw = sw2_para.rc[0][0];
  sw2_para.lambda[0][0][0] = epsilon * lambda;
  sw2_para.cos0[0][0][0] = cos0;
}

void ILP_TMD_SW::initialize_sw_1985_2(FILE* fid)
{
  printf("Use two-element Stillinger-Weber potential.\n");
  int count;

  // 2-body parameters and the force cutoff
  double A[3], B[3], a[3], sigma[3], gamma[3];
  rc_sw = 0.0;
  for (int n = 0; n < 3; n++) {
    count = fscanf(fid, "%lf%lf%lf%lf%lf", &A[n], &B[n], &a[n], &sigma[n], &gamma[n]);
    PRINT_SCANF_ERROR(count, 5, "Reading error for SW potential.");
  }
  for (int n1 = 0; n1 < 2; n1++)
    for (int n2 = 0; n2 < 2; n2++) {
      sw2_para.A[n1][n2] = A[n1 + n2];
      sw2_para.B[n1][n2] = B[n1 + n2];
      sw2_para.a[n1][n2] = a[n1 + n2];
      sw2_para.sigma[n1][n2] = sigma[n1 + n2];
      sw2_para.gamma[n1][n2] = gamma[n1 + n2];
      sw2_para.rc[n1][n2] = sigma[n1 + n2] * a[n1 + n2];
      if (rc_sw < sw2_para.rc[n1][n2])
        rc_sw = sw2_para.rc[n1][n2];
    }

  // 3-body parameters
  double lambda, cos0;
  for (int n1 = 0; n1 < 2; n1++)
    for (int n2 = 0; n2 < 2; n2++)
      for (int n3 = 0; n3 < 2; n3++) {
        count = fscanf(fid, "%lf%lf", &lambda, &cos0);
        PRINT_SCANF_ERROR(count, 2, "Reading error for SW potential.");
        sw2_para.lambda[n1][n2][n3] = lambda;
        sw2_para.cos0[n1][n2][n3] = cos0;
      }
}

void ILP_TMD_SW::initialize_sw_1985_3(FILE* fid)
{
  printf("Use three-element Stillinger-Weber potential.\n");
  int count;

  // 2-body parameters and the force cutoff
  double A, B, a, sigma, gamma;
  rc_sw = 0.0;
  for (int n1 = 0; n1 < 3; n1++)
    for (int n2 = 0; n2 < 3; n2++) {
      count = fscanf(fid, "%lf%lf%lf%lf%lf", &A, &B, &a, &sigma, &gamma);
      PRINT_SCANF_ERROR(count, 5, "Reading error for SW potential.");
      sw2_para.A[n1][n2] = A;
      sw2_para.B[n1][n2] = B;
      sw2_para.a[n1][n2] = a;
      sw2_para.sigma[n1][n2] = sigma;
      sw2_para.gamma[n1][n2] = gamma;
      sw2_para.rc[n1][n2] = sigma * a;
      if (rc_sw < sw2_para.rc[n1][n2])
        rc_sw = sw2_para.rc[n1][n2];
    }

  // 3-body parameters
  double lambda, cos0;
  for (int n1 = 0; n1 < 3; n1++) {
    for (int n2 = 0; n2 < 3; n2++) {
      for (int n3 = 0; n3 < 3; n3++) {
        count = fscanf(fid, "%lf%lf", &lambda, &cos0);
        PRINT_SCANF_ERROR(count, 2, "Reading error for SW potential.");
        sw2_para.lambda[n1][n2][n3] = lambda;
        sw2_para.cos0[n1][n2][n3] = cos0;
      }
    }
  }
}

static __device__ __forceinline__ float calc_Tap(const float r_ij, const float Rcutinv)
{
  float Tap, r;

  r = r_ij * Rcutinv;
  if (r >= 1.0f) {
    Tap = 0.0f;
  } else {
    Tap = Tap_coeff_tmd[7];
    for (int i = 6; i >= 0; --i) {
      Tap = Tap * r + Tap_coeff_tmd[i];
    }
  }

  return Tap;
}

// calculate the derivatives of long-range cutoff term
static __device__ __forceinline__ float calc_dTap(const float r_ij, const float Rcut, const float Rcutinv)
{
  float dTap, r;
  
  r = r_ij * Rcutinv;
  if (r >= Rcut) {
    dTap = 0.0f;
  } else {
    dTap = 7.0f * Tap_coeff_tmd[7];
    for (int i = 6; i > 0; --i) {
      dTap = dTap * r + i * Tap_coeff_tmd[i];
    }
    dTap *= Rcutinv;
  }

  return dTap;
}

// create ILP neighbor list from main neighbor list to calculate normals
static __global__ void ILP_neighbor(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  const int *g_type,
  ILP_TMD_Para ilp_para,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int *ilp_neighbor_number,
  int *ilp_neighbor_list,
  const int *group_label)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
    // TMD
    int neighptr[10], check[10], neighsort[10];
    for (int ll = 0; ll < 10; ++ll) {
      neighptr[ll] = -1;
      neighsort[ll] = -1;
      check[ll] = -1;
    }

    int count = 0;
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      int type2 = g_type[n2];

      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12sq = x12 * x12 + y12 * y12 + z12 * z12;
      double rcutsq = ilp_para.rcutsq_ilp[type1][type2];


      if (group_label[n1] == group_label[n2] && d12sq < rcutsq && type1 == type2 && d12sq != 0) {
        // ilp_neighbor_list[count++ * number_of_particles + n1] = n2;
        neighptr[count++] = n2;
      }
    }

    // TMD
    for (int ll = 0; ll < count; ++ll) {
      neighsort[ll] = neighptr[ll];
      check[ll] = neighptr[ll];
    }

    // TMD
    if (count == NNEI) {
      neighsort[0] = neighptr[0];
      check[0] = -1;
    } else if (count < NNEI && count > 0) {
      for (int jj = 0; jj < count; ++jj) {
        int j = neighptr[jj];
        int jtype = g_type[j];
        int count_temp = 0;
        for (int ll = 0; ll < count; ++ll) {
          int l = neighptr[ll];
          int ltype = g_type[l];
          if (l == j) continue;
          double deljx = g_x[l] - g_x[j];
          double deljy = g_y[l] - g_y[j];
          double deljz = g_z[l] - g_z[j];
          apply_mic(box, deljx, deljy, deljz);
          double rsqlj = deljx * deljx + deljy * deljy + deljz * deljz;
          if (rsqlj != 0 && rsqlj < ilp_para.rcutsq_ilp[ltype][jtype]) {
            ++count_temp;
          }

        }
        if (count_temp == 1) {
          neighsort[0] = neighptr[jj];
          check[jj] = -1;
          break;
        }
      }
    } else if (count > NNEI) {
      printf("ERROR in ILP NEIGHBOR LIST\n");
      printf("\n===== ILP neighbor number[%d] is greater than 6 =====\n", count);
      return;
    }

    // TMD
    // sort the order of neighbors of atom n1
    for (int jj = 0; jj < count; ++jj) {
      int j = neighsort[jj];
      int jtype = g_type[j];
      int ll = 0;
      while (ll < count) {
        int l = neighptr[ll];
        if (check[ll] == -1) {
          ++ll;
          continue;
        }
        int ltype = g_type[l];
        double deljx = g_x[l] - g_x[j];
        double deljy = g_y[l] - g_y[j];
        double deljz = g_z[l] - g_z[j];
        apply_mic(box, deljx, deljy, deljz);
        double rsqlj = deljx * deljx + deljy * deljy + deljz * deljz;
        
        if (abs(rsqlj) >= 1e-6 && rsqlj < ilp_para.rcutsq_ilp[ltype][jtype]) {
          neighsort[jj + 1] = l;
          check[ll] = -1;
          break;
        }
        ++ll;
      }
    }
    ilp_neighbor_number[n1] = count;
    for (int jj = 0; jj < count; ++jj) {
      ilp_neighbor_list[jj * number_of_particles + n1] = neighsort[jj];
    }
  }
}

// modulo func to change atom index
static __device__ __forceinline__ int modulo(int k, int range)
{
  return (k + range) % range;
}

// calculate the normals and its derivatives
static __device__ void calc_normal(
  float (&vect)[NNEI][3],
  int cont,
  float (&normal)[3],
  float (&dnormdri)[3][3],
  float (&dnormal)[3][NNEI][3])
{
  int id, ip, m;
  float  dni[3];
  float  dnn[3][3], dpvdri[3][3];
  float Nave[3], pvet[NNEI][3], dpvet1[NNEI][3][3], dpvet2[NNEI][3][3], dNave[3][NNEI][3];

  float nninv;

  // initialize the arrays
  for (id = 0; id < 3; id++) {
    dni[id] = 0.0f;

    Nave[id] = 0.0f;
    for (ip = 0; ip < 3; ip++) {
      dpvdri[ip][id] = 0.0f;
      for (m = 0; m < NNEI; m++) {
        dnn[m][id] = 0.0f;
        pvet[m][id] = 0.0f;
        dpvet1[m][ip][id] = 0.0f;
        dpvet2[m][ip][id] = 0.0f;
        dNave[id][m][ip] = 0.0f;
      }
    }
  }

  if (cont <= 1) {
    normal[0] = 0.0f;
    normal[1] = 0.0f;
    normal[2] = 1.0f;
    for (id = 0; id < 3; ++id) {
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[id][ip] = 0.0f;
        for (m = 0; m < NNEI; ++m) {
          dnormal[id][m][ip] = 0.0f;
        }
      }
    }
  } else if (cont > 1 && cont < NNEI) {
    for (int k = 0; k < cont - 1; ++k) {
      for (ip = 0; ip < 3; ++ip) {
        pvet[k][ip] = vect[k][modulo(ip + 1, 3)] * vect[k + 1][modulo(ip + 2, 3)] -
                vect[k][modulo(ip + 2, 3)] * vect[k + 1][modulo(ip + 1, 3)];
      }
      // dpvet1[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l
      // derivatives respect to atom l
      // dNik,x/drl
      dpvet1[k][0][0] = 0.0f;
      dpvet1[k][0][1] = vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][0][2] = -vect[modulo(k + 1, NNEI)][1];
      // dNik,y/drl
      dpvet1[k][1][0] = -vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][1][1] = 0.0f;
      dpvet1[k][1][2] = vect[modulo(k + 1, NNEI)][0];
      // dNik,z/drl
      dpvet1[k][2][0] = vect[modulo(k + 1, NNEI)][1];
      dpvet1[k][2][1] = -vect[modulo(k + 1, NNEI)][0];
      dpvet1[k][2][2] = 0.0f;

      // dpvet2[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l+1
      // derivatives respect to atom l+1
      // dNik,x/drl+1
      dpvet2[k][0][0] = 0.0f;
      dpvet2[k][0][1] = -vect[modulo(k, NNEI)][2];
      dpvet2[k][0][2] = vect[modulo(k, NNEI)][1];
      // dNik,y/drl+1
      dpvet2[k][1][0] = vect[modulo(k, NNEI)][2];
      dpvet2[k][1][1] = 0.0f;
      dpvet2[k][1][2] = -vect[modulo(k, NNEI)][0];
      // dNik,z/drl+1
      dpvet2[k][2][0] = -vect[modulo(k, NNEI)][1];
      dpvet2[k][2][1] = vect[modulo(k, NNEI)][0];
      dpvet2[k][2][2] = 0.0f;
    }

    // average the normal vectors by using the NNEI neighboring planes
    for (ip = 0; ip < 3; ip++) {
      Nave[ip] = 0.0f;
      for (int k = 0; k < cont - 1; k++) {
        Nave[ip] += pvet[k][ip];
      }
      Nave[ip] /= (cont - 1);
    }
    nninv = rnorm3df(Nave[0], Nave[1], Nave[2]);
    
    // the unit normal vector
    normal[0] = Nave[0] * nninv;
    normal[1] = Nave[1] * nninv;
    normal[2] = Nave[2] * nninv;

    // derivatives of non-normalized normal vector, dNave:3xcontx3 array
    // dNave[id][m][ip]: the derivatve of the id component of Nave respect to the ip component of atom m
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        for (m = 0; m < cont; m++) {
          if (m == 0) {
            dNave[id][m][ip] = dpvet1[m][id][ip] / (cont - 1);
          } else if (m == cont - 1) {
            dNave[id][m][ip] = dpvet2[m - 1][id][ip] / (cont - 1);
          } else {    // sum of the derivatives of the mth and (m-1)th normal vector respect to the atom m
            dNave[id][m][ip] = (dpvet1[m][id][ip] + dpvet2[m - 1][id][ip]) / (cont - 1);
          }
        }
      }
    }
    // derivatives of nn, dnn:contx3 vector
    // dnn[m][id]: the derivative of nn respect to r[m][id], m=0,...NNEI-1; id=0,1,2
    // r[m][id]: the id's component of atom m
    for (m = 0; m < cont; m++) {
      for (id = 0; id < 3; id++) {
        dnn[m][id] = (Nave[0] * dNave[0][m][id] + Nave[1] * dNave[1][m][id] +
                      Nave[2] * dNave[2][m][id]) * nninv;
      }
    }
    // dnormal[i][id][m][ip]: the derivative of normal[i][id] respect to r[m][ip], id,ip=0,1,2.
    // for atom m, which is a neighbor atom of atom i, m = 0,...,NNEI-1
    for (m = 0; m < cont; m++) {
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormal[id][m][ip] = dNave[id][m][ip] * nninv - Nave[id] * dnn[m][ip] * nninv * nninv;
        }
      }
    }
    // Calculte dNave/dri, defined as dpvdri
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        dpvdri[id][ip] = 0.0;
        for (int k = 0; k < cont; k++) {
          dpvdri[id][ip] -= dNave[id][k][ip];
        }
      }
    }

    // derivatives of nn, dnn:3x1 vector
    dni[0] = (Nave[0] * dpvdri[0][0] + Nave[1] * dpvdri[1][0] + Nave[2] * dpvdri[2][0]) * nninv;
    dni[1] = (Nave[0] * dpvdri[0][1] + Nave[1] * dpvdri[1][1] + Nave[2] * dpvdri[2][1]) * nninv;
    dni[2] = (Nave[0] * dpvdri[0][2] + Nave[1] * dpvdri[1][2] + Nave[2] * dpvdri[2][2]) * nninv;
    // derivatives of unit vector ni respect to ri, the result is 3x3 matrix
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        dnormdri[id][ip] = dpvdri[id][ip] * nninv - Nave[id] * dni[ip] * nninv * nninv;
      }
    }
  } else if (cont == NNEI) {
    // derivatives of Ni[l] respect to the NNEI neighbors
    for (int k = 0; k < NNEI; ++k) {
      for (ip = 0; ip < 3; ++ip) {
        pvet[k][ip] = vect[modulo(k, NNEI)][modulo(ip + 1, 3)] *
                vect[modulo(k + 1, NNEI)][modulo(ip + 2, 3)] -
            vect[modulo(k, NNEI)][modulo(ip + 2, 3)] *
                vect[modulo(k + 1, NNEI)][modulo(ip + 1, 3)];
      }
      // dpvet1[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l
      // derivatives respect to atom l
      // dNik,x/drl
      dpvet1[k][0][0] = 0.0f;
      dpvet1[k][0][1] = vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][0][2] = -vect[modulo(k + 1, NNEI)][1];
      // dNik,y/drl
      dpvet1[k][1][0] = -vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][1][1] = 0.0f;
      dpvet1[k][1][2] = vect[modulo(k + 1, NNEI)][0];
      // dNik,z/drl
      dpvet1[k][2][0] = vect[modulo(k + 1, NNEI)][1];
      dpvet1[k][2][1] = -vect[modulo(k + 1, NNEI)][0];
      dpvet1[k][2][2] = 0.0f;

      // dpvet2[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l+1
      // derivatives respect to atom l+1
      // dNik,x/drl+1
      dpvet2[k][0][0] = 0.0f;
      dpvet2[k][0][1] = -vect[modulo(k, NNEI)][2];
      dpvet2[k][0][2] = vect[modulo(k, NNEI)][1];
      // dNik,y/drl+1
      dpvet2[k][1][0] = vect[modulo(k, NNEI)][2];
      dpvet2[k][1][1] = 0.0f;
      dpvet2[k][1][2] = -vect[modulo(k, NNEI)][0];
      // dNik,z/drl+1
      dpvet2[k][2][0] = -vect[modulo(k, NNEI)][1];
      dpvet2[k][2][1] = vect[modulo(k, NNEI)][0];
      dpvet2[k][2][2] = 0.0f;
    }

    // average the normal vectors by using the NNEI neighboring planes
    for (ip = 0; ip < 3; ++ip) {
      Nave[ip] = 0.0f;
      for (int k = 0; k < NNEI; ++k) {
        Nave[ip] += pvet[k][ip];
      }
      Nave[ip] /= NNEI;
    }
    // the magnitude of the normal vector
    // nn2 = Nave[0] * Nave[0] + Nave[1] * Nave[1] + Nave[2] * Nave[2];
    nninv = rnorm3df(Nave[0], Nave[1], Nave[2]);
    // the unit normal vector
    normal[0] = Nave[0] * nninv;
    normal[1] = Nave[1] * nninv;
    normal[2] = Nave[2] * nninv;

    // for the central atoms, dnormdri is always zero
    for (id = 0; id < 3; ++id) {
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[id][ip] = 0.0f;
      }
    }

    // derivatives of non-normalized normal vector, dNave:3xNNEIx3 array
    // dNave[id][m][ip]: the derivatve of the id component of Nave respect to the ip component of atom m
    for (id = 0; id < 3; ++id) {
      for (ip = 0; ip < 3; ++ip) {
        for (
            m = 0; m < NNEI;
            ++m) {    // sum of the derivatives of the mth and (m-1)th normal vector respect to the atom m
          dNave[id][m][ip] =
              (dpvet1[modulo(m, NNEI)][id][ip] + dpvet2[modulo(m - 1, NNEI)][id][ip]) / NNEI;
        }
      }
    }
    // derivatives of nn, dnn:NNEIx3 vector
    // dnn[m][id]: the derivative of nn respect to r[m][id], m=0,...NNEI-1; id=0,1,2
    // r[m][id]: the id's component of atom m
    for (m = 0; m < NNEI; ++m) {
      for (id = 0; id < 3; ++id) {
        dnn[m][id] =
            (Nave[0] * dNave[0][m][id] + Nave[1] * dNave[1][m][id] + Nave[2] * dNave[2][m][id]) *
            nninv;
      }
    }
    // dnormal[i][id][m][ip]: the derivative of normal[i][id] respect to r[m][ip], id,ip=0,1,2.
    // for atom m, which is a neighbor atom of atom i, m = 0,...,NNEI-1
    for (m = 0; m < NNEI; ++m) {
      for (id = 0; id < 3; ++id) {
        for (ip = 0; ip < 3; ++ip) {
          dnormal[id][m][ip] = dNave[id][m][ip] * nninv - Nave[id] * dnn[m][ip] * nninv * nninv;
        }
      }
    }
  } else {
    printf("\n===== ILP neighbor number[%d] is greater than 6 =====\n", cont);
    return;
  }
}

// calculate the van der Waals force and energy
static __device__ void calc_vdW(
  float r,
  float rinv,
  float rsq,
  float d,
  float d_Seff,
  float C_6,
  float Tap,
  float dTap,
  float &p2_vdW,
  float &f2_vdW)
{
  float r2inv, r6inv, r8inv;
  float TSvdw, TSvdwinv, Vilp;
  float fpair, fsum;

  r2inv = 1.0f / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  // TSvdw = 1.0 + exp(-d_Seff * r + d);
  TSvdw = 1.0f + expf(-d_Seff * r + d);
  TSvdwinv = 1.0f / TSvdw;
  Vilp = -C_6 * r6inv * TSvdwinv;

  // derivatives
  // fpair = -6.0 * C_6 * r8inv * TSvdwinv + \
  //   C_6 * d_Seff * (TSvdw - 1.0) * TSvdwinv * TSvdwinv * r8inv * r;
  fpair = (-6.0f + d_Seff * (TSvdw - 1.0f) * TSvdwinv * r ) * C_6 * TSvdwinv * r8inv;
  fsum = fpair * Tap - Vilp * dTap * rinv;

  p2_vdW = Tap * Vilp;
  f2_vdW = fsum;
  
}

// force evaluation kernel
static __global__ void gpu_find_force(
  ILP_TMD_Para ilp_para,
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  int *g_ilp_neighbor_number,
  int *g_ilp_neighbor_list,
  const int *group_label,
  const int *g_type,
  const double *__restrict__ g_x,
  const double *__restrict__ g_y,
  const double *__restrict__ g_z,
  double *g_fx,
  double *g_fy,
  double *g_fz,
  double *g_virial,
  double *g_potential,
  float *g_f12x,
  float *g_f12y,
  float *g_f12z,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  float s_fx = 0.0f;                                   // force_x
  float s_fy = 0.0f;                                   // force_y
  float s_fz = 0.0f;                                   // force_z
  float s_pe = 0.0f;                                   // potential energy
  float s_sxx = 0.0f;                                  // virial_stress_xx
  float s_sxy = 0.0f;                                  // virial_stress_xy
  float s_sxz = 0.0f;                                  // virial_stress_xz
  float s_syx = 0.0f;                                  // virial_stress_yx
  float s_syy = 0.0f;                                  // virial_stress_yy
  float s_syz = 0.0f;                                  // virial_stress_yz
  float s_szx = 0.0f;                                  // virial_stress_zx
  float s_szy = 0.0f;                                  // virial_stress_zy
  float s_szz = 0.0f;                                  // virial_stress_zz

  float r = 0.0f;
  float rsq = 0.0f;
  float Rcut = 0.0f;

  if (n1 < N2) {
    double x12d, y12d, z12d;
    float x12f, y12f, z12f;
    int neighor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    float delkix_half[NNEI] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float delkiy_half[NNEI] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float delkiz_half[NNEI] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // calculate the normal
    int cont = 0;
    float normal[3];
    float dnormdri[3][3];
    float dnormal[3][NNEI][3];

    float vet[NNEI][3];
    int id, ip, m;
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0f;
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[ip][id] = 0.0f;
        for (m = 0; m < NNEI; ++m) {
          dnormal[id][m][ip] = 0.0f;
          vet[m][id] = 0.0f;
        }
      }
    }

    int ilp_neighbor_number = g_ilp_neighbor_number[n1];
    for (int i1 = 0; i1 < ilp_neighbor_number; ++i1) {
      int n2_ilp = g_ilp_neighbor_list[n1 + number_of_particles * i1];
      x12d = g_x[n2_ilp] - x1;
      y12d = g_y[n2_ilp] - y1;
      z12d = g_z[n2_ilp] - z1;
      apply_mic(box, x12d, y12d, z12d);
      vet[cont][0] = float(x12d);
      vet[cont][1] = float(y12d);
      vet[cont][2] = float(z12d);
      ++cont;

      delkix_half[i1] = float(x12d) * 0.5f;
      delkiy_half[i1] = float(y12d) * 0.5f;
      delkiz_half[i1] = float(z12d) * 0.5f;
    }
    
    calc_normal(vet, cont, normal, dnormdri, dnormal);

    // calculate energy and force
    double tt1,tt2,tt3;
    for (int i1 = 0; i1 < neighor_number; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2];

      tt1 = g_x[n2];
      tt2 = g_y[n2];
      tt3 = g_z[n2];
      x12d = tt1 - x1;
      y12d = tt2 - y1;
      z12d = tt3 - z1;
      apply_mic(box, x12d, y12d, z12d);

      // save x12, y12, z12 in float
      x12f = float(x12d);
      y12f = float(y12d);
      z12f = float(z12d);

      // calculate distance between atoms
      rsq = x12f * x12f + y12f * y12f + z12f * z12f;
      r = sqrtf(rsq);
      Rcut = ilp_para.rcut_global[type1][type2];

      if (r >= Rcut) {
        continue;
      }

      // calc att
      float Tap, dTap, rinv;
      float Rcutinv = 1.0f / Rcut;
      rinv = 1.0f / r;
      Tap = calc_Tap(r, Rcutinv);
      dTap = calc_dTap(r, Rcut, Rcutinv);

      float p2_vdW, f2_vdW;
      calc_vdW(
        r,
        rinv,
        rsq,
        ilp_para.d[type1][type2],
        ilp_para.d_Seff[type1][type2],
        ilp_para.C_6[type1][type2],
        Tap,
        dTap,
        p2_vdW,
        f2_vdW);
      
      float f12x = -f2_vdW * x12f * 0.5f;
      float f12y = -f2_vdW * y12f * 0.5f;
      float f12z = -f2_vdW * z12f * 0.5f;
      float f21x = -f12x;
      float f21y = -f12y;
      float f21z = -f12z;

      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      s_pe += p2_vdW * 0.5f;
      s_sxx += x12f * f21x;
      s_sxy += x12f * f21y;
      s_sxz += x12f * f21z;
      s_syx += y12f * f21x;
      s_syy += y12f * f21y;
      s_syz += y12f * f21z;
      s_szx += z12f * f21x;
      s_szy += z12f * f21y;
      s_szz += z12f * f21z;

      
      // calc rep
      float C = ilp_para.C[type1][type2];
      float lambda_ = ilp_para.lambda[type1][type2];
      float delta2inv = ilp_para.delta2inv[type1][type2];
      float epsilon = ilp_para.epsilon[type1][type2];
      float z0 = ilp_para.z0[type1][type2];
      // calc_rep
      float prodnorm1, rhosq1, rdsq1, exp0, exp1, frho1, Erep, Vilp;
      float fpair, fpair1, fsum, delx, dely, delz, fkcx, fkcy, fkcz;
      float dprodnorm1[3] = {0.0f, 0.0f, 0.0f};
      float fp1[3] = {0.0f, 0.0f, 0.0f};
      float fprod1[3] = {0.0f, 0.0f, 0.0f};
      float fk[3] = {0.0f, 0.0f, 0.0f};

      delx = -x12f;
      dely = -y12f;
      delz = -z12f;

      float delx_half = delx * 0.5f;
      float dely_half = dely * 0.5f;
      float delz_half = delz * 0.5f;

      // calculate the transverse distance
      prodnorm1 = normal[0] * delx + normal[1] * dely + normal[2] * delz;
      rhosq1 = rsq - prodnorm1 * prodnorm1;
      rdsq1 = rhosq1 * delta2inv;

      // store exponents
      // exp0 = exp(-lambda_ * (r - z0));
      // exp1 = exp(-rdsq1);
      exp0 = expf(-lambda_ * (r - z0));
      exp1 = expf(-rdsq1);

      frho1 = exp1 * C;
      Erep = 0.5f * epsilon + frho1;
      Vilp = exp0 * Erep;

      // derivatives
      fpair = lambda_ * exp0 * rinv * Erep;
      fpair1 = 2.0f * exp0 * frho1 * delta2inv;
      fsum = fpair + fpair1;

      float prodnorm1_m_fpair1 = prodnorm1 * fpair1;
      float Vilp_m_dTap_m_rinv = Vilp * dTap * rinv;

      // derivatives of the product of rij and ni, the resutl is a vector
      dprodnorm1[0] = 
        dnormdri[0][0] * delx + dnormdri[1][0] * dely + dnormdri[2][0] * delz;
      dprodnorm1[1] = 
        dnormdri[0][1] * delx + dnormdri[1][1] * dely + dnormdri[2][1] * delz;
      dprodnorm1[2] = 
        dnormdri[0][2] * delx + dnormdri[1][2] * dely + dnormdri[2][2] * delz;
      // fp1[0] = prodnorm1 * normal[0] * fpair1;
      // fp1[1] = prodnorm1 * normal[1] * fpair1;
      // fp1[2] = prodnorm1 * normal[2] * fpair1;
      // fprod1[0] = prodnorm1 * dprodnorm1[0] * fpair1;
      // fprod1[1] = prodnorm1 * dprodnorm1[1] * fpair1;
      // fprod1[2] = prodnorm1 * dprodnorm1[2] * fpair1;
      fp1[0] = prodnorm1_m_fpair1 * normal[0];
      fp1[1] = prodnorm1_m_fpair1 * normal[1];
      fp1[2] = prodnorm1_m_fpair1 * normal[2];
      fprod1[0] = prodnorm1_m_fpair1 * dprodnorm1[0];
      fprod1[1] = prodnorm1_m_fpair1 * dprodnorm1[1];
      fprod1[2] = prodnorm1_m_fpair1 * dprodnorm1[2];

      // fkcx = (delx * fsum - fp1[0]) * Tap - Vilp * dTap * delx * rinv;
      // fkcy = (dely * fsum - fp1[1]) * Tap - Vilp * dTap * dely * rinv;
      // fkcz = (delz * fsum - fp1[2]) * Tap - Vilp * dTap * delz * rinv;
      fkcx = (delx * fsum - fp1[0]) * Tap - Vilp_m_dTap_m_rinv * delx;
      fkcy = (dely * fsum - fp1[1]) * Tap - Vilp_m_dTap_m_rinv * dely;
      fkcz = (delz * fsum - fp1[2]) * Tap - Vilp_m_dTap_m_rinv * delz;

      s_fx += fkcx - fprod1[0] * Tap;
      s_fy += fkcy - fprod1[1] * Tap;
      s_fz += fkcz - fprod1[2] * Tap;

      g_f12x[index] = fkcx;
      g_f12y[index] = fkcy;
      g_f12z[index] = fkcz;

      float minus_prodnorm1_m_fpair1_m_Tap = -prodnorm1 * fpair1 * Tap;
      for (int kk = 0; kk < ilp_neighbor_number; ++kk) {
      // for (int kk = 0; kk < 0; ++kk) {
        // int index_ilp = n1 + number_of_particles * kk;
        // int n2_ilp = g_ilp_neighbor_list[index_ilp];
        // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
        dprodnorm1[0] = dnormal[0][kk][0] * delx + dnormal[1][kk][0] * dely +
            dnormal[2][kk][0] * delz;
        dprodnorm1[1] = dnormal[0][kk][1] * delx + dnormal[1][kk][1] * dely +
            dnormal[2][kk][1] * delz;
        dprodnorm1[2] = dnormal[0][kk][2] * delx + dnormal[1][kk][2] * dely +
            dnormal[2][kk][2] * delz;
        // fk[0] = (-prodnorm1 * dprodnorm1[0] * fpair1) * Tap;
        // fk[1] = (-prodnorm1 * dprodnorm1[1] * fpair1) * Tap;
        // fk[2] = (-prodnorm1 * dprodnorm1[2] * fpair1) * Tap;
        fk[0] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[0];
        fk[1] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[1];
        fk[2] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[2];

        g_f12x_ilp_neigh[n1 + number_of_particles * kk] += fk[0];
        g_f12y_ilp_neigh[n1 + number_of_particles * kk] += fk[1];
        g_f12z_ilp_neigh[n1 + number_of_particles * kk] += fk[2];

        // delki[0] = g_x[n2_ilp] - x1;
        // delki[1] = g_y[n2_ilp] - y1;
        // delki[2] = g_z[n2_ilp] - z1;
        // apply_mic(box, delki[0], delki[1], delki[2]);

        // s_sxx += delki[0] * fk[0] * 0.5;
        // s_sxy += delki[0] * fk[1] * 0.5;
        // s_sxz += delki[0] * fk[2] * 0.5;
        // s_syx += delki[1] * fk[0] * 0.5;
        // s_syy += delki[1] * fk[1] * 0.5;
        // s_syz += delki[1] * fk[2] * 0.5;
        // s_szx += delki[2] * fk[0] * 0.5;
        // s_szy += delki[2] * fk[1] * 0.5;
        // s_szz += delki[2] * fk[2] * 0.5;
        s_sxx += delkix_half[kk] * fk[0];
        s_sxy += delkix_half[kk] * fk[1];
        s_sxz += delkix_half[kk] * fk[2];
        s_syx += delkiy_half[kk] * fk[0];
        s_syy += delkiy_half[kk] * fk[1];
        s_syz += delkiy_half[kk] * fk[2];
        s_szx += delkiz_half[kk] * fk[0];
        s_szy += delkiz_half[kk] * fk[1];
        s_szz += delkiz_half[kk] * fk[2];
      }
      s_pe += Tap * Vilp;
      s_sxx += delx_half * fkcx;
      s_sxy += delx_half * fkcy;
      s_sxz += delx_half * fkcz;
      s_syx += dely_half * fkcx;
      s_syy += dely_half * fkcy;
      s_syz += dely_half * fkcz;
      s_szx += delz_half * fkcx;
      s_szy += delz_half * fkcy;
      s_szz += delz_half * fkcz;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;

    // save potential
    g_potential[n1] += s_pe;

  }
}

// build a neighbor list for reducing force
static __global__ void build_reduce_neighbor_list(
  const int number_of_particles,
  const int N1,
  const int N2,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  int *g_reduce_neighbor_list)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (N1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int l, r, m, tmp_value;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + i1 * number_of_particles;
      int n2 = g_neighbor_list[index];

      l = 0;
      r = g_neighbor_number[n2];
      while (l < r) {
        m = (l + r) >> 1;
        tmp_value = g_neighbor_list[n2 + number_of_particles * m];
        if (tmp_value < n1) {
          l = m + 1;
        } else if (tmp_value > n1) {
          r = m - 1;
        } else {
          break;
        }
      }
      g_reduce_neighbor_list[index] = (l + r) >> 1;
    }
  }
}

// reduce the rep force
static __global__ void reduce_force_many_body(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int *g_neighbor_number,
  const int *g_neighbor_list,
  int *g_reduce_neighbor_list,
  int *g_ilp_neighbor_number,
  int *g_ilp_neighbor_list,
  const double *__restrict__ g_x,
  const double *__restrict__ g_y,
  const double *__restrict__ g_z,
  double *g_fx,
  double *g_fy,
  double *g_fz,
  double *g_virial,
  float *g_f12x,
  float *g_f12y,
  float *g_f12z,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  float s_fx = 0.0f;                                   // force_x
  float s_fy = 0.0f;                                   // force_y
  float s_fz = 0.0f;                                   // force_z
  float s_sxx = 0.0f;                                  // virial_stress_xx
  float s_sxy = 0.0f;                                  // virial_stress_xy
  float s_sxz = 0.0f;                                  // virial_stress_xz
  float s_syx = 0.0f;                                  // virial_stress_yx
  float s_syy = 0.0f;                                  // virial_stress_yy
  float s_syz = 0.0f;                                  // virial_stress_yz
  float s_szx = 0.0f;                                  // virial_stress_zx
  float s_szy = 0.0f;                                  // virial_stress_zy
  float s_szz = 0.0f;                                  // virial_stress_zz


  if (n1 < N2) {
    double x12d, y12d, z12d;
    float x12f, y12f, z12f;
    int neighbor_number_1 = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    // calculate energy and force
    for (int i1 = 0; i1 < neighbor_number_1; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);
      x12f = float(x12d);
      y12f = float(y12d);
      z12f = float(z12d);

      index = n2 + number_of_particles * g_reduce_neighbor_list[index];
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      s_fx -= f21x;
      s_fy -= f21y;
      s_fz -= f21z;

      // per-atom virial
      s_sxx += x12f * f21x * 0.5f;
      s_sxy += x12f * f21y * 0.5f;
      s_sxz += x12f * f21z * 0.5f;
      s_syx += y12f * f21x * 0.5f;
      s_syy += y12f * f21y * 0.5f;
      s_syz += y12f * f21z * 0.5f;
      s_szx += z12f * f21x * 0.5f;
      s_szy += z12f * f21y * 0.5f;
      s_szz += z12f * f21z * 0.5f;
    }

    int ilp_neighbor_number_1 = g_ilp_neighbor_number[n1];

    for (int i1 = 0; i1 < ilp_neighbor_number_1; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_ilp_neighbor_list[index];
      int ilp_neighor_number_2 = g_ilp_neighbor_number[n2];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);
      x12f = float(x12d);
      y12f = float(y12d);
      z12f = float(z12d);

      int offset = 0;
      for (int k = 0; k < ilp_neighor_number_2; ++k) {
        if (n1 == g_ilp_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = n2 + number_of_particles * offset;
      float f21x = g_f12x_ilp_neigh[index];
      float f21y = g_f12y_ilp_neigh[index];
      float f21z = g_f12z_ilp_neigh[index];

      s_fx += f21x;
      s_fy += f21y;
      s_fz += f21z;

      // per-atom virial
      s_sxx += -x12f * f21x * 0.5f;
      s_sxy += -x12f * f21y * 0.5f;
      s_sxz += -x12f * f21z * 0.5f;
      s_syx += -y12f * f21x * 0.5f;
      s_syy += -y12f * f21y * 0.5f;
      s_syz += -y12f * f21z * 0.5f;
      s_szx += -z12f * f21x * 0.5f;
      s_szy += -z12f * f21y * 0.5f;
      s_szz += -z12f * f21z * 0.5f;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}

// SW term
// two-body part of the SW potential
static __device__ void find_p2_and_f2(
  double sigma, double a, double B, double epsilon_times_A, double d12, double& p2, double& f2)
{
  double r12 = d12 / sigma;
  double B_over_r12power4 = B / (r12 * r12 * r12 * r12);
  double exp_factor = epsilon_times_A * exp(1.0 / (r12 - a));
  p2 = exp_factor * (B_over_r12power4 - 1.0);
  f2 = -p2 / ((r12 - a) * (r12 - a)) - exp_factor * 4.0 * B_over_r12power4 / r12;
  f2 /= (sigma * d12);
}

// find the partial forces dU_i/dr_ij of SW potential
static __global__ void gpu_find_force_sw3_partial(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const SW2_Para sw3,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_potential,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    double potential_energy = 0.0;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double d12inv = 1.0 / d12;
      if (d12 >= sw3.rc[type1][type2]) {
        continue;
      }

      double gamma12 = sw3.gamma[type1][type2];
      double sigma12 = sw3.sigma[type1][type2];
      double a12 = sw3.a[type1][type2];
      double tmp = gamma12 / (sigma12 * (d12 / sigma12 - a12) * (d12 / sigma12 - a12));
      double p2, f2;
      find_p2_and_f2(sigma12, a12, sw3.B[type1][type2], sw3.A[type1][type2], d12, p2, f2);

      // treat the two-body part in the same way as the many-body part
      double f12x = f2 * x12 * 0.5;
      double f12y = f2 * y12 * 0.5;
      double f12z = f2 * z12 * 0.5;
      // accumulate potential energy
      potential_energy += p2 * 0.5;

      // three-body part
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + number_of_particles * i2];
        if (n3 == n2) {
          continue;
        }
        int type3 = g_type[n3];
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        if (d13 >= sw3.rc[type1][type3]) {
          continue;
        }

        double cos0 = sw3.cos0[type1][type2][type3];
        double lambda = sw3.lambda[type1][type2][type3];
        double exp123 = d13 / sw3.sigma[type1][type3] - sw3.a[type1][type3];
        exp123 = sw3.gamma[type1][type3] / exp123;
        exp123 = exp(gamma12 / (d12 / sigma12 - a12) + exp123);
        double one_over_d12d13 = 1.0 / (d12 * d13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) * one_over_d12d13;
        double cos123_over_d12d12 = cos123 * d12inv * d12inv;
        // cos123 - cos0
        double delta_cos = cos123 - cos0;

        // modification to (cos123 - cos0)
        double abs_delta_cos = fabs(delta_cos);
        if (abs_delta_cos >= DELTA2) {
          delta_cos = 0.0;
        } else if (abs_delta_cos < DELTA2 && abs_delta_cos > DELTA1) {
          double factor = 0.5 + 0.5 * cos(PI * (abs_delta_cos - DELTA1) / (DELTA2 - DELTA1));
          delta_cos *= factor;
        }

        // double tmp1 = exp123 * (cos123 - cos0) * lambda;
        // double tmp2 = tmp * (cos123 - cos0) * d12inv;
        double tmp1 = exp123 * delta_cos * lambda;
        double tmp2 = tmp * delta_cos * d12inv;

        // accumulate potential energy
        // potential_energy += (cos123 - cos0) * tmp1 * 0.5;
        // double tmp_e = (cos123 - cos0) * tmp1 * 0.5;
        double tmp_e = delta_cos * tmp1 * 0.5;
        potential_energy += tmp_e;

        double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
        f12x += tmp1 * (2.0 * cos_d - tmp2 * x12);

        cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
        f12y += tmp1 * (2.0 * cos_d - tmp2 * y12);

        cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
        f12z += tmp1 * (2.0 * cos_d - tmp2 * z12);
      }
      g_f12x[index] = f12x;
      g_f12y[index] = f12y;
      g_f12z[index] = f12z;
    }
    // save potential
    g_potential[n1] += potential_energy;
  }
}

// define the pure virtual func
void ILP_TMD_SW::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom)
{
  // nothing
}

#define USE_FIXED_NEIGHBOR 1
#define UPDATE_TEMP 10
#define BIG_ILP_CUTOFF_SQUARE 50.0
// find force and related quantities
void ILP_TMD_SW::compute_ilp(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom,
  std::vector<Group> &group)
{
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

  // assume the first group column is for ILP
  const int *group_label = group[0].label.data();

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor_ilp(
      N1,
      N2,
      rc,
      BIG_ILP_CUTOFF_SQUARE,
      box,
      group_label,
      type,
      position_per_atom,
      ilp_data.cell_count,
      ilp_data.cell_count_sum,
      ilp_data.cell_contents,
      ilp_data.NN,
      ilp_data.NL,
      ilp_data.big_ilp_NN,
      ilp_data.big_ilp_NL);
    
    find_neighbor_SW(
      N1,
      N2,
      rc_sw,
      box,
      group_label,
      type,
      position_per_atom,
      sw2_data.cell_count,
      sw2_data.cell_count_sum,
      sw2_data.cell_contents,
      sw2_data.NN,
      sw2_data.NL
    );

    build_reduce_neighbor_list<<<grid_size, BLOCK_SIZE_FORCE>>>(
      number_of_atoms,
      N1,
      N2,
      ilp_data.NN.data(),
      ilp_data.NL.data(),
      ilp_data.reduce_NL.data());
#ifdef USE_FIXED_NEIGHBOR
  }
  num_calls %= UPDATE_TEMP;
#endif

  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + number_of_atoms;
  const double* z = position_per_atom.data() + number_of_atoms * 2;
  const int *NN = ilp_data.NN.data();
  const int *NL = ilp_data.NL.data();
  const int* big_ilp_NN = ilp_data.big_ilp_NN.data();
  const int* big_ilp_NL = ilp_data.big_ilp_NL.data();
  int *reduce_NL = ilp_data.reduce_NL.data();
  int *ilp_NL = ilp_data.ilp_NL.data();
  int *ilp_NN = ilp_data.ilp_NN.data();

  const int* NN_sw = sw2_data.NN.data();
  const int* NL_sw = sw2_data.NL.data();

  ilp_data.ilp_NL.fill(0);
  ilp_data.ilp_NN.fill(0);

  // find ILP neighbor list
  ILP_neighbor<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, big_ilp_NN, big_ilp_NL, \
    type.data(), ilp_para, x, y, z, ilp_NN, \
    ilp_NL, group[1].label.data());
  GPU_CHECK_KERNEL

  // initialize force of ilp neighbor temporary vector
  ilp_data.f12x_ilp_neigh.fill(0);
  ilp_data.f12y_ilp_neigh.fill(0);
  ilp_data.f12z_ilp_neigh.fill(0);
  ilp_data.f12x.fill(0);
  ilp_data.f12y.fill(0);
  ilp_data.f12z.fill(0);

  sw2_data.f12x.fill(0);
  sw2_data.f12y.fill(0);
  sw2_data.f12z.fill(0);

  double *g_fx = force_per_atom.data();
  double *g_fy = force_per_atom.data() + number_of_atoms;
  double *g_fz = force_per_atom.data() + number_of_atoms * 2;
  double *g_virial = virial_per_atom.data();
  double *g_potential = potential_per_atom.data();
  float *g_f12x = ilp_data.f12x.data();
  float *g_f12y = ilp_data.f12y.data();
  float *g_f12z = ilp_data.f12z.data();
  float *g_f12x_ilp_neigh = ilp_data.f12x_ilp_neigh.data();
  float *g_f12y_ilp_neigh = ilp_data.f12y_ilp_neigh.data();
  float *g_f12z_ilp_neigh = ilp_data.f12z_ilp_neigh.data();

  gpu_find_force<<<grid_size, BLOCK_SIZE_FORCE>>>(
    ilp_para,
    number_of_atoms,
    N1,
    N2,
    box,
    NN,
    NL,
    ilp_NN,
    ilp_NL,
    group_label,
    type.data(),
    x,
    y,
    z,
    g_fx,
    g_fy,
    g_fz,
    g_virial,
    g_potential,
    g_f12x,
    g_f12y,
    g_f12z,
    g_f12x_ilp_neigh,
    g_f12y_ilp_neigh,
    g_f12z_ilp_neigh);
  GPU_CHECK_KERNEL

  reduce_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    NN,
    NL,
    reduce_NL,
    ilp_NN,
    ilp_NL,
    x,
    y,
    z,
    g_fx,
    g_fy,
    g_fz,
    g_virial,
    g_f12x,
    g_f12y,
    g_f12z,
    g_f12x_ilp_neigh,
    g_f12y_ilp_neigh,
    g_f12z_ilp_neigh);
    GPU_CHECK_KERNEL

  // step 1: calculate the partial forces
  gpu_find_force_sw3_partial<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, sw2_para, sw2_data.NN.data(), sw2_data.NL.data(),
    type.data(), position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(), sw2_data.f12x.data(),
    sw2_data.f12y.data(), sw2_data.f12z.data());
  GPU_CHECK_KERNEL

  // step 2: calculate force and related quantities
  find_properties_many_body(
    box, sw2_data.NN.data(), sw2_data.NL.data(), sw2_data.f12x.data(),
    sw2_data.f12y.data(), sw2_data.f12z.data(), position_per_atom, force_per_atom, virial_per_atom);
}