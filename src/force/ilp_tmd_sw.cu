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

#define BLOCK_SIZE_FORCE 128

// there are most 6 intra-layer neighbors for TMD
#define NNEI 6

ILP_TMD_SW::ILP_TMD_SW(FILE* fid, int num_types, int num_atoms)
{
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP_TMD_SW)) {
    PRINT_INPUT_ERROR("Incorrect type number of ILP_TMD_SW parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid, "%s", atom_symbol);
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
      int count = fscanf(fid, "%f%f%f%f%f%f%f%f%f%f%f%f", \
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

  // init constant cutoff coeff
  float h_tap_coeff[8] = \
    {1.0f, 0.0f, 0.0f, 0.0f, -35.0f, 84.0f, -70.0f, 20.0f};
  cudaMemcpyToSymbol(Tap_coeff_tmd, h_tap_coeff, 8 * sizeof(float));
  CUDA_CHECK_KERNEL

  // set ilp_flag to 1
  ilp_flag = 1;
}

ILP_TMD_SW::~ILP_TMD_SW(void)
{
  // nothing
}

// calculate the long-range cutoff term
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
  ILP_TMD_SW_Para ilp_para,
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