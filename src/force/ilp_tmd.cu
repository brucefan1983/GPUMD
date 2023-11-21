
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
The class dealing with the Lennard-Jones (LJ) pairwise potentials.
------------------------------------------------------------------------------*/

#include "ilp_tmd.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"

#define BLOCK_SIZE_FORCE 128

// TMD
#define NNEI 6

ILP_TMD::ILP_TMD(FILE* fid, int num_types, int num_atoms)
{
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP_TMD)) {
    PRINT_INPUT_ERROR("Incorrect number of ILP parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for ILP potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  // read parameters
  double beta, alpha, delta, epsilon, C, d, sR;
  double reff, C6, S, rcut_ilp, rcut_global;
  rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", \
      &beta, &alpha, &delta, &epsilon, &C, &d, &sR, &reff, &C6, &S, \
      &rcut_ilp, &rcut_global);
      PRINT_SCANF_ERROR(count, 12, "Reading error for ILP potential.");

      ilp_para.C[n][m] = C;
      ilp_para.C_6[n][m] = C6;
      ilp_para.d[n][m] = d;
      ilp_para.d_Seff[n][m] = d / sR / reff;
      ilp_para.epsilon[n][m] = epsilon;
      ilp_para.z0[n][m] = beta;
      ilp_para.lambda[n][m] = alpha / beta;
      ilp_para.delta2inv[n][m] = 1.0 / (delta * delta); //TODO: how faster?
      ilp_para.S[n][m] = S;
      ilp_para.rcutsq_ilp[n][m] = rcut_ilp * rcut_ilp;
      ilp_para.rcut_global[n][m] = rcut_global;
      double meV = 1e-3 * S;
      ilp_para.C[n][m] *= meV;
      ilp_para.C_6[n][m] *= meV;
      ilp_para.epsilon[n][m] *= meV;

      // TODO: ILP has taper function, check if necessary
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
  double h_tap_coeff[8] = \
    {1.0, 0.0, 0.0, 0.0, -35.0, 84.0, -70.0, 20.0};
  cudaMemcpyToSymbol(Tap_coeff_tmd, h_tap_coeff, 8 * sizeof(double));
  CUDA_CHECK_KERNEL
}

ILP_TMD::~ILP_TMD(void)
{
  // TODO
}

// calculate the long-range cutoff term
inline static __device__ double calc_Tap(const double r_ij, const double Rcutinv)
{
  double Tap, r;

  r = r_ij * Rcutinv;
  // printf("!!!!! r[%24.16f] r_ij[%24.16f] !!!!!\n", r, r_ij);
  if (r >= 1.0) {
    Tap = 0.0;
  } else {
    Tap = Tap_coeff_tmd[7];
    for (int i = 6; i >= 0; --i) {
      Tap = Tap * r + Tap_coeff_tmd[i];
    }
  }

  return Tap;
}

// calculate the derivatives of long-range cutoff term
inline static __device__ double calc_dTap(const double r_ij, const double Rcut, const double Rcutinv)
{
  double dTap, r;
  
  r = r_ij * Rcutinv;
  if (r >= Rcut) {
    dTap = 0.0;
  } else {
    dTap = 7.0 * Tap_coeff_tmd[7];
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
      // TODO: use local memory to save rcutsq to reduce global read
      double rcutsq = ilp_para.rcutsq_ilp[type1][type2];


      // if (n1 == 1) {
      //   printf("#$#$#$ x1[%f] y1[%f] z1[%f] n2[%d] x12[%f] y12[%f] z12[%f] \n", x1, y1, z1, n2, x12, y12, z12);
      //   printf("#$#$#$ gl1[%d] gl2[%d] d12sq[%f] rcsq[%f]\n", group_label[n1], group_label[n2], d12sq, rcutsq);
      // }
      if (group_label[n1] == group_label[n2] && d12sq < rcutsq && type1 == type2 && d12sq != 0) {
        // ilp_neighbor_list[count++ * number_of_particles + n1] = n2;
        neighptr[count++] = n2;
        // if (n1 == 1) {
        //   printf("!@!@ x1[%f] y1[%f] z1[%f] count[%d] n2[%d]\n", x1, y1, z1, count, n2);
        // }
      }
    }
    // if (n1 == 1) printf("$$ nn[%d] cont[%d] \n", neighbor_number, count);

    // TMD
    for (int ll = 0; ll < count; ++ll) {
      neighsort[ll] = neighptr[ll];
      // if (n1 == 1) {
      //   printf("##### nei%d[%d]\n", ll, neighsort[ll]);
      // }
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
          // if (n1 == 1) {printf("##### nei%d[%d]\n", jj, neighsort[0]);}
          check[jj] = -1;
          break;
        }
      }
    } else if (count > NNEI) {
      if (n1 == 1)
      {
        printf("ERROR in ILP NEIGHBOR LIST\n");
      printf("\n===== ILP neighbor number[%d] is greater than 3 =====\n", count);
      
      int nei1 = neighptr[0];
      int nei2 = neighptr[1];
      int nei3 = neighptr[2];
      int nei4 = neighptr[3];
      int nei5 = neighptr[4];
      int nei6 = neighptr[5];
      int nei7 = neighptr[6];
      printf("===== n1[%d] nei1[%d] nei2 [%d] nei3[%d] nei4[%d] nei5[%d] nei6[%d] nei7[%d] =====\n", n1, nei1, nei2, nei3, nei4, nei5, nei6, nei7);
      printf("===== n1[%d] x[%f] y[%f] z[%f] =====\n", n1, g_x[n1], g_y[n1], g_z[n1]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei1, g_x[nei1], g_y[nei1], g_z[nei1]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei2, g_x[nei2], g_y[nei2], g_z[nei2]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei3, g_x[nei3], g_y[nei3], g_z[nei3]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei4, g_x[nei4], g_y[nei4], g_z[nei4]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei5, g_x[nei5], g_y[nei5], g_z[nei5]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei6, g_x[nei6], g_y[nei6], g_z[nei6]);
      printf("===== nei[%d] x[%f] y[%f] z[%f] =====\n", nei7, g_x[nei7], g_y[nei7], g_z[nei7]);
      }
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
          // if (n1 == 1) {
          //   printf("*&*&*& l[%d]\n", l);
          // }
          check[ll] = -1;
          break;
        }
        ++ll;
      }
    }
    ilp_neighbor_number[n1] = count;
    for (int jj = 0; jj < count; ++jj) {
      ilp_neighbor_list[jj * number_of_particles + n1] = neighsort[jj];
      // if (n1 == 1) {
      //   int ppp=neighsort[jj];
      //   int ttx=g_x[ppp];
      //   int tty=g_y[ppp];
      //   int ttz=g_z[ppp];
      //   printf("(((((nei%d[%d] x[%f] y[%f] z[%f] \n", jj, neighsort[jj], ttx, tty, ttz);
      // }
    }

    if (count > MAX_ILP_NEIGHBOR_TMD) {
      // error, there are too many neighbors for some atoms, 
      printf("\n===== ILP neighbor number[%d] is greater than 3 =====\n", count);
      
      int nei1 = ilp_neighbor_list[0 * number_of_particles + n1];
      int nei2 = ilp_neighbor_list[1 * number_of_particles + n1];
      int nei3 = ilp_neighbor_list[2 * number_of_particles + n1];
      int nei4 = ilp_neighbor_list[3 * number_of_particles + n1];
      printf("===== n1[%d] nei1[%d] nei2 [%d] nei3[%d] nei4[%d] =====\n", n1, nei1, nei2, nei3, nei4);
      return;
      // please check your configuration
    }
  }
}

// TMD
inline static __device__ int modulo(int k, int range)
{
  return (k + range) % range;
}

// calculate the normals and its derivatives
static __device__ void calc_normal(
  int t,
  double (&vect)[NNEI][3],
  int cont,
  double (&normal)[3],
  double (&dnormdri)[3][3],
  double (&dnormal)[3][NNEI][3])
{
  int id, ip, m;
  double pv12[3], pv31[3], pv23[3], n1[3], dni[3];
  double dnn[3][3], dpvdri[3][3];
  double dn1[3][3][3], dpv12[3][3][3], dpv23[3][3][3], dpv31[3][3][3];
  // TMD
  double Nave[3], pvet[NNEI][3], dpvet1[NNEI][3][3], dpvet2[NNEI][3][3], dNave[3][NNEI][3];

  double nninv, continv;

  // initialize the arrays
  for (id = 0; id < 3; id++) {
    pv12[id] = 0.0;
    pv31[id] = 0.0;
    pv23[id] = 0.0;
    n1[id] = 0.0;
    dni[id] = 0.0;

    // TMD
    Nave[id] = 0.0;
    for (ip = 0; ip < 3; ip++) {
      // dnn[ip][id] = 0.0f;
      dpvdri[ip][id] = 0.0;
      for (m = 0; m < NNEI; m++) {
        dpv12[ip][id][m] = 0.0;
        dpv31[ip][id][m] = 0.0;
        dpv23[ip][id][m] = 0.0;
        dn1[ip][id][m] = 0.0;

        // TMD
        dnn[m][id] = 0.0;
        pvet[m][id] = 0.0;
        dpvet1[m][ip][id] = 0.0;
        dpvet2[m][ip][id] = 0.0;
        dNave[id][m][ip] = 0.0;
      }
    }
  }

  if (cont <= 1) {
    normal[0] = 0.0;
    normal[1] = 0.0;
    normal[2] = 1.0;
    for (id = 0; id < 3; ++id) {
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[id][ip] = 0.0;
        for (m = 0; m < NNEI; ++m) {
          dnormal[id][m][ip] = 0.0;
        }
      }
    }
  } else if (cont > 1 && cont < NNEI) {
    for (int k = 0; k < cont - 1; ++k) {
      for (ip = 0; ip < 3; ++ip) {
        pvet[k][ip] = vect[k][modulo(ip + 1, 3)] * vect[k + 1][modulo(ip + 2, 3)] -
                vect[k][modulo(ip + 2, 3)] * vect[k + 1][modulo(ip + 1, 3)];
                // if (t == 10) {
                // printf("@@@@@ pvet[%d][%d] = %f @@@@@\n", k, ip, pvet[k][ip]);
                // printf("***** pvet[%d][%d] = vect[%d][%d] * vect[%d][%d] - vect[%d][%d] * vect[%d][%d] *****\n", k, ip, k, 
                // modulo(ip + 1, 3), k+1, modulo(ip+2, 3), k, modulo(ip+2, 3), k, modulo(ip+1, 3));}
      }
      // TODO: don't need modulo here because k+1<cont<NNEI
      // dpvet1[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l
      // derivatives respect to atom l
      // dNik,x/drl
      dpvet1[k][0][0] = 0.0;
      dpvet1[k][0][1] = vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][0][2] = -vect[modulo(k + 1, NNEI)][1];
      // dNik,y/drl
      dpvet1[k][1][0] = -vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][1][1] = 0.0;
      dpvet1[k][1][2] = vect[modulo(k + 1, NNEI)][0];
      // dNik,z/drl
      dpvet1[k][2][0] = vect[modulo(k + 1, NNEI)][1];
      dpvet1[k][2][1] = -vect[modulo(k + 1, NNEI)][0];
      dpvet1[k][2][2] = 0.0;

      // dpvet2[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l+1
      // derivatives respect to atom l+1
      // dNik,x/drl+1
      dpvet2[k][0][0] = 0.0;
      dpvet2[k][0][1] = -vect[modulo(k, NNEI)][2];
      dpvet2[k][0][2] = vect[modulo(k, NNEI)][1];
      // dNik,y/drl+1
      dpvet2[k][1][0] = vect[modulo(k, NNEI)][2];
      dpvet2[k][1][1] = 0.0;
      dpvet2[k][1][2] = -vect[modulo(k, NNEI)][0];
      // dNik,z/drl+1
      dpvet2[k][2][0] = -vect[modulo(k, NNEI)][1];
      dpvet2[k][2][1] = vect[modulo(k, NNEI)][0];
      dpvet2[k][2][2] = 0.0;
    }

    // average the normal vectors by using the NNEI neighboring planes
    for (ip = 0; ip < 3; ip++) {
      Nave[ip] = 0.0;
      for (int k = 0; k < cont - 1; k++) {
        Nave[ip] += pvet[k][ip];
      }
      Nave[ip] /= (cont - 1);
    }
    // pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
    // pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
    // pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
    // // derivatives of pv12[0] to ri
    // dpvdri[0][0] = 0.0f;
    // dpvdri[0][1] = vet[0][2] - vet[1][2];
    // dpvdri[0][2] = vet[1][1] - vet[0][1];
    // // derivatives of pv12[1] to ri
    // dpvdri[1][0] = vet[1][2] - vet[0][2];
    // dpvdri[1][1] = 0.0f;
    // dpvdri[1][2] = vet[0][0] - vet[1][0];
    // // derivatives of pv12[2] to ri
    // dpvdri[2][0] = vet[0][1] - vet[1][1];
    // dpvdri[2][1] = vet[1][0] - vet[0][0];
    // dpvdri[2][2] = 0.0f;

    // dpv12[0][0][0] = 0.0f;
    // dpv12[0][1][0] = vet[1][2];
    // dpv12[0][2][0] = -vet[1][1];
    // dpv12[1][0][0] = -vet[1][2];
    // dpv12[1][1][0] = 0.0f;
    // dpv12[1][2][0] = vet[1][0];
    // dpv12[2][0][0] = vet[1][1];
    // dpv12[2][1][0] = -vet[1][0];
    // dpv12[2][2][0] = 0.0f;

    // // derivatives respect to the second neighbor, atom l
    // dpv12[0][0][1] = 0.0f;
    // dpv12[0][1][1] = -vet[0][2];
    // dpv12[0][2][1] = vet[0][1];
    // dpv12[1][0][1] = vet[0][2];
    // dpv12[1][1][1] = 0.0f;
    // dpv12[1][2][1] = -vet[0][0];
    // dpv12[2][0][1] = -vet[0][1];
    // dpv12[2][1][1] = vet[0][0];
    // dpv12[2][2][1] = 0.0f;

    // // derivatives respect to the third neighbor, atom n
    // // derivatives of pv12 to rn is zero
    // for (id = 0; id < 3; id++) {
    //   for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0f; }
    // }

    // n1[0] = pv12[0];
    // n1[1] = pv12[1];
    // n1[2] = pv12[2];
    // the magnitude of the normal vector
    // nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    // nn = sqrt(nn2);
    // nninv = 1.0 / nn;
    nninv = rnorm3d(Nave[0], Nave[1], Nave[2]);
    //if (t == 16)
    //   printf("##### n1[%d] cont[%d] nninv[%f] Nave[%f %f %f] #####\n", t, cont, nninv, Nave[0], Nave[1], Nave[2]);
    
    // TODO
    // if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
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
          // if (t == 11) {
          //   printf("=====dn[%16.10f]=====\n", dnormal[id][m][ip]);
          // }
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
        // if (t == 11) {
        //   printf("-----dndr[%16.10f]-----\n", dnormdri[id][ip]);
        // }
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
      dpvet1[k][0][0] = 0.0;
      dpvet1[k][0][1] = vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][0][2] = -vect[modulo(k + 1, NNEI)][1];
      // dNik,y/drl
      dpvet1[k][1][0] = -vect[modulo(k + 1, NNEI)][2];
      dpvet1[k][1][1] = 0.0;
      dpvet1[k][1][2] = vect[modulo(k + 1, NNEI)][0];
      // dNik,z/drl
      dpvet1[k][2][0] = vect[modulo(k + 1, NNEI)][1];
      dpvet1[k][2][1] = -vect[modulo(k + 1, NNEI)][0];
      dpvet1[k][2][2] = 0.0;

      // dpvet2[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l+1
      // derivatives respect to atom l+1
      // dNik,x/drl+1
      dpvet2[k][0][0] = 0.0;
      dpvet2[k][0][1] = -vect[modulo(k, NNEI)][2];
      dpvet2[k][0][2] = vect[modulo(k, NNEI)][1];
      // dNik,y/drl+1
      dpvet2[k][1][0] = vect[modulo(k, NNEI)][2];
      dpvet2[k][1][1] = 0.0;
      dpvet2[k][1][2] = -vect[modulo(k, NNEI)][0];
      // dNik,z/drl+1
      dpvet2[k][2][0] = -vect[modulo(k, NNEI)][1];
      dpvet2[k][2][1] = vect[modulo(k, NNEI)][0];
      dpvet2[k][2][2] = 0.0;
    }

    // average the normal vectors by using the NNEI neighboring planes
    for (ip = 0; ip < 3; ++ip) {
      Nave[ip] = 0.0;
      for (int k = 0; k < NNEI; ++k) {
        Nave[ip] += pvet[k][ip];
      }
      Nave[ip] /= NNEI;
    }
    // the magnitude of the normal vector
    // nn2 = Nave[0] * Nave[0] + Nave[1] * Nave[1] + Nave[2] * Nave[2];
    // nn = sqrt(nn2);
    nninv = rnorm3d(Nave[0], Nave[1], Nave[2]);
    // if (nn == 0.0) error->one(FLERR, "The magnitude of the normal vector is zero");
    // the unit normal vector
    normal[0] = Nave[0] * nninv;
    normal[1] = Nave[1] * nninv;
    normal[2] = Nave[2] * nninv;

    // for the central atoms, dnormdri is always zero
    for (id = 0; id < 3; ++id) {
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[id][ip] = 0.0;
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


    // continv = 1.0 / cont;

    // pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
    // pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
    // pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
    // // derivatives respect to the first neighbor, atom k
    // dpv12[0][0][0] = 0.0f;
    // dpv12[0][1][0] = vet[1][2];
    // dpv12[0][2][0] = -vet[1][1];
    // dpv12[1][0][0] = -vet[1][2];
    // dpv12[1][1][0] = 0.0f;
    // dpv12[1][2][0] = vet[1][0];
    // dpv12[2][0][0] = vet[1][1];
    // dpv12[2][1][0] = -vet[1][0];
    // dpv12[2][2][0] = 0.0f;
    // // derivatives respect to the second neighbor, atom l
    // dpv12[0][0][1] = 0.0f;
    // dpv12[0][1][1] = -vet[0][2];
    // dpv12[0][2][1] = vet[0][1];
    // dpv12[1][0][1] = vet[0][2];
    // dpv12[1][1][1] = 0.0f;
    // dpv12[1][2][1] = -vet[0][0];
    // dpv12[2][0][1] = -vet[0][1];
    // dpv12[2][1][1] = vet[0][0];
    // dpv12[2][2][1] = 0.0f;

    // // derivatives respect to the third neighbor, atom n
    // for (id = 0; id < 3; id++) {
    //   for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0f; }
    // }

    // pv31[0] = vet[2][1] * vet[0][2] - vet[0][1] * vet[2][2];
    // pv31[1] = vet[2][2] * vet[0][0] - vet[0][2] * vet[2][0];
    // pv31[2] = vet[2][0] * vet[0][1] - vet[0][0] * vet[2][1];
    // // derivatives respect to the first neighbor, atom k
    // dpv31[0][0][0] = 0.0f;
    // dpv31[0][1][0] = -vet[2][2];
    // dpv31[0][2][0] = vet[2][1];
    // dpv31[1][0][0] = vet[2][2];
    // dpv31[1][1][0] = 0.0f;
    // dpv31[1][2][0] = -vet[2][0];
    // dpv31[2][0][0] = -vet[2][1];
    // dpv31[2][1][0] = vet[2][0];
    // dpv31[2][2][0] = 0.0f;
    // // derivatives respect to the third neighbor, atom n
    // dpv31[0][0][2] = 0.0f;
    // dpv31[0][1][2] = vet[0][2];
    // dpv31[0][2][2] = -vet[0][1];
    // dpv31[1][0][2] = -vet[0][2];
    // dpv31[1][1][2] = 0.0f;
    // dpv31[1][2][2] = vet[0][0];
    // dpv31[2][0][2] = vet[0][1];
    // dpv31[2][1][2] = -vet[0][0];
    // dpv31[2][2][2] = 0.0f;
    // // derivatives respect to the second neighbor, atom l
    // for (id = 0; id < 3; id++) {
    //   for (ip = 0; ip < 3; ip++) { dpv31[id][ip][1] = 0.0f; }
    // }

    // pv23[0] = vet[1][1] * vet[2][2] - vet[2][1] * vet[1][2];
    // pv23[1] = vet[1][2] * vet[2][0] - vet[2][2] * vet[1][0];
    // pv23[2] = vet[1][0] * vet[2][1] - vet[2][0] * vet[1][1];
    // // derivatives respect to the second neighbor, atom k
    // for (id = 0; id < 3; id++) {
    //   for (ip = 0; ip < 3; ip++) { dpv23[id][ip][0] = 0.0f; }
    // }
    // // derivatives respect to the second neighbor, atom l
    // dpv23[0][0][1] = 0.0f;
    // dpv23[0][1][1] = vet[2][2];
    // dpv23[0][2][1] = -vet[2][1];
    // dpv23[1][0][1] = -vet[2][2];
    // dpv23[1][1][1] = 0.0f;
    // dpv23[1][2][1] = vet[2][0];
    // dpv23[2][0][1] = vet[2][1];
    // dpv23[2][1][1] = -vet[2][0];
    // dpv23[2][2][1] = 0.0f;
    // // derivatives respect to the third neighbor, atom n
    // dpv23[0][0][2] = 0.0f;
    // dpv23[0][1][2] = -vet[1][2];
    // dpv23[0][2][2] = vet[1][1];
    // dpv23[1][0][2] = vet[1][2];
    // dpv23[1][1][2] = 0.0f;
    // dpv23[1][2][2] = -vet[1][0];
    // dpv23[2][0][2] = -vet[1][1];
    // dpv23[2][1][2] = vet[1][0];
    // dpv23[2][2][2] = 0.0f;

    // //############################################################################################
    // // average the normal vectors by using the 3 neighboring planes
    // n1[0] = (pv12[0] + pv31[0] + pv23[0]) * continv;
    // n1[1] = (pv12[1] + pv31[1] + pv23[1]) * continv;
    // n1[2] = (pv12[2] + pv31[2] + pv23[2]) * continv;
    // // the magnitude of the normal vector
    // // nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    // // nn = sqrt(nn2);

    // // nninv = 1.0 / nn;
    // nninv = rnorm3df(n1[0], n1[1], n1[2]);
    // // TODO
    // // if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
    // // the unit normal vector
    // normal[0] = n1[0] * nninv;
    // normal[1] = n1[1] * nninv;
    // normal[2] = n1[2] * nninv;

    // // for the central atoms, dnormdri is always zero
    // for (id = 0; id < 3; id++) {
    //   for (ip = 0; ip < 3; ip++) { dnormdri[id][ip] = 0.0f; }
    // }

    // // derivatives of non-normalized normal vector, dn1:3x3x3 array
    // for (id = 0; id < 3; id++) {
    //   for (ip = 0; ip < 3; ip++) {
    //     for (m = 0; m < 3; m++) {
    //       dn1[id][ip][m] = (dpv12[id][ip][m] + dpv23[id][ip][m] + dpv31[id][ip][m]) * continv;
    //     }
    //   }
    // }
    // // derivatives of nn, dnn:3x3 vector
    // // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
    // // r[id][m]: the id's component of atom m
    // for (m = 0; m < 3; m++) {
    //   for (id = 0; id < 3; id++) {
    //     dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) * nninv;
    //   }
    // }
    // // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
    // // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
    // for (m = 0; m < 3; m++) {
    //   for (id = 0; id < 3; id++) {
    //     for (ip = 0; ip < 3; ip++) {
    //       dnormal[id][ip][m] = dn1[id][ip][m] * nninv - n1[id] * dnn[ip][m] * nninv * nninv;
    //     }
    //   }
    // }
  } else {
    // TODO: error! too many neighbors for calculating normals
  }
}

// calculate the van der Waals force and energy
inline static __device__ void calc_vdW(
  double r,
  double rinv,
  double rsq,
  double d,
  double d_Seff,
  double C_6,
  double Tap,
  double dTap,
  double &p2_vdW,
  double &f2_vdW)
{
  double r2inv, r6inv, r8inv;
  double TSvdw, TSvdwinv, Vilp;
  double fpair, fsum;

  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  // TODO: use double
  // TSvdw = 1.0 + exp(-d_Seff * r + d);
  TSvdw = 1.0 + exp(-d_Seff * r + d);
  TSvdwinv = 1.0 / TSvdw;
  Vilp = -C_6 * r6inv * TSvdwinv;

  // derivatives
  // fpair = -6.0 * C_6 * r8inv * TSvdwinv + \
  //   C_6 * d_Seff * (TSvdw - 1.0) * TSvdwinv * TSvdwinv * r8inv * r;
  fpair = (-6.0 + d_Seff * (TSvdw - 1.0) * TSvdwinv * r ) * C_6 * TSvdwinv * r8inv;
  fsum = fpair * Tap - Vilp * dTap * rinv;

  p2_vdW = Tap * Vilp;
  f2_vdW = fsum;
  
}


__device__ double EATT;
__device__ double EREP;

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
  double *g_f12x,
  double *g_f12y,
  double *g_f12z,
  double *g_f12x_ilp_neigh,
  double *g_f12y_ilp_neigh,
  double *g_f12z_ilp_neigh)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  double s_fx = 0.0;                                   // force_x
  double s_fy = 0.0;                                   // force_y
  double s_fz = 0.0;                                   // force_z
  double s_pe = 0.0;                                   // potential energy
  double s_sxx = 0.0;                                  // virial_stress_xx
  double s_sxy = 0.0;                                  // virial_stress_xy
  double s_sxz = 0.0;                                  // virial_stress_xz
  double s_syx = 0.0;                                  // virial_stress_yx
  double s_syy = 0.0;                                  // virial_stress_yy
  double s_syz = 0.0;                                  // virial_stress_yz
  double s_szx = 0.0;                                  // virial_stress_zx
  double s_szy = 0.0;                                  // virial_stress_zy
  double s_szz = 0.0;                                  // virial_stress_zz

  double r = 0.0;
  double rsq = 0.0;
  double Rcut = 0.0;

  if (n1 < N2) {
    double x12d, y12d, z12d;
    double x12f, y12f, z12f;
    int neighor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    int index_ilp_vec[3] = {n1, n1 + number_of_particles, n1 + (number_of_particles << 1)};
    double fk_temp[9] = {0.0};

    double delkix_half[NNEI] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double delkiy_half[NNEI] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double delkiz_half[NNEI] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // calculate the normal
    int cont = 0;
    double normal[3];
    double dnormdri[3][3];
    double dnormal[3][NNEI][3];

    double vet[NNEI][3];
    int id, ip, m;
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0;
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[ip][id] = 0.0;
        for (m = 0; m < NNEI; ++m) {
          dnormal[id][m][ip] = 0.0;
          vet[m][id] = 0.0;
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
      vet[cont][0] = double(x12d);
      vet[cont][1] = double(y12d);
      vet[cont][2] = double(z12d);
      // if (n1 == 1) {
      //   printf("))))n2[%d] x1[%f] y1[%f] z1[%f] x2[%f] y2[%f] z2[%f] \n", n2_ilp, x1, y1, z1, g_x[n2_ilp], g_y[n2_ilp], g_z[n2_ilp]);
      // }
      ++cont;

      delkix_half[i1] = double(x12d) * 0.5;
      delkiy_half[i1] = double(y12d) * 0.5;
      delkiz_half[i1] = double(z12d) * 0.5;
    }

    
    // if (n1 == 10) {
    //   printf("&&&&&&&&&&\n");
    //   for (int ttt = 0; ttt < NNEI; ++ttt) {
    //     for (int yyy = 0; yyy < 3; ++yyy) {
    //       printf("%f ", vet[ttt][yyy]);
    //     }
    //     printf("\n");
    //   }
    //   printf("&&&&&&&&&&\n");
    // }
    calc_normal(n1, vet, cont, normal, dnormdri, dnormal);

    // calculate energy and force
    double tt1,tt2,tt3;
    // TODO: TMD
    double tmp_vdw = 0.0;
    double tmp_rep1 = 0.0;
    double tmp_rep2 = 0.0;
    double tmp_rep3 = 0.0;
    for (int i1 = 0; i1 < neighor_number; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2];

      // TODO shared double?
      tt1 = g_x[n2];
      tt2 = g_y[n2];
      tt3 = g_z[n2];
      x12d = tt1 - x1;
      y12d = tt2 - y1;
      z12d = tt3 - z1;
      // x12d = g_x[n2] - x1;
      // y12d = g_y[n2] - y1;
      // z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);

      // save x12, y12, z12 in double
      x12f = double(x12d);
      y12f = double(y12d);
      z12f = double(z12d);

      // calculate distance between atoms
      rsq = x12f * x12f + y12f * y12f + z12f * z12f;
      r = sqrt(rsq);
      Rcut = ilp_para.rcut_global[type1][type2];
      // not in the same layer
      // if (r >= Rcut || group_label[n1] == group_label[n2]) {
      //   continue;
      // }

      if (r >= Rcut) {
        continue;
      }

      // calc att
      double Tap, dTap, rinv;
      double Rcutinv = 1.0 / Rcut;
      rinv = 1.0 / r;
      Tap = calc_Tap(r, Rcutinv);
      dTap = calc_dTap(r, Rcut, Rcutinv);

      double p2_vdW, f2_vdW;
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
      
      double f12x = -f2_vdW * x12f * 0.5;
      double f12y = -f2_vdW * y12f * 0.5;
      double f12z = -f2_vdW * z12f * 0.5;
      double f21x = -f12x;
      double f21y = -f12y;
      double f21z = -f12z;

      s_fx += f12x - f21x;
      tmp_vdw += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;
      // if (n1 == 7) {
      //   printf("!!!!! n1[%d] x[%f] y[%f] z[%f] fsx[    %f    ] f2_vdw[%f] delx[%f] \n", 
      //   n1, x1, y1, z1, f2_vdW * x12f, f2_vdW, x12f);
      // }

      s_pe += p2_vdW * 0.5;
      // EATT += p2_vdW * 0.5;
      //printf("@@@@@ att: n1[%d] n2[%d] vatt[%24.16f] Tap[%24.16f] @@@@@\n", n1, n2, p2_vdW*0.5, Tap);
      // atomicAdd(&EATT, p2_vdW * 0.5);
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
      double C = ilp_para.C[type1][type2];
      double lambda_ = ilp_para.lambda[type1][type2];
      double delta2inv = ilp_para.delta2inv[type1][type2];
      double epsilon = ilp_para.epsilon[type1][type2];
      double z0 = ilp_para.z0[type1][type2];
      // calc_rep
      double prodnorm1, rhosq1, rdsq1, exp0, exp1, frho1, Erep, Vilp;
      double fpair, fpair1, fsum, delx, dely, delz, fkcx, fkcy, fkcz;
      double dprodnorm1[3] = {0.0, 0.0, 0.0};
      double fp1[3] = {0.0, 0.0, 0.0};
      double fprod1[3] = {0.0, 0.0, 0.0};
      double fk[3] = {0.0, 0.0, 0.0};

      delx = -x12f;
      dely = -y12f;
      delz = -z12f;

      double delx_half = delx * 0.5;
      double dely_half = dely * 0.5;
      double delz_half = delz * 0.5;

      // calculate the transverse distance
      prodnorm1 = normal[0] * delx + normal[1] * dely + normal[2] * delz;
      rhosq1 = rsq - prodnorm1 * prodnorm1;
      rdsq1 = rhosq1 * delta2inv;

      // store exponents
      // exp0 = exp(-lambda_ * (r - z0));
      // exp1 = exp(-rdsq1);
      // TODO: use double
      exp0 = exp(-lambda_ * (r - z0));
      exp1 = exp(-rdsq1);

      frho1 = exp1 * C;
      Erep = 0.5 * epsilon + frho1;
      Vilp = exp0 * Erep;

      // derivatives
      fpair = lambda_ * exp0 * rinv * Erep;
      fpair1 = 2.0 * exp0 * frho1 * delta2inv;
      fsum = fpair + fpair1;

      double prodnorm1_m_fpair1 = prodnorm1 * fpair1;
      double Vilp_m_dTap_m_rinv = Vilp * dTap * rinv;

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
      // if (n1 == 15) {
      //   printf("&&&&& n1[%d] n2[%d] x[%f] y[%f] z[%f] fkcx[    %f    ] Tap[%f] fprod10[%f] fp10[%f] Vilp[%f] dTap[%f] rinv[%f]\n", 
      //   n1, n2, x1, y1, z1, fkcx, Tap, fprod1[0], fp1[0], Vilp, dTap, rinv);
      // }
      // if (n2 == 7) {
      //   printf("&&&&& n1[%d] n2[%d] x[%f] y[%f] z[%f] fkcx[    %f    ] Tap[%f] fprod10[%f] fp10[%f] Vilp[%f] dTap[%f] rinv[%f]\n", 
      //   n1, n2, x1, y1, z1, -fkcx, Tap, fprod1[0], fp1[0], Vilp, dTap, rinv);
      // }

      s_fx += fkcx - fprod1[0] * Tap;
      s_fy += fkcy - fprod1[1] * Tap;
      s_fz += fkcz - fprod1[2] * Tap;
      tmp_rep1 += fkcx - fprod1[0] * Tap;

      g_f12x[index] = fkcx;
      g_f12y[index] = fkcy;
      g_f12z[index] = fkcz;
      tmp_rep2 -=fkcx;

      double minus_prodnorm1_m_fpair1_m_Tap = -prodnorm1 * fpair1 * Tap;
      // if (n1 == 11) printf("\n\n ilp_nn[%d] nei1[%d] \n\n", ilp_neighbor_number, g_ilp_neighbor_list[51]);
      for (int kk = 0; kk < ilp_neighbor_number; ++kk) {
      // for (int kk = 0; kk < 0; ++kk) {
        // int index_ilp = n1 + number_of_particles * kk;
        // int n2_ilp = g_ilp_neighbor_list[index_ilp];
        // if (n2_ilp_vec[kk] == n1) continue;
        // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
        // TODO: TMD trans
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
        // fk_temp[kk] += fk[0];
        // fk_temp[kk + 3] += fk[1];
        // fk_temp[kk + 6] += fk[2];
        // if (g_ilp_neighbor_list[n1+number_of_particles*kk]== 15) {
        //   // printf("----- n1[%d] kk[%d] index[%d]\n", n1, kk, n1+number_of_particles*kk);
        //   printf("&&&&& x[%f] y[%f] z[%f] fk0[    %20.12f    ] prodnorm1[%f] dprodnorm10[%f] fpair[%f] Tap[%f]\n", 
        //   x1, y1, z1, fk[2], prodnorm1, dprodnorm1[0], Tap);
        // }

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

        // TODO: !!!! in LAMMPS, here is for k and j
        // s_sxx += delkix_half[kk] * fk[0];
        // s_sxy += delkix_half[kk] * fk[1];
        // s_sxz += delkix_half[kk] * fk[2];
        // s_syx += delkiy_half[kk] * fk[0];
        // s_syy += delkiy_half[kk] * fk[1];
        // s_syz += delkiy_half[kk] * fk[2];
        // s_szx += delkiz_half[kk] * fk[0];
        // s_szy += delkiz_half[kk] * fk[1];
        // s_szz += delkiz_half[kk] * fk[2];
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
      // EREP += Tap * Vilp;
      // atomicAdd(&EREP, Tap * Vilp);
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
    // if (n1==7) printf("????? tvdw[%f] trep1[%f] trep2[%f] \n", tmp_vdw, tmp_rep1, tmp_rep2);

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // g_f12x_ilp_neigh[index_ilp_vec[0]] = fk_temp[0];
    // g_f12x_ilp_neigh[index_ilp_vec[1]] = fk_temp[1];
    // g_f12x_ilp_neigh[index_ilp_vec[2]] = fk_temp[2];
    // g_f12y_ilp_neigh[index_ilp_vec[0]] = fk_temp[3];
    // g_f12y_ilp_neigh[index_ilp_vec[1]] = fk_temp[4];
    // g_f12y_ilp_neigh[index_ilp_vec[2]] = fk_temp[5];
    // g_f12z_ilp_neigh[index_ilp_vec[0]] = fk_temp[6];
    // g_f12z_ilp_neigh[index_ilp_vec[1]] = fk_temp[7];
    // g_f12z_ilp_neigh[index_ilp_vec[2]] = fk_temp[8];

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
  double *g_f12x,
  double *g_f12y,
  double *g_f12z,
  double *g_f12x_ilp_neigh,
  double *g_f12y_ilp_neigh,
  double *g_f12z_ilp_neigh)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  double s_fx = 0.0;                                   // force_x
  double s_fy = 0.0;                                   // force_y
  double s_fz = 0.0;                                   // force_z
  double s_sxx = 0.0;                                  // virial_stress_xx
  double s_sxy = 0.0;                                  // virial_stress_xy
  double s_sxz = 0.0;                                  // virial_stress_xz
  double s_syx = 0.0;                                  // virial_stress_yx
  double s_syy = 0.0;                                  // virial_stress_yy
  double s_syz = 0.0;                                  // virial_stress_yz
  double s_szx = 0.0;                                  // virial_stress_zx
  double s_szy = 0.0;                                  // virial_stress_zy
  double s_szz = 0.0;                                  // virial_stress_zz


  if (n1 < N2) {
    double x12d, y12d, z12d;
    double x12f, y12f, z12f;
    int neighbor_number_1 = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    //TODO: TMD
    double tmp_rep2 = 0.0;
    double tmp_rep3 = 0.0;
    double tmp_syz_1 = 0;

    // calculate energy and force
    for (int i1 = 0; i1 < neighbor_number_1; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];
      int neighor_number_2 = g_neighbor_number[n2];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);
      x12f = double(x12d);
      y12f = double(y12d);
      z12f = double(z12d);

      // int offset = 0;
      // // for (int k = 0; k < neighor_number_2; ++k) {
      // //   if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
      // //     offset = k;
      // //     break;
      // //   }
      // // }
      // // TODO: binary search
      // int l = 0;
      // int r = neighor_number_2;
      // int m = 0;
      // int tmp_value = 0;
      // while (l < r) {
      //   m = (l + r) >> 1;
      //   tmp_value = g_neighbor_list[n2 + number_of_particles * m];
      //   if (tmp_value < n1) {
      //     l = m + 1;
      //   } else if (tmp_value > n1) {
      //     r = m - 1;
      //   } else {
      //     break;
      //   }
      // }
      // offset = (l + r) >> 1;
      // index = n2 + number_of_particles * offset;
      index = n2 + number_of_particles * g_reduce_neighbor_list[index];
      double f21x = g_f12x[index];
      double f21y = g_f12y[index];
      double f21z = g_f12z[index];

      s_fx -= f21x;
      s_fy -= f21y;
      s_fz -= f21z;

      // per-atom virial
      s_sxx += x12f * f21x * 0.5;
      s_sxy += x12f * f21y * 0.5;
      s_sxz += x12f * f21z * 0.5;
      s_syx += y12f * f21x * 0.5;
      s_syy += y12f * f21y * 0.5;
      s_syz += y12f * f21z * 0.5;
      tmp_syz_1 += y12f * f21z * 0.5;
      s_szx += z12f * f21x * 0.5;
      s_szy += z12f * f21y * 0.5;
      s_szz += z12f * f21z * 0.5;
    }
    tmp_rep2 = s_fx;

    int ilp_neighbor_number_1 = g_ilp_neighbor_number[n1];
    // printf("===== n1[%d] nei0[%d] nei1[%d] nei2[%d] nei3[%d] nei4[%d] nei5[%d]\n", n1, 
    // g_ilp_neighbor_list[n1], g_ilp_neighbor_list[n1+number_of_particles], 
    // g_ilp_neighbor_list[n1+number_of_particles*2],
    // g_ilp_neighbor_list[n1+number_of_particles*3],
    // g_ilp_neighbor_list[n1+number_of_particles*4],
    // g_ilp_neighbor_list[n1+number_of_particles*5]);
    double tmp_syz_2 = 0;

    for (int i1 = 0; i1 < ilp_neighbor_number_1; ++i1) {
    // for (int i1 = 0; i1 < 0; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_ilp_neighbor_list[index];
      int ilp_neighor_number_2 = g_ilp_neighbor_number[n2];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);
      x12f = double(x12d);
      y12f = double(y12d);
      z12f = double(z12d);

      int offset = 0;
      for (int k = 0; k < ilp_neighor_number_2; ++k) {
        if (n1 == g_ilp_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = n2 + number_of_particles * offset;
      double f21x = g_f12x_ilp_neigh[index];
      double f21y = g_f12y_ilp_neigh[index];
      double f21z = g_f12z_ilp_neigh[index];

      s_fx += f21x;
      // if (n1 == 15) printf("!!!!! n2[%d] offset[%d] index[%d] fx[    %20.12f    ] \n", n2, offset, index, f21z);
      tmp_rep3 += f21x;
      s_fy += f21y;
      s_fz += f21z;

      // per-atom virial
      s_sxx += -x12f * f21x * 0.5;
      s_sxy += -x12f * f21y * 0.5;
      s_sxz += -x12f * f21z * 0.5;
      s_syx += -y12f * f21x * 0.5;
      s_syy += -y12f * f21y * 0.5;
      s_syz += -y12f * f21z * 0.5;
      tmp_syz_2 += -y12f * f21z * 0.5;
      s_szx += -z12f * f21x * 0.5;
      s_szy += -z12f * f21y * 0.5;
      s_szz += -z12f * f21z * 0.5;
    }
    // if (n1 == 15) printf("##### tmp_syz_1[%20.12f] tmp_syz_2[%20.12f]\n", tmp_syz_1, tmp_syz_2);

    // save force
    // if (n1 == 7) printf("+++++ reduce: trep2[%f] trep3[%f] +++++\n", tmp_rep2, tmp_rep3);
    g_fx[n1] += s_fx;
    // if (n1 == 7) printf("+++++ reduce: g_fx[%f] s_fx[%f] +++++\n", g_fx[n1], s_fx);
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


void ILP_TMD::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom)
{
  // TODO
}


#define USE_FIXED_NEIGHBOR 1
#define UPDATE_TEMP 1
#define BIG_ILP_CUTOFF_SQUARE 50.0
// find force and related quantities
void ILP_TMD::compute(
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

  // TODO: assume the first group column is for ILP
  const int *group_label = group[0].label.data();

// what's this??
#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor(
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

  ilp_data.ilp_NL.fill(0);
  ilp_data.ilp_NN.fill(0);

  // find ILP neighbor list
  // TODO: TMD, group label
  ILP_neighbor<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, big_ilp_NN, big_ilp_NL, \
    type.data(), ilp_para, x, y, z, ilp_NN, \
    ilp_NL, group[1].label.data());
  CUDA_CHECK_KERNEL

  // initialize force of ilp neighbor temporary vector
  ilp_data.f12x_ilp_neigh.fill(0);
  ilp_data.f12y_ilp_neigh.fill(0);
  ilp_data.f12z_ilp_neigh.fill(0);
  ilp_data.f12x.fill(0);
  ilp_data.f12y.fill(0);
  ilp_data.f12z.fill(0);

  double *g_fx = force_per_atom.data();
  double *g_fy = force_per_atom.data() + number_of_atoms;
  double *g_fz = force_per_atom.data() + number_of_atoms * 2;
  double *g_virial = virial_per_atom.data();
  double *g_potential = potential_per_atom.data();
  double *g_f12x = ilp_data.f12x.data();
  double *g_f12y = ilp_data.f12y.data();
  double *g_f12z = ilp_data.f12z.data();
  double *g_f12x_ilp_neigh = ilp_data.f12x_ilp_neigh.data();
  double *g_f12y_ilp_neigh = ilp_data.f12y_ilp_neigh.data();
  double *g_f12z_ilp_neigh = ilp_data.f12z_ilp_neigh.data();

  // double _eatt = 0.0;
  // double _erep = 0.0;
  // CHECK(cudaMemcpyToSymbol(EATT, &_eatt, sizeof(double)));
  // CHECK(cudaMemcpyToSymbol(EREP, &_erep, sizeof(double)));
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
  CUDA_CHECK_KERNEL
  // CHECK(cudaMemcpyFromSymbol(&_eatt, EATT, sizeof(double)));
  // CHECK(cudaMemcpyFromSymbol(&_erep, EREP, sizeof(double)));
  // printf("##### EATT[%24.16f] EREP[%24.16f] #####\n", _eatt, _erep);

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
    CUDA_CHECK_KERNEL
}
