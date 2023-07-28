
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

#include "ilp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"

// TODO: best size here: 128
#define BLOCK_SIZE_FORCE 128

ILP::ILP(FILE* fid, int num_types, int num_atoms)
{
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP)) {
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
  double beta, alpha, delta, epsilon, C, d, sR, reff, C6, S, rcut;
  rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", \
      &beta, &alpha, &delta, &epsilon, &C, &d, &sR, &reff, &C6, &S, &rcut);
      PRINT_SCANF_ERROR(count, 11, "Reading error for ILP potential.");

      ilp_para.C[n][m] = C;
      ilp_para.C_6[n][m] = C6;
      ilp_para.d[n][m] = d;
      ilp_para.d_Seff[n][m] = d / sR / reff;
      ilp_para.epsilon[n][m] = epsilon;
      ilp_para.z0[n][m] = beta;
      ilp_para.lambda[n][m] = alpha / beta;
      ilp_para.delta2inv[n][m] = 1.0 / (delta * delta); //TODO: how faster?
      ilp_para.S[n][m] = S;
      ilp_para.r_cut[n][m] = rcut;
      // TODO: meV???
      double meV = 1e-3 * S;
      ilp_para.C[n][m] *= meV;
      ilp_para.C_6[n][m] *= meV;
      ilp_para.epsilon[n][m] *= meV;

      // TODO: ILP has taper function, check if necessary
      if (rc < rcut)
        rc = rcut;
    }
  }

  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * CUDA_MAX_NL);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);

  // init ilp neighbor list
  ilp_data.ilp_NN.resize(num_atoms);
  ilp_data.ilp_NL.resize(num_atoms * CUDA_MAX_NL);

  ilp_data.f12x.resize(num_atoms);
  ilp_data.f12y.resize(num_atoms);
  ilp_data.f12z.resize(num_atoms);

  // init constant cutoff coeff
  double h_tap_coeff[8] = \
    {1.0, 0.0, 0.0, 0.0, -35.0, 84.0, 70.0, 20.0};
  cudaMemcpyToSymbol(Tap_coeff, h_tap_coeff, 8 * sizeof(double));
  CUDA_CHECK_KERNEL
}

ILP::~ILP(void)
{
  // TODO
}

// TODO: set inline???
// calculate the long-range cutoff term
static __device__ double calc_Tap(const double r_ij, const double Rcut)
{
  double Tap, r;

  r = r_ij / Rcut;
  if (r >= 1.0) {
    Tap = 0.0;
  } else {
    Tap = Tap_coeff[7];
    for (int i = 6; i >= 0; --i) {
      Tap = Tap * r + Tap_coeff[i];
    }
  }

  return Tap;
}

// TODO: set inline???
// calculate the derivatives of long-range cutoff term
static __device__ double calc_dTap(const double r_ij, const double Rcut)
{
  double dTap, r;
  
  r = r_ij / Rcut;
  if (r >= Rcut) {
    dTap = 0.0;
  } else {
    dTap = 7.0 * Tap_coeff[7];
    for (int i = 6; i > 0; --i) {
      dTap = dTap * r + i * Tap_coeff[i];
    }
    dTap /= Rcut;
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
  ILP_Para ilp_para,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int *ilp_neighbor_number,
  int *ilp_neighbor_list,
  const int *group_label)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
    printf("***** ilp neighbor GPU *****\n");
    int count = 0;
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    printf("***** n1: %d, neigh: %d *****\n", n1, neighbor_number);
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      int type2 = g_type[n2];
      printf("***** n1: %d, n2: %d *****\n", n1, n2);
      printf("***** group lable[0]: %d, [1]: %d *****\n", group_label[0], group_label[1]);
      printf("***** %f *****\n", ilp_para.r_cut[0][0]);

      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12sq = x12 * x12 + y12 * y12 + z12 * z12;
      double r_cut = ilp_para.r_cut[type1][type2];

      printf("***** type1: %d, type2: %d *****\n", type1, type2);
      printf("***** size of r_cut: %d*****\n", sizeof(r_cut));
//      printf("***** size of r_cut[0]: %d*****\n", sizeof(*r_cut[0]));
//      printf("***** r_cut: %p *****\n", r_cut);
//      printf("***** r_cut[1]: %p *****\n", r_cut[1]);
//      printf("***** *r_cut: %p *****\n", *r_cut);
//      printf("***** **r_cut: %p *****\n", **r_cut);
//
      // TODO: store cutILPsq to calc fast
      double cutILPsq = r_cut * r_cut;

      if (group_label[n1] == group_label[n2] && d12sq < cutILPsq && d12sq != 0) {
        printf("----- count: %d -----\n", count);
        ilp_neighbor_list[count++ * number_of_particles + n1] = n2;
      }
    }
    printf("===== get here =====\n");
    ilp_neighbor_number[n1] = count;

    if (count > 3) {
      // TODO: error, there are too many neighbors for some atoms, 
      // please check your configuration
    }
  }
  // TODO: check group id before calc potential(calc in defferent layers)
}

// calculate the normals and its derivatives
static __device__ void calc_normal(
  double (&vet)[3][3],
  int cont,
  double (&normal)[3],
  double (&dnormdri)[3][3],
  double (&dnormal)[3][3][3])
{
  int id, ip, m;
  double nn2, nn;
  double pv12[3], pv31[3], pv23[3], n1[3], dni[3];
  double dnn[3][3], dpvdri[3][3];
  double dn1[3][3][3], dpv12[3][3][3], dpv23[3][3][3], dpv31[3][3][3];

  // initialize the arrays
  for (id = 0; id < 3; id++) {
    pv12[id] = 0.0;
    pv31[id] = 0.0;
    pv23[id] = 0.0;
    n1[id] = 0.0;
    dni[id] = 0.0;
    for (ip = 0; ip < 3; ip++) {
      dnn[ip][id] = 0.0;
      dpvdri[ip][id] = 0.0;
      for (m = 0; m < 3; m++) {
        dpv12[ip][id][m] = 0.0;
        dpv31[ip][id][m] = 0.0;
        dpv23[ip][id][m] = 0.0;
        dn1[ip][id][m] = 0.0;
      }
    }
  }

  if (cont <= 1) {
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0;
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[id][ip] = 0.0;
        for (m = 0; m < 3; ++m) {
          dnormal[id][ip][m] = 0.0;
        }
      }
    }
  } else if (cont == 2) {
    pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
    pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
    pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
    // derivatives of pv12[0] to ri
    dpvdri[0][0] = 0.0;
    dpvdri[0][1] = vet[0][2] - vet[1][2];
    dpvdri[0][2] = vet[1][1] - vet[0][1];
    // derivatives of pv12[1] to ri
    dpvdri[1][0] = vet[1][2] - vet[0][2];
    dpvdri[1][1] = 0.0;
    dpvdri[1][2] = vet[0][0] - vet[1][0];
    // derivatives of pv12[2] to ri
    dpvdri[2][0] = vet[0][1] - vet[1][1];
    dpvdri[2][1] = vet[1][0] - vet[0][0];
    dpvdri[2][2] = 0.0;

    dpv12[0][0][0] = 0.0;
    dpv12[0][1][0] = vet[1][2];
    dpv12[0][2][0] = -vet[1][1];
    dpv12[1][0][0] = -vet[1][2];
    dpv12[1][1][0] = 0.0;
    dpv12[1][2][0] = vet[1][0];
    dpv12[2][0][0] = vet[1][1];
    dpv12[2][1][0] = -vet[1][0];
    dpv12[2][2][0] = 0.0;

    // derivatives respect to the second neighbor, atom l
    dpv12[0][0][1] = 0.0;
    dpv12[0][1][1] = -vet[0][2];
    dpv12[0][2][1] = vet[0][1];
    dpv12[1][0][1] = vet[0][2];
    dpv12[1][1][1] = 0.0;
    dpv12[1][2][1] = -vet[0][0];
    dpv12[2][0][1] = -vet[0][1];
    dpv12[2][1][1] = vet[0][0];
    dpv12[2][2][1] = 0.0;

    // derivatives respect to the third neighbor, atom n
    // derivatives of pv12 to rn is zero
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0; }
    }

    n1[0] = pv12[0];
    n1[1] = pv12[1];
    n1[2] = pv12[2];
    // the magnitude of the normal vector
    nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    nn = sqrt(nn2);
    
    // TODO
    // if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
    // the unit normal vector
    normal[0] = n1[0] / nn;
    normal[1] = n1[1] / nn;
    normal[2] = n1[2] / nn;
    // derivatives of nn, dnn:3x1 vector
    dni[0] = (n1[0] * dpvdri[0][0] + n1[1] * dpvdri[1][0] + n1[2] * dpvdri[2][0]) / nn;
    dni[1] = (n1[0] * dpvdri[0][1] + n1[1] * dpvdri[1][1] + n1[2] * dpvdri[2][1]) / nn;
    dni[2] = (n1[0] * dpvdri[0][2] + n1[1] * dpvdri[1][2] + n1[2] * dpvdri[2][2]) / nn;
    // derivatives of unit vector ni respect to ri, the result is 3x3 matrix
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        dnormdri[id][ip] = dpvdri[id][ip] / nn - n1[id] * dni[ip] / nn2;
      }
    }
    // derivatives of non-normalized normal vector, dn1:3x3x3 array
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        for (m = 0; m < 3; m++) { dn1[id][ip][m] = dpv12[id][ip][m]; }
      }
    }
    // derivatives of nn, dnn:3x3 vector
    // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
    // r[id][m]: the id's component of atom m
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) / nn;
      }
    }
    // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
    // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormal[id][ip][m] = dn1[id][ip][m] / nn - n1[id] * dnn[ip][m] / nn2;
        }
      }
    }
    // TODO
  } else if (cont == 3) {
    pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
    pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
    pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
    // derivatives respect to the first neighbor, atom k
    dpv12[0][0][0] = 0.0;
    dpv12[0][1][0] = vet[1][2];
    dpv12[0][2][0] = -vet[1][1];
    dpv12[1][0][0] = -vet[1][2];
    dpv12[1][1][0] = 0.0;
    dpv12[1][2][0] = vet[1][0];
    dpv12[2][0][0] = vet[1][1];
    dpv12[2][1][0] = -vet[1][0];
    dpv12[2][2][0] = 0.0;
    // derivatives respect to the second neighbor, atom l
    dpv12[0][0][1] = 0.0;
    dpv12[0][1][1] = -vet[0][2];
    dpv12[0][2][1] = vet[0][1];
    dpv12[1][0][1] = vet[0][2];
    dpv12[1][1][1] = 0.0;
    dpv12[1][2][1] = -vet[0][0];
    dpv12[2][0][1] = -vet[0][1];
    dpv12[2][1][1] = vet[0][0];
    dpv12[2][2][1] = 0.0;

    // derivatives respect to the third neighbor, atom n
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0; }
    }

    pv31[0] = vet[2][1] * vet[0][2] - vet[0][1] * vet[2][2];
    pv31[1] = vet[2][2] * vet[0][0] - vet[0][2] * vet[2][0];
    pv31[2] = vet[2][0] * vet[0][1] - vet[0][0] * vet[2][1];
    // derivatives respect to the first neighbor, atom k
    dpv31[0][0][0] = 0.0;
    dpv31[0][1][0] = -vet[2][2];
    dpv31[0][2][0] = vet[2][1];
    dpv31[1][0][0] = vet[2][2];
    dpv31[1][1][0] = 0.0;
    dpv31[1][2][0] = -vet[2][0];
    dpv31[2][0][0] = -vet[2][1];
    dpv31[2][1][0] = vet[2][0];
    dpv31[2][2][0] = 0.0;
    // derivatives respect to the third neighbor, atom n
    dpv31[0][0][2] = 0.0;
    dpv31[0][1][2] = vet[0][2];
    dpv31[0][2][2] = -vet[0][1];
    dpv31[1][0][2] = -vet[0][2];
    dpv31[1][1][2] = 0.0;
    dpv31[1][2][2] = vet[0][0];
    dpv31[2][0][2] = vet[0][1];
    dpv31[2][1][2] = -vet[0][0];
    dpv31[2][2][2] = 0.0;
    // derivatives respect to the second neighbor, atom l
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv31[id][ip][1] = 0.0; }
    }

    pv23[0] = vet[1][1] * vet[2][2] - vet[2][1] * vet[1][2];
    pv23[1] = vet[1][2] * vet[2][0] - vet[2][2] * vet[1][0];
    pv23[2] = vet[1][0] * vet[2][1] - vet[2][0] * vet[1][1];
    // derivatives respect to the second neighbor, atom k
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv23[id][ip][0] = 0.0; }
    }
    // derivatives respect to the second neighbor, atom l
    dpv23[0][0][1] = 0.0;
    dpv23[0][1][1] = vet[2][2];
    dpv23[0][2][1] = -vet[2][1];
    dpv23[1][0][1] = -vet[2][2];
    dpv23[1][1][1] = 0.0;
    dpv23[1][2][1] = vet[2][0];
    dpv23[2][0][1] = vet[2][1];
    dpv23[2][1][1] = -vet[2][0];
    dpv23[2][2][1] = 0.0;
    // derivatives respect to the third neighbor, atom n
    dpv23[0][0][2] = 0.0;
    dpv23[0][1][2] = -vet[1][2];
    dpv23[0][2][2] = vet[1][1];
    dpv23[1][0][2] = vet[1][2];
    dpv23[1][1][2] = 0.0;
    dpv23[1][2][2] = -vet[1][0];
    dpv23[2][0][2] = -vet[1][1];
    dpv23[2][1][2] = vet[1][0];
    dpv23[2][2][2] = 0.0;

    //############################################################################################
    // average the normal vectors by using the 3 neighboring planes
    n1[0] = (pv12[0] + pv31[0] + pv23[0]) / cont;
    n1[1] = (pv12[1] + pv31[1] + pv23[1]) / cont;
    n1[2] = (pv12[2] + pv31[2] + pv23[2]) / cont;
    // the magnitude of the normal vector
    nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    nn = sqrt(nn2);
    // TODO
    // if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
    // the unit normal vector
    normal[0] = n1[0] / nn;
    normal[1] = n1[1] / nn;
    normal[2] = n1[2] / nn;

    // for the central atoms, dnormdri is always zero
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dnormdri[id][ip] = 0.0; }
    }

    // derivatives of non-normalized normal vector, dn1:3x3x3 array
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        for (m = 0; m < 3; m++) {
          dn1[id][ip][m] = (dpv12[id][ip][m] + dpv23[id][ip][m] + dpv31[id][ip][m]) / cont;
        }
      }
    }
    // derivatives of nn, dnn:3x3 vector
    // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
    // r[id][m]: the id's component of atom m
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) / nn;
      }
    }
    // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
    // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormal[id][ip][m] = dn1[id][ip][m] / nn - n1[id] * dnn[ip][m] / nn2;
        }
      }
    }// TODO
  } else {
    // TODO: too many neighbors for calculating normals
  }
  // TODO
}

// calculate the van der Waals force and energy
static __device__ void calc_vdW(
  double r,
  double d,
  double d_Seff,
  double C_6,
  double Tap,
  double dTap,
  double &p2_vdW,
  double &f2_vdW)
{
  double rsq, r2inv, r6inv, r8inv;
  double TSvdw, TSvdw2inv, Vilp;
  double fpair, fsum;

  rsq = r * r;
  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  TSvdw = 1.0 + exp(-d_Seff * r + d);
  TSvdw2inv = pow(TSvdw, -2.0);
  Vilp = -C_6 * r6inv / TSvdw;

  // derivatives
  fpair = -6.0 * C_6 * r8inv / TSvdw + \
    C_6 * d_Seff * (TSvdw - 1) * TSvdw2inv * r8inv * r;
  fsum = fpair * Tap - Vilp * dTap / r;

  p2_vdW = Tap * Vilp;
  f2_vdW = fsum;
  
}

// calculate the repulsive force and energy
static __device__ void calc_rep(
  double (&delxyz)[3],
  double r,
  double C,
  double lambda_,
  double delta2inv,
  double epsilon,
  double z0,
  double (&normal)[3],
  double (&dnormdri)[3][3],
  double (&dnormal)[3][3][3])
{
  double prodnorm1, rsq, rhosq1, rdsq1, exp0, exp1, frho1, Erep, Vilp;
  double fpair, fpair1, fsum, delx, dely, delz, fkcx, fkcy, fkcz;
  double dprodnorm1[3] = {0.0, 0.0, 0.0};
  double fp1[3] = {0.0, 0.0, 0.0};
  double fprod1[3] = {0.0, 0.0, 0.0};
  double delki[3] = {0.0, 0.0, 0.0};
  double fk[3] = {0.0, 0.0, 0.0};

  delx = delxyz[0];
  dely = delxyz[1];
  delz = delxyz[2];

  rsq = r * r;
  // calculate the transverse distance
  prodnorm1 = normal[0] * delx + normal[1] * dely + normal[2] * delz;
  rhosq1 = rsq - prodnorm1 * prodnorm1;
  rdsq1 = rhosq1 * delta2inv;

  // store exponents
  exp0 = exp(-lambda_ * (r - z0));
  exp1 = exp(-rdsq1);

  frho1 = exp1 * C;
  Erep = 0.5 * epsilon + frho1;
  Vilp = exp0 * Erep;
  // TODO

  // derivatives
  fpair = lambda_ * exp0 / r * Erep;
  fpair1 = 2.0 * exp0 * frho1 * delta2inv;
  fsum = fpair + fpair1;

  // derivatives of the product of rij and ni, the resutl is a vector
  dprodnorm1[0] = 
    dnormdri[0][0] * delx + dnormdri[1][0] * dely + dnormdri[2][0] * delz;
  dprodnorm1[1] = 
    dnormdri[0][1] * delx + dnormdri[1][1] * dely + dnormdri[2][1] * delz;
  dprodnorm1[2] = 
    dnormdri[0][2] * delx + dnormdri[1][2] * dely + dnormdri[2][2] * delz;
  fp1[0] = prodnorm1 * normal[0] * fpair1;
  fp1[1] = prodnorm1 * normal[1] * fpair1;
  fp1[2] = prodnorm1 * normal[2] * fpair1;
  fprod1[0] = prodnorm1 * dprodnorm1[0] * fpair1;
  fprod1[1] = prodnorm1 * dprodnorm1[1] * fpair1;
  fprod1[2] = prodnorm1 * dprodnorm1[2] * fpair1;

  // fkcx = (delx * fsum - fp1[0]) * Tap - Vilp * dTap * delx / r;
  // fkcy = (dely * fsum - fp1[1]) * Tap - Vilp * dTap * dely / r;
  // fkcz = (delz * fsum - fp1[2]) * Tap - Vilp * dTap * delz / r;
  // TODO
  
}

// force evaluation kernel
static __global__ void gpu_find_force(
  ILP_Para ilp_para,
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
  double *g_potential)
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
  // double r2inv, r6inv, r8inv;

  if (n1 < N2) {
    printf("********* ILP find force GPU **********\n");
    double x12, y12, z12;
    int neighor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    // calculate the normal
    // TODO: loop the ILP_neigh to create the vet and cont
    int cont = 0;
    // TODO: how to initialize normals
    double normal[3];
    double dnormdri[3][3];
    double dnormal[3][3][3];

    double vet[3][3];
    int id, ip, m;
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0;
      for (ip = 0; ip < 3; ++ip) {
        vet[id][ip] = 0.0;
        dnormdri[id][ip] = 0.0;
        for (m = 0; m < 3; ++m) {
          dnormal[id][ip][m] = 0.0;
        }
      }
    }

    int ilp_neighbor_number = g_ilp_neighbor_number[n1];
    for (int i1 = 0; i1 < ilp_neighbor_number; ++i1) {
      int n2_ilp = g_ilp_neighbor_list[n1 + number_of_particles * i1];
      x12 = g_x[n2_ilp] - x1;
      y12 = g_y[n2_ilp] - y1;
      z12 = g_z[n2_ilp] - z1;
      vet[cont][0] = x12;
      vet[cont][1] = y12;
      vet[cont][2] = z12;
      ++cont;
    }

    printf("********* ILP calc normal **********\n");
    calc_normal(vet, cont, normal, dnormdri, dnormal);

    // calculate energy and force
    for (int i1 = 0; i1 < neighor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      int type2 = g_type[n2];

      x12 = g_x[n2] - x1;
      y12 = g_y[n2] - y1;
      z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);

      // calculate distance between atoms
      rsq = x12 * x12 + y12 * y12 + z12 * z12;
      r = sqrt(rsq);
      Rcut = ilp_para.r_cut[type1][type2];
      // TODO: not in the same layer
      if (r >= Rcut || group_label[n1] == group_label[n2]) {
        continue;
      }

      printf("********* ILP calc Tap **********\n");
      double Tap, dTap;
      Tap = calc_Tap(r, Rcut);
      dTap = calc_dTap(r, Rcut);

      printf("********* ILP calc vdW **********\n");
      double p2_vdW, f2_vdW;
      calc_vdW(
        r,
        ilp_para.d[type1][type2],
        ilp_para.d_Seff[type1][type2],
        ilp_para.C_6[type1][type2],
        Tap,
        dTap,
        p2_vdW,
        f2_vdW);
      
      double f12x = f2_vdW * x12 * 0.5;
      double f12y = f2_vdW * y12 * 0.5;
      double f12z = f2_vdW * z12 * 0.5;
      double f21x = -f12x;
      double f21y = -f12y;
      double f21z = -f12z;

      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      s_pe += p2_vdW * 0.5;
      s_sxx += x12 * f21x;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syy += y12 * f21y;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
      s_szz += z12 * f21z;

      
      printf("********* ILP calc rep **********\n");
      double delxyz[3] = {x12, y12, z12};
      // calc_rep(
      //   delxyz,
      //   r,
      //   ilp_para->C[type1][type2],
      //   ilp_para->lambda[type1][type2],
      //   ilp_para->delta2inv[type1][type2],
      //   ilp_para->epsilon[type1][type2],
      //   ilp_para->z0[type1][type2],
      //   normal,
      //   dnormdri,
      //   dnormal);

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
      double delki[3] = {0.0, 0.0, 0.0};
      double fk[3] = {0.0, 0.0, 0.0};

      delx = delxyz[0];
      dely = delxyz[1];
      delz = delxyz[2];

      // rsq = r * r;
      // calculate the transverse distance
      prodnorm1 = normal[0] * delx + normal[1] * dely + normal[2] * delz;
      rhosq1 = rsq - prodnorm1 * prodnorm1;
      rdsq1 = rhosq1 * delta2inv;

      // store exponents
      exp0 = exp(-lambda_ * (r - z0));
      exp1 = exp(-rdsq1);

      frho1 = exp1 * C;
      Erep = 0.5 * epsilon + frho1;
      Vilp = exp0 * Erep;
      // TODO

      // derivatives
      fpair = lambda_ * exp0 / r * Erep;
      fpair1 = 2.0 * exp0 * frho1 * delta2inv;
      fsum = fpair + fpair1;

      // derivatives of the product of rij and ni, the resutl is a vector
      dprodnorm1[0] = 
        dnormdri[0][0] * delx + dnormdri[1][0] * dely + dnormdri[2][0] * delz;
      dprodnorm1[1] = 
        dnormdri[0][1] * delx + dnormdri[1][1] * dely + dnormdri[2][1] * delz;
      dprodnorm1[2] = 
        dnormdri[0][2] * delx + dnormdri[1][2] * dely + dnormdri[2][2] * delz;
      fp1[0] = prodnorm1 * normal[0] * fpair1;
      fp1[1] = prodnorm1 * normal[1] * fpair1;
      fp1[2] = prodnorm1 * normal[2] * fpair1;
      fprod1[0] = prodnorm1 * dprodnorm1[0] * fpair1;
      fprod1[1] = prodnorm1 * dprodnorm1[1] * fpair1;
      fprod1[2] = prodnorm1 * dprodnorm1[2] * fpair1;

      fkcx = (delx * fsum - fp1[0]) * Tap - Vilp * dTap * delx / r;
      fkcy = (dely * fsum - fp1[1]) * Tap - Vilp * dTap * dely / r;
      fkcz = (delz * fsum - fp1[2]) * Tap - Vilp * dTap * delz / r;

      s_fx += fkcx - fprod1[0] * Tap;
      s_fy += fkcy - fprod1[1] * Tap;
      s_fz += fkcz - fprod1[2] * Tap;

      // TODO: write data of other atoms, need atomic operation???
      g_fx[n2] -= fkcx;
      g_fy[n2] -= fkcy;
      g_fz[n2] -= fkcz;

      for (int kk = 0; kk < ilp_neighbor_number; ++kk) {
        int n2_ilp = g_ilp_neighbor_list[n1 + number_of_particles * kk];
        if (n2_ilp == n1) continue;
        // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
        dprodnorm1[0] = dnormal[0][0][kk] * delx + dnormal[1][0][kk] * dely +
            dnormal[2][0][kk] * delz;
        dprodnorm1[1] = dnormal[0][1][kk] * delx + dnormal[1][1][kk] * dely +
            dnormal[2][1][kk] * delz;
        dprodnorm1[2] = dnormal[0][2][kk] * delx + dnormal[1][2][kk] * dely +
            dnormal[2][2][kk] * delz;
        fk[0] = (-prodnorm1 * dprodnorm1[0] * fpair1) * Tap;
        fk[1] = (-prodnorm1 * dprodnorm1[1] * fpair1) * Tap;
        fk[2] = (-prodnorm1 * dprodnorm1[2] * fpair1) * Tap;

        // TODO: write data of other atoms, need atomic operation???
        g_fx[n2_ilp] += fk[0];
        g_fy[n2_ilp] += fk[1];
        g_fz[n2_ilp] += fk[2];

        delki[0] = g_x[n2_ilp] - x1;
        delki[1] = g_y[n2_ilp] - y1;
        delki[2] = g_z[n2_ilp] - z1;

      }
      s_pe += Tap * Vilp;

    }

    // TODO

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

void ILP::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom)
{
  // TODO
}
// find force and related quantities
void ILP::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom,
  std::vector<Group> &group)
{
  printf("********* ILP compute **********\n");
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

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
      box,
      type,
      position_per_atom,
      ilp_data.cell_count,
      ilp_data.cell_count_sum,
      ilp_data.cell_contents,
      ilp_data.NN,
      ilp_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + number_of_atoms;
  const double* z = position_per_atom.data() + number_of_atoms * 2;
  const int *NN = ilp_data.NN.data();
  const int *NL = ilp_data.NL.data();
  int *ilp_NL = ilp_data.ilp_NL.data();
  int *ilp_NN = ilp_data.ilp_NN.data();

  // find ILP neighbor list
  // TODO: __global__ ???
  // TODO: assume the first group column is for ILP
  printf("********* ILP neigh **********\n");
  const int *group_label = group[0].label.data();
  ILP_neighbor<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, NN, NL, \
    type.data(), ilp_para, x, y, z, ilp_NN, \
    ilp_NL, group_label);
  CUDA_CHECK_KERNEL

  printf("********* ILP find force **********\n");
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
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    virial_per_atom.data(),
    potential_per_atom.data());
  // TODO
  CUDA_CHECK_KERNEL
}
