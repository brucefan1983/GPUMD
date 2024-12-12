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

#include "ilp_nep_gr_hbn.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define BLOCK_SIZE_FORCE 128

// there are most 3 intra-layer neighbors for graphene and h-BN
#define NNEI 3

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};


ILP_NEP_GR_HBN::ILP_NEP_GR_HBN(FILE* fid_ilp, const char* file_nep, int num_types, int num_atoms)
{
  // read ILP TMD potential parameter
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP_GR_HBN)) {
    PRINT_INPUT_ERROR("Incorrect type number of ILP_NEP_GR_HBN parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid_ilp, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for ILP_NEP_GR_HBN potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  // read ILP group method
  PRINT_SCANF_ERROR(fscanf(fid_ilp, "%d", &ilp_group_method), 1, 
  "Reading error for ILP group method.");
  printf("Use group method %d to identify molecule for ILP.\n", ilp_group_method);

  // read parameters
  float beta, alpha, delta, epsilon, C, d, sR;
  float reff, C6, S, rcut_ilp, rcut_global;
  ilp_rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid_ilp, "%f%f%f%f%f%f%f%f%f%f%f%f", \
      &beta, &alpha, &delta, &epsilon, &C, &d, &sR, &reff, &C6, &S, \
      &rcut_ilp, &rcut_global);
      PRINT_SCANF_ERROR(count, 12, "Reading error for ILP_NEP_GR_HBN potential.");

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

      if (rc_ilp < rcut_global)
        rc_ilp = rcut_global;
    }
  }


  // initialize neighbor lists and some temp vectors
  int max_neighbor_number = min(num_atoms, CUDA_MAX_NL_GR_HBN);
  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * max_neighbor_number);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);

  // init ilp neighbor list
  ilp_data.ilp_NN.resize(num_atoms);
  ilp_data.ilp_NL.resize(num_atoms * MAX_ILP_NEIGHBOR_GR_HBN);
  ilp_data.reduce_NL.resize(num_atoms * max_neighbor_number);
  ilp_data.big_ilp_NN.resize(num_atoms);
  ilp_data.big_ilp_NL.resize(num_atoms * MAX_BIG_ILP_NEIGHBOR_GR_HBN);

  ilp_data.f12x.resize(num_atoms * max_neighbor_number);
  ilp_data.f12y.resize(num_atoms * max_neighbor_number);
  ilp_data.f12z.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_GR_HBN);
  ilp_data.f12y_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_GR_HBN);
  ilp_data.f12z_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_GR_HBN);


  // init constant cutoff coeff
  float h_tap_coeff[8] = \
    {1.0f, 0.0f, 0.0f, 0.0f, -35.0f, 84.0f, -70.0f, 20.0f};
  CHECK(gpuMemcpyToSymbol(Tap_coeff_gr_hbn, h_tap_coeff, 8 * sizeof(float)));

  // set ilp_flag to 1
  ilp_flag = 1;


  std::ifstream input(file_nep);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_nep << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep3") {
    paramb.version = 3;
  } else if (tokens[0] == "nep4") {
    paramb.version = 4;
  } else if (tokens[0] == "nep5") {
    paramb.version = 5;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP3 and NEP4 models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("Use the NEP%d potential with %d atom type.\n", paramb.version, paramb.num_types);
  } else {
    printf("Use the NEP%d potential with %d atom types.\n", paramb.version, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    paramb.atomic_numbers[n] = atomic_number - 1;
    printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), atomic_number);
  }


  // cutoff 4.2 3.7 80 47 1
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != 8) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                 "[radial_factor] [angular_factor] [zbl_factor].\n";
    exit(1);
  }
  paramb.rc_radial = get_float_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_float_from_token(tokens[2], __FILE__, __LINE__);
  printf("    radial cutoff = %g A.\n", paramb.rc_radial);
  printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  printf("    MN_radial = %d.\n", MN_radial);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  if (tokens.size() == 8) {
    paramb.typewise_cutoff_radial_factor = get_float_from_token(tokens[5], __FILE__, __LINE__);
    paramb.typewise_cutoff_angular_factor = get_float_from_token(tokens[6], __FILE__, __LINE__);
    paramb.typewise_cutoff_zbl_factor = get_float_from_token(tokens[7], __FILE__, __LINE__);
    if (paramb.typewise_cutoff_radial_factor > 0.0f) {
      paramb.use_typewise_cutoff = true;
    }
    if (paramb.typewise_cutoff_zbl_factor > 0.0f) {
      paramb.use_typewise_cutoff_zbl = true;
    }
  }
#ifdef USE_TABLE
  if (paramb.use_typewise_cutoff) {
    PRINT_INPUT_ERROR("Cannot use tabulated radial functions with typewise cutoff.");
  }
#endif

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() != 4) {
    std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  printf("    l_max_4body = %d.\n", L_max_4body);
  printf("    l_max_5body = %d.\n", L_max_5body);
  if (L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  nep_model_type = paramb.model_type;
  if (paramb.model_type == 3) {
    annmb.dim += 1;
  }
  printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  if (paramb.version == 3) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else {
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }
  if (paramb.model_type == 2) {
    // Polarizability models have twice as many parameters
    annmb.num_para_ann *= 2;
  }
  printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;
  printf("    total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data.parameters.resize(annmb.num_para);
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }


  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(num_atoms * annmb.dim);
  nep_data.sum_fxyz.resize(num_atoms * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  nep_data.cell_count.resize(num_atoms);
  nep_data.cell_count_sum.resize(num_atoms);
  nep_data.cell_contents.resize(num_atoms);
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);

#ifdef USE_TABLE
  construct_table(parameters.data());
  printf("    use tabulated radial functions to speed up.\n");
#endif

}

ILP_NEP_GR_HBN::~ILP_NEP_GR_HBN(void)
{
  // nothing
}

static __device__ __forceinline__ float calc_Tap(const float r_ij, const float Rcutinv)
{
  float Tap, r;

  r = r_ij * Rcutinv;
  if (r >= 1.0f) {
    Tap = 0.0f;
  } else {
    Tap = Tap_coeff_gr_hbn[7];
    for (int i = 6; i >= 0; --i) {
      Tap = Tap * r + Tap_coeff_gr_hbn[i];
    }
  }

  return Tap;
}

// calculate the derivatives of long-range cutoff term
static __device__ __forceinline__ float calc_dTap(const float r_ij, const float Rcut, const float Rcutinv)
{
  float dTap, r;
  
  r = r_ij * Rcutinv;
  if (r >= 1.0f) {
    dTap = 0.0f;
  } else {
    dTap = 7.0f * Tap_coeff_gr_hbn[7];
    for (int i = 6; i > 0; --i) {
      dTap = dTap * r + i * Tap_coeff_gr_hbn[i];
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
  ILP_GR_HBN_Para ilp_para,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int *ilp_neighbor_number,
  int *ilp_neighbor_list,
  const int *group_label)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
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


      if (group_label[n1] == group_label[n2] && d12sq < rcutsq && d12sq != 0) {
        ilp_neighbor_list[count++ * number_of_particles + n1] = n2;
      }
    }
    ilp_neighbor_number[n1] = count;

    if (count > MAX_ILP_NEIGHBOR_GR_HBN) {
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

// calculate the normals and its derivatives
static __device__ void calc_normal(
  float (&vet)[3][3],
  int cont,
  float (&normal)[3],
  float (&dnormdri)[3][3],
  float (&dnormal)[3][3][3])
{
  int id, ip, m;
  float pv12[3], pv31[3], pv23[3], n1[3], dni[3];
  float dnn[3][3], dpvdri[3][3];
  float dn1[3][3][3], dpv12[3][3][3], dpv23[3][3][3], dpv31[3][3][3];

  float nninv, continv;

  // initialize the arrays
  for (id = 0; id < 3; id++) {
    pv12[id] = 0.0f;
    pv31[id] = 0.0f;
    pv23[id] = 0.0f;
    n1[id] = 0.0f;
    dni[id] = 0.0f;
    for (ip = 0; ip < 3; ip++) {
      dnn[ip][id] = 0.0f;
      dpvdri[ip][id] = 0.0f;
      for (m = 0; m < 3; m++) {
        dpv12[ip][id][m] = 0.0f;
        dpv31[ip][id][m] = 0.0f;
        dpv23[ip][id][m] = 0.0f;
        dn1[ip][id][m] = 0.0f;
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
    dpvdri[0][0] = 0.0f;
    dpvdri[0][1] = vet[0][2] - vet[1][2];
    dpvdri[0][2] = vet[1][1] - vet[0][1];
    // derivatives of pv12[1] to ri
    dpvdri[1][0] = vet[1][2] - vet[0][2];
    dpvdri[1][1] = 0.0f;
    dpvdri[1][2] = vet[0][0] - vet[1][0];
    // derivatives of pv12[2] to ri
    dpvdri[2][0] = vet[0][1] - vet[1][1];
    dpvdri[2][1] = vet[1][0] - vet[0][0];
    dpvdri[2][2] = 0.0f;

    dpv12[0][0][0] = 0.0f;
    dpv12[0][1][0] = vet[1][2];
    dpv12[0][2][0] = -vet[1][1];
    dpv12[1][0][0] = -vet[1][2];
    dpv12[1][1][0] = 0.0f;
    dpv12[1][2][0] = vet[1][0];
    dpv12[2][0][0] = vet[1][1];
    dpv12[2][1][0] = -vet[1][0];
    dpv12[2][2][0] = 0.0f;

    // derivatives respect to the second neighbor, atom l
    dpv12[0][0][1] = 0.0f;
    dpv12[0][1][1] = -vet[0][2];
    dpv12[0][2][1] = vet[0][1];
    dpv12[1][0][1] = vet[0][2];
    dpv12[1][1][1] = 0.0f;
    dpv12[1][2][1] = -vet[0][0];
    dpv12[2][0][1] = -vet[0][1];
    dpv12[2][1][1] = vet[0][0];
    dpv12[2][2][1] = 0.0f;

    // derivatives respect to the third neighbor, atom n
    // derivatives of pv12 to rn is zero
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0f; }
    }

    n1[0] = pv12[0];
    n1[1] = pv12[1];
    n1[2] = pv12[2];
    // the magnitude of the normal vector
    // nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    // nn = sqrt(nn2);
    // nninv = 1.0 / nn;
    nninv = rnorm3df(n1[0], n1[1], n1[2]);
    
    // TODO
    // if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
    // the unit normal vector
    normal[0] = n1[0] * nninv;
    normal[1] = n1[1] * nninv;
    normal[2] = n1[2] * nninv;
    // derivatives of nn, dnn:3x1 vector
    dni[0] = (n1[0] * dpvdri[0][0] + n1[1] * dpvdri[1][0] + n1[2] * dpvdri[2][0]) * nninv;
    dni[1] = (n1[0] * dpvdri[0][1] + n1[1] * dpvdri[1][1] + n1[2] * dpvdri[2][1]) * nninv;
    dni[2] = (n1[0] * dpvdri[0][2] + n1[1] * dpvdri[1][2] + n1[2] * dpvdri[2][2]) * nninv;
    // derivatives of unit vector ni respect to ri, the result is 3x3 matrix
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        dnormdri[id][ip] = dpvdri[id][ip] * nninv - n1[id] * dni[ip] * nninv * nninv;
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
        dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) * nninv;
      }
    }
    // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
    // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormal[id][ip][m] = dn1[id][ip][m] * nninv - n1[id] * dnn[ip][m] * nninv * nninv;
        }
      }
    }
  } else if (cont == 3) {
    continv = 1.0 / cont;

    pv12[0] = vet[0][1] * vet[1][2] - vet[1][1] * vet[0][2];
    pv12[1] = vet[0][2] * vet[1][0] - vet[1][2] * vet[0][0];
    pv12[2] = vet[0][0] * vet[1][1] - vet[1][0] * vet[0][1];
    // derivatives respect to the first neighbor, atom k
    dpv12[0][0][0] = 0.0f;
    dpv12[0][1][0] = vet[1][2];
    dpv12[0][2][0] = -vet[1][1];
    dpv12[1][0][0] = -vet[1][2];
    dpv12[1][1][0] = 0.0f;
    dpv12[1][2][0] = vet[1][0];
    dpv12[2][0][0] = vet[1][1];
    dpv12[2][1][0] = -vet[1][0];
    dpv12[2][2][0] = 0.0f;
    // derivatives respect to the second neighbor, atom l
    dpv12[0][0][1] = 0.0f;
    dpv12[0][1][1] = -vet[0][2];
    dpv12[0][2][1] = vet[0][1];
    dpv12[1][0][1] = vet[0][2];
    dpv12[1][1][1] = 0.0f;
    dpv12[1][2][1] = -vet[0][0];
    dpv12[2][0][1] = -vet[0][1];
    dpv12[2][1][1] = vet[0][0];
    dpv12[2][2][1] = 0.0f;

    // derivatives respect to the third neighbor, atom n
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv12[id][ip][2] = 0.0f; }
    }

    pv31[0] = vet[2][1] * vet[0][2] - vet[0][1] * vet[2][2];
    pv31[1] = vet[2][2] * vet[0][0] - vet[0][2] * vet[2][0];
    pv31[2] = vet[2][0] * vet[0][1] - vet[0][0] * vet[2][1];
    // derivatives respect to the first neighbor, atom k
    dpv31[0][0][0] = 0.0f;
    dpv31[0][1][0] = -vet[2][2];
    dpv31[0][2][0] = vet[2][1];
    dpv31[1][0][0] = vet[2][2];
    dpv31[1][1][0] = 0.0f;
    dpv31[1][2][0] = -vet[2][0];
    dpv31[2][0][0] = -vet[2][1];
    dpv31[2][1][0] = vet[2][0];
    dpv31[2][2][0] = 0.0f;
    // derivatives respect to the third neighbor, atom n
    dpv31[0][0][2] = 0.0f;
    dpv31[0][1][2] = vet[0][2];
    dpv31[0][2][2] = -vet[0][1];
    dpv31[1][0][2] = -vet[0][2];
    dpv31[1][1][2] = 0.0f;
    dpv31[1][2][2] = vet[0][0];
    dpv31[2][0][2] = vet[0][1];
    dpv31[2][1][2] = -vet[0][0];
    dpv31[2][2][2] = 0.0f;
    // derivatives respect to the second neighbor, atom l
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv31[id][ip][1] = 0.0f; }
    }

    pv23[0] = vet[1][1] * vet[2][2] - vet[2][1] * vet[1][2];
    pv23[1] = vet[1][2] * vet[2][0] - vet[2][2] * vet[1][0];
    pv23[2] = vet[1][0] * vet[2][1] - vet[2][0] * vet[1][1];
    // derivatives respect to the second neighbor, atom k
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dpv23[id][ip][0] = 0.0f; }
    }
    // derivatives respect to the second neighbor, atom l
    dpv23[0][0][1] = 0.0f;
    dpv23[0][1][1] = vet[2][2];
    dpv23[0][2][1] = -vet[2][1];
    dpv23[1][0][1] = -vet[2][2];
    dpv23[1][1][1] = 0.0f;
    dpv23[1][2][1] = vet[2][0];
    dpv23[2][0][1] = vet[2][1];
    dpv23[2][1][1] = -vet[2][0];
    dpv23[2][2][1] = 0.0f;
    // derivatives respect to the third neighbor, atom n
    dpv23[0][0][2] = 0.0f;
    dpv23[0][1][2] = -vet[1][2];
    dpv23[0][2][2] = vet[1][1];
    dpv23[1][0][2] = vet[1][2];
    dpv23[1][1][2] = 0.0f;
    dpv23[1][2][2] = -vet[1][0];
    dpv23[2][0][2] = -vet[1][1];
    dpv23[2][1][2] = vet[1][0];
    dpv23[2][2][2] = 0.0f;

    // average the normal vectors by using the 3 neighboring planes
    n1[0] = (pv12[0] + pv31[0] + pv23[0]) * continv;
    n1[1] = (pv12[1] + pv31[1] + pv23[1]) * continv;
    n1[2] = (pv12[2] + pv31[2] + pv23[2]) * continv;
    // the magnitude of the normal vector
    // nn2 = n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2];
    // nn = sqrt(nn2);

    // nninv = 1.0 / nn;
    nninv = rnorm3df(n1[0], n1[1], n1[2]);
    // TODO
    // if (nn == 0) error->one(FLERR, "The magnitude of the normal vector is zero");
    // the unit normal vector
    normal[0] = n1[0] * nninv;
    normal[1] = n1[1] * nninv;
    normal[2] = n1[2] * nninv;

    // for the central atoms, dnormdri is always zero
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) { dnormdri[id][ip] = 0.0f; }
    }

    // derivatives of non-normalized normal vector, dn1:3x3x3 array
    for (id = 0; id < 3; id++) {
      for (ip = 0; ip < 3; ip++) {
        for (m = 0; m < 3; m++) {
          dn1[id][ip][m] = (dpv12[id][ip][m] + dpv23[id][ip][m] + dpv31[id][ip][m]) * continv;
        }
      }
    }
    // derivatives of nn, dnn:3x3 vector
    // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
    // r[id][m]: the id's component of atom m
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        dnn[id][m] = (n1[0] * dn1[0][id][m] + n1[1] * dn1[1][id][m] + n1[2] * dn1[2][id][m]) * nninv;
      }
    }
    // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
    // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
    for (m = 0; m < 3; m++) {
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormal[id][ip][m] = dn1[id][ip][m] * nninv - n1[id] * dnn[ip][m] * nninv * nninv;
        }
      }
    }
  } else {
    // TODO: error! too many neighbors for calculating normals
  }
}

// calculate the van der Waals force and energy
inline static __device__ void calc_vdW(
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

  // TODO: use float
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
  ILP_GR_HBN_Para ilp_para,
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

    int index_ilp_vec[3] = {n1, n1 + number_of_particles, n1 + (number_of_particles << 1)};
    float fk_temp[9] = {0.0f};

    float delkix_half[3] = {0.0f, 0.0f, 0.0f};
    float delkiy_half[3] = {0.0f, 0.0f, 0.0f};
    float delkiz_half[3] = {0.0f, 0.0f, 0.0f};

    // calculate the normal
    int cont = 0;
    float normal[3];
    float dnormdri[3][3];
    float dnormal[3][3][3];

    float vet[3][3];
    int id, ip, m;
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0f;
      for (ip = 0; ip < 3; ++ip) {
        vet[id][ip] = 0.0f;
        dnormdri[id][ip] = 0.0f;
        for (m = 0; m < 3; ++m) {
          dnormal[id][ip][m] = 0.0f;
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
    for (int i1 = 0; i1 < neighor_number; ++i1) {
      int index = n1 + number_of_particles * i1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2];

      x12d = g_x[n2] - x1;
      y12d = g_y[n2] - y1;
      z12d = g_z[n2] - z1;
      apply_mic(box, x12d, y12d, z12d);

      // save x12, y12, z12 in float
      x12f = float(x12d);
      y12f = float(y12d);
      z12f = float(z12d);

      // calculate distance between atoms
      rsq = x12f * x12f + y12f * y12f + z12f * z12f;
      r = sqrtf(rsq);
      Rcut = ilp_para.rcut_global[type1][type2];
      // not in the same layer

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
      // TODO: use float
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
        // int index_ilp = n1 + number_of_particles * kk;
        // int n2_ilp = g_ilp_neighbor_list[index_ilp];
        // if (n2_ilp_vec[kk] == n1) continue;
        // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
        dprodnorm1[0] = dnormal[0][0][kk] * delx + dnormal[1][0][kk] * dely +
            dnormal[2][0][kk] * delz;
        dprodnorm1[1] = dnormal[0][1][kk] * delx + dnormal[1][1][kk] * dely +
            dnormal[2][1][kk] * delz;
        dprodnorm1[2] = dnormal[0][2][kk] * delx + dnormal[1][2][kk] * dely +
            dnormal[2][2][kk] * delz;
        // fk[0] = (-prodnorm1 * dprodnorm1[0] * fpair1) * Tap;
        // fk[1] = (-prodnorm1 * dprodnorm1[1] * fpair1) * Tap;
        // fk[2] = (-prodnorm1 * dprodnorm1[2] * fpair1) * Tap;
        fk[0] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[0];
        fk[1] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[1];
        fk[2] = minus_prodnorm1_m_fpair1_m_Tap * dprodnorm1[2];

        // g_f12x_ilp_neigh[index_ilp_vec[kk]] += fk[0];
        // g_f12y_ilp_neigh[index_ilp_vec[kk]] += fk[1];
        // g_f12z_ilp_neigh[index_ilp_vec[kk]] += fk[2];
        fk_temp[kk] += fk[0];
        fk_temp[kk + 3] += fk[1];
        fk_temp[kk + 6] += fk[2];

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
    g_f12x_ilp_neigh[index_ilp_vec[0]] = fk_temp[0];
    g_f12x_ilp_neigh[index_ilp_vec[1]] = fk_temp[1];
    g_f12x_ilp_neigh[index_ilp_vec[2]] = fk_temp[2];
    g_f12y_ilp_neigh[index_ilp_vec[0]] = fk_temp[3];
    g_f12y_ilp_neigh[index_ilp_vec[1]] = fk_temp[4];
    g_f12y_ilp_neigh[index_ilp_vec[2]] = fk_temp[5];
    g_f12z_ilp_neigh[index_ilp_vec[0]] = fk_temp[6];
    g_f12z_ilp_neigh[index_ilp_vec[1]] = fk_temp[7];
    g_f12z_ilp_neigh[index_ilp_vec[2]] = fk_temp[8];

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
      s_sxx += -x12f * f21x * 0.5;
      s_sxy += -x12f * f21y * 0.5;
      s_sxz += -x12f * f21z * 0.5;
      s_syx += -y12f * f21x * 0.5;
      s_syy += -y12f * f21y * 0.5;
      s_syz += -y12f * f21z * 0.5;
      s_szx += -z12f * f21x * 0.5;
      s_szy += -z12f * f21y * 0.5;
      s_szz += -z12f * f21z * 0.5;
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


void ILP_NEP_GR_HBN::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }
  ann.b1 = pointer;
  pointer += 1;


  ann.c = pointer;
}

static __global__ void find_neighbor_list_nep(
  ILP_NEP_GR_HBN::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];
  int count_radial = 0;
  int count_angular = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f * paramb.rcinv_radial,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
        if (cell_id_x + xx < 0)
          neighbor_cell += nx;
        if (cell_id_x + xx >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + yy < 0)
          neighbor_cell += ny * nx;
        if (cell_id_y + yy >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + zz < 0)
          neighbor_cell += nz * ny * nx;
        if (cell_id_z + zz >= nz)
          neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n2 < N1 || n2 >= N2 || n1 == n2) {
            continue;
          }

          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
          float d12_square = x12 * x12 + y12 * y12 + z12 * z12;

          int t2 = g_type[n2];
          float rc_radial = paramb.rc_radial;
          float rc_angular = paramb.rc_angular;
          if (paramb.use_typewise_cutoff) {
            int z1 = paramb.atomic_numbers[t1];
            int z2 = paramb.atomic_numbers[t2];
            rc_radial = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor,
              rc_radial);
            rc_angular = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor,
              rc_angular);
          }

          if (d12_square >= rc_radial * rc_radial) {
            continue;
          }

          g_NL_radial[count_radial++ * N + n1] = n2;

          if (d12_square < rc_angular * rc_angular) {
            g_NL_angular[count_angular++ * N + n1] = n2;
          }
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  ILP_NEP_GR_HBN::ParaMB paramb,
  ILP_NEP_GR_HBN::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
#ifdef USE_TABLE
  const float* __restrict__ g_gn_radial,
  const float* __restrict__ g_gn_angular,
#endif
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
#endif
    }

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
#ifdef USE_TABLE
        int index_left, index_right;
        float weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * paramb.num_types + g_type[n2];
        float gn12 =
          g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
#else
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
#endif
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};


    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp);
    } else {
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
    }
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  ILP_NEP_GR_HBN::ParaMB paramb,
  ILP_NEP_GR_HBN::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
#ifdef USE_TABLE
  const float* __restrict__ g_gnp_radial,
#endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
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
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      int t21 = t2 * paramb.num_types + t1;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 =
          g_gnp_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        float gnp21 =
          g_gnp_radial[(index_left * paramb.num_types_sq + t21) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * paramb.num_types_sq + t21) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
#else
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
          gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
#endif
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];

      s_sxx += r12[0] * f21[0];
      s_syy += r12[1] * f21[1];
      s_szz += r12[2] * f21[2];
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
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

static __global__ void find_partial_force_angular(
  ILP_NEP_GR_HBN::ParaMB paramb,
  ILP_NEP_GR_HBN::ANN annmb,
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
#ifdef USE_TABLE
  const float* __restrict__ g_gn_angular,
  const float* __restrict__ g_gnp_angular,
#endif
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
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
      float f12[3] = {0.0f};
#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        int index_left_all =
          (index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        int index_right_all =
          (index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        float gn12 =
          g_gn_angular[index_left_all] * weight_left + g_gn_angular[index_right_all] * weight_right;
        float gnp12 = g_gnp_angular[index_left_all] * weight_left +
                      g_gnp_angular[index_right_all] * weight_right;
        accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }
#else
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }
#endif
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

//#define USE_FIXED_NEIGHBOR 1
#define UPDATE_TEMP 1
#define BIG_ILP_CUTOFF_SQUARE 16.0
// find force and related quantities
void ILP_NEP_GR_HBN::compute_ilp(
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
  const int *group_label_ilp = group[ilp_group_method].label.data();

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
  if (num_calls++ == 0) {
#endif
    find_neighbor_ilp(
      N1,
      N2,
      rc_ilp,
      BIG_ILP_CUTOFF_SQUARE,
      box,
      group_label_ilp,
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
  ILP_neighbor<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, big_ilp_NN, big_ilp_NL, \
    type.data(), ilp_para, x, y, z, ilp_NN, \
    ilp_NL, group_label_ilp);
  GPU_CHECK_KERNEL

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
    group_label_ilp,
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



  // NEP term
  const int BLOCK_SIZE_nep = 64;
  const int N = type.size();
  const int grid_size_nep = (N2 - N1 - 1) / BLOCK_SIZE_nep + 1;

  const double rc_cell_list = 0.5 * rc;
  nep_data.f12x.fill(0);
  nep_data.f12y.fill(0);
  nep_data.f12z.fill(0);

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list,
    num_bins,
    box,
    position_per_atom,
    nep_data.cell_count,
    nep_data.cell_count_sum,
    nep_data.cell_contents);

  find_neighbor_list_nep<<<grid_size_nep, BLOCK_SIZE_nep>>>(
    paramb,
    N,
    N1,
    N2,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    box,
    type.data(),
    nep_data.cell_count.data(),
    nep_data.cell_count_sum.data(),
    nep_data.cell_contents.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  static int num_calls_nei = 0;
  if (num_calls_nei++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls_nei - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  gpu_sort_neighbor_list<<<N, paramb.MN_radial, paramb.MN_radial * sizeof(int)>>>(
    N, nep_data.NN_radial.data(), nep_data.NL_radial.data());
  GPU_CHECK_KERNEL

  gpu_sort_neighbor_list<<<N, paramb.MN_angular, paramb.MN_angular * sizeof(int)>>>(
    N, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  find_descriptor<<<grid_size_nep, BLOCK_SIZE_nep>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
#ifdef USE_TABLE
    nep_data.gn_radial.data(),
    nep_data.gn_angular.data(),
#endif
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  bool is_dipole = paramb.model_type == 1;
  find_force_radial<<<grid_size_nep, BLOCK_SIZE_nep>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
#ifdef USE_TABLE
    nep_data.gnp_radial.data(),
#endif
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size_nep, BLOCK_SIZE_nep>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
#ifdef USE_TABLE
    nep_data.gn_angular.data(),
    nep_data.gnp_angular.data(),
#endif
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body(
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    is_dipole,
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL
}