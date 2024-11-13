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
The class dealing with the interlayer potential(ILP) and neuroevolution 
potential(NEP).
TODO:
------------------------------------------------------------------------------*/

#include "ilp_nep.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <iostream>
#include <fstream>
#include <string>


const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

static inline bool check_sublayer(const char* element)
{
  return strcmp(element, "Mo") == 0 || strcmp(element, "S") == 0 ||
         strcmp(element, "Se") == 0 || strcmp(element, "W") == 0 ||
         strcmp(element, "Te");
}

ILP_NEP::ILP_NEP(FILE* fid_ilp, FILE* fid_nep_map, int num_types, int num_atoms)
{
  // read ILP elements
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP_NEP)) {
    PRINT_INPUT_ERROR("Incorrect type number of ILP parameters.\n");
  }
  std::vector<std::string> ilp_elements(num_types);
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid_ilp, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for ILP potential.");
    printf(" %s", atom_symbol);
    ilp_elements[n] = atom_symbol;
    sublayer_flag[n] = check_sublayer(atom_symbol);
  }
  printf("\n");

  // read ILP group method
  PRINT_SCANF_ERROR(fscanf(fid_ilp, "%d", &ilp_group_method), 1, 
  "Reading error for ILP group method.");
  printf("Use group method %d to identify molecule for ILP.\n", ilp_group_method);

  // read ILP parameters
  float beta, alpha, delta, epsilon, C, d, sR;
  float reff, C6, S, rcut_ilp, rcut_global;
  rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid_ilp, "%f%f%f%f%f%f%f%f%f%f%f%f", \
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

  // read NEP group method from nep map file
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &nep_group_method), 1, 
  "Reading error for NEP group method.");
  printf("Use group method %d to identify molecule for NEP.\n", nep_group_method);

  // read the number of NEP file
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &num_nep), 1, 
  "Reading error for the number of NEP file.");
  printf("NEP file number: %d\n", num_nep);

  // init parameter vectors
  parambs.resize(num_nep);
  annmbs.resize(num_nep);
  nep_data.parameters.resize(num_nep);

  // init type map cpu
  type_map_cpu.resize(num_types * num_nep, -1);
  
  // read NEP parameter from each NEP file
  for (int i = 0; i < num_nep; ++i) {
    printf("\nReading NEP %d.\n", i);
    char nep_file[100];
    int count = fscanf(fid_nep_map, "%s", nep_file);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP filename");

    std::ifstream input(nep_file);
    if (!input.is_open()) {
      std::cout << "Failed to open " << nep_file << std::endl;
      exit(1);
    }

    // nep3 1 C
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() < 3) {
      std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
      exit(1);
    }
    if (tokens[0] == "nep3") {
      parambs[i].version = 3;
    } else if (tokens[0] == "nep4") {
      parambs[i].version = 4;
    } else if (tokens[0] == "nep5") {
      parambs[i].version = 5;
    } else {
      std::cout << tokens[0]
                << " is an unsupported NEP model. We only support NEP3 and NEP4 models now."
                << std::endl;
      exit(1);
    }
    parambs[i].num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
    if (tokens.size() != 2 + parambs[i].num_types) {
      std::cout << "The first line of nep.txt should have " << parambs[i].num_types << " atom symbols."
                << std::endl;
      exit(1);
    }

    if (parambs[i].num_types == 1) {
      printf("Use the NEP%d potential with %d atom type.\n", parambs[i].version, parambs[i].num_types);
    } else {
      printf("Use the NEP%d potential with %d atom types.\n", parambs[i].version, parambs[i].num_types);
    }

    for (int n = 0; n < parambs[i].num_types; ++n) {
      int atomic_number = 0;
      for (int m = 0; m < NUM_ELEMENTS; ++m) {
        if (tokens[2 + n] == ELEMENTS[m]) {
          atomic_number = m + 1;
          break;
        }
      }
      parambs[i].atomic_numbers[n] = atomic_number - 1;
      printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), atomic_number);

      // update type map
      // for example: if ilp elements is C N B, element in nep 0 is C, elements in nep 1 are B N
      // type map should be [0,    -1,    -1,    -1,    1,    0]
      //   ilp element       C      N      B      C     N     B
      //   nep 0 element     C(0)   null   null
      //   nep 1 element                          null  N(1)  B(0)
      for (int m = 0; m < num_types; ++m) {
        if (tokens[2 + n] == ilp_elements[m]) {
          type_map_cpu[m + i * num_types] = n;
        }
      }
    }

    // cutoff 4.2 3.7 80 47 1
    tokens = get_tokens(input);
    if (tokens.size() != 5 && tokens.size() != 8) {
      std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                   "[radial_factor] [angular_factor] [zbl_factor].\n";
      exit(1);
    }
    parambs[i].rc_radial = get_float_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].rc_angular = get_float_from_token(tokens[2], __FILE__, __LINE__);
    printf("    radial cutoff = %g A.\n", parambs[i].rc_radial);
    printf("    angular cutoff = %g A.\n", parambs[i].rc_angular);

    int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
    printf("    MN_radial = %d.\n", MN_radial);
    if (MN_radial > 819) {
      std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
                << std::endl;
      exit(1);
    }
    parambs[i].MN_radial = int(ceil(MN_radial * 1.25));
    parambs[i].MN_angular = int(ceil(MN_angular * 1.25));
    max_MN_radial = max(max_MN_radial, parambs[i].MN_radial);
    max_MN_angular = max(max_MN_angular, parambs[i].MN_angular);
    printf("    enlarged MN_radial = %d.\n", parambs[i].MN_radial);
    printf("    enlarged MN_angular = %d.\n", parambs[i].MN_angular);

    if (tokens.size() == 8) {
      parambs[i].typewise_cutoff_radial_factor = get_float_from_token(tokens[5], __FILE__, __LINE__);
      parambs[i].typewise_cutoff_angular_factor = get_float_from_token(tokens[6], __FILE__, __LINE__);
      if (parambs[i].typewise_cutoff_radial_factor > 0.0f) {
        parambs[i].use_typewise_cutoff = true;
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
    parambs[i].n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
    max_n_max_angular = max(max_n_max_angular, parambs[i].n_max_angular);
    printf("    n_max_radial = %d.\n", parambs[i].n_max_radial);
    printf("    n_max_angular = %d.\n", parambs[i].n_max_angular);

    // basis_size 10 8
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
                << std::endl;
      exit(1);
    }
    parambs[i].basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
    printf("    basis_size_radial = %d.\n", parambs[i].basis_size_radial);
    printf("    basis_size_angular = %d.\n", parambs[i].basis_size_angular);

    // l_max
    tokens = get_tokens(input);
    if (tokens.size() != 4) {
      std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
      exit(1);
    }

    parambs[i].L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
    printf("    l_max_3body = %d.\n", parambs[i].L_max);
    parambs[i].num_L = parambs[i].L_max;

    int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
    int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
    printf("    l_max_4body = %d.\n", L_max_4body);
    printf("    l_max_5body = %d.\n", L_max_5body);
    if (L_max_4body == 2) {
      parambs[i].num_L += 1;
    }
    if (L_max_5body == 1) {
      parambs[i].num_L += 1;
    }

    parambs[i].dim_angular = (parambs[i].n_max_angular + 1) * parambs[i].num_L;

    // ANN
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be ANN num_neurons 0." << std::endl;
      exit(1);
    }
    annmbs[i].num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
    annmbs[i].dim = (parambs[i].n_max_radial + 1) + parambs[i].dim_angular;
    nep_model_type = parambs[i].model_type;
    if (parambs[i].model_type == 3) {
      annmbs[i].dim += 1;
    }
    max_dim = max(max_dim, annmbs[i].dim);
    printf("    ANN = %d-%d-1.\n", annmbs[i].dim, annmbs[i].num_neurons1);

    // calculated parameters:
    parambs[i].rcinv_radial = 1.0f / parambs[i].rc_radial;
    parambs[i].rcinv_angular = 1.0f / parambs[i].rc_angular;
    parambs[i].num_types_sq = parambs[i].num_types * parambs[i].num_types;

    if (parambs[i].version == 3) {
      annmbs[i].num_para_ann = (annmbs[i].dim + 2) * annmbs[i].num_neurons1 + 1;
    } else if (parambs[i].version == 4) {
      annmbs[i].num_para_ann = (annmbs[i].dim + 2) * annmbs[i].num_neurons1 * parambs[i].num_types + 1;
    } else {
      annmbs[i].num_para_ann = ((annmbs[i].dim + 2) * annmbs[i].num_neurons1 + 1) * parambs[i].num_types + 1;
    }
    if (parambs[i].model_type == 2) {
      // Polarizability models have twice as many parameters
      annmbs[i].num_para_ann *= 2;
    }
    printf("    number of neural network parameters = %d.\n", annmbs[i].num_para_ann);
    int num_para_descriptor =
      parambs[i].num_types_sq * ((parambs[i].n_max_radial + 1) * (parambs[i].basis_size_radial + 1) +
                             (parambs[i].n_max_angular + 1) * (parambs[i].basis_size_angular + 1));
    printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
    annmbs[i].num_para = annmbs[i].num_para_ann + num_para_descriptor;
    printf("    total number of parameters = %d.\n", annmbs[i].num_para);

    parambs[i].num_c_radial =
      parambs[i].num_types_sq * (parambs[i].n_max_radial + 1) * (parambs[i].basis_size_radial + 1);

    // NN and descriptor parameters
    std::vector<float> parameters(annmbs[i].num_para);
    for (int n = 0; n < annmbs[i].num_para; ++n) {
      tokens = get_tokens(input);
      parameters[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
    nep_data.parameters[i].resize(annmbs[i].num_para);
    nep_data.parameters[i].copy_from_host(parameters.data());
    update_potential(nep_data.parameters[i].data(), parambs[i], annmbs[i]);
    for (int d = 0; d < annmbs[i].dim; ++d) {
      tokens = get_tokens(input);
      parambs[i].q_scaler[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }

  }

  // read nep map to identify the nep for each group
  int num_nep_group = 0;
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &num_nep_group), 1, 
  "Reading error for the number of nep group.");
  nep_map_cpu.resize(num_nep_group);
  for (int i = 0; i < num_nep_group; ++i) {
    int nep_i = 0;    // which nep this group use
    int count = fscanf(fid_nep_map, "%d", &nep_i);
    PRINT_SCANF_ERROR(count, 1, "reading error for nep number of group.");
    if (nep_i >= num_nep) {
      if (num_nep == 1) {
        printf("There is only 1 nep file, but you set group %d of group method %d \
        to nep %d", i, nep_group_method, nep_i);
      } else {
        printf("There are %d nep files, but you set group %d of group method %d \
        to nep %d", num_nep, i, nep_group_method, nep_i);
      }
      exit(1);
    }
    nep_map_cpu[i] = nep_i;
    printf("group %d uses NEP %d.\n", i, nep_i);
  }

  // cp two maps to gpu
  nep_map.resize(num_nep_group);
  type_map.resize(num_types * num_nep);
  nep_map.copy_from_host(nep_map_cpu.data());
  type_map.copy_from_host(type_map_cpu.data());


  // initialize ilp neighbor lists and some temp vectors
  int max_neighbor_number = min(num_atoms, CUDA_MAX_NL_ILP_NEP_CBN);
  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * max_neighbor_number);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);

  // init ilp neighbor list
  ilp_data.ilp_NN.resize(num_atoms);
  ilp_data.ilp_NL.resize(num_atoms * MAX_ILP_NEIGHBOR_CBN);
  ilp_data.reduce_NL.resize(num_atoms * max_neighbor_number);
  ilp_data.big_ilp_NN.resize(num_atoms);
  ilp_data.big_ilp_NL.resize(num_atoms * MAX_BIG_ILP_NEIGHBOR_CBN);

  ilp_data.f12x.resize(num_atoms * max_neighbor_number);
  ilp_data.f12y.resize(num_atoms * max_neighbor_number);
  ilp_data.f12z.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_CBN);
  ilp_data.f12y_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_CBN);
  ilp_data.f12z_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_CBN);

  // init constant cutoff coeff
  float h_tap_coeff[8] = \
    {1.0f, 0.0f, 0.0f, 0.0f, -35.0f, 84.0f, -70.0f, 20.0f};
  CHECK(gpuMemcpyToSymbol(Tap_coeff, h_tap_coeff, 8 * sizeof(float)));

  // set ilp_flag to 1
  ilp_flag = 1;

  // initialize nep neighbor lists
  nep_data.f12x.resize(num_atoms * max_MN_angular);
  nep_data.f12y.resize(num_atoms * max_MN_angular);
  nep_data.f12z.resize(num_atoms * max_MN_angular);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * max_MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * max_MN_angular);
  nep_data.Fp.resize(num_atoms * max_dim);
  nep_data.sum_fxyz.resize(num_atoms * (max_n_max_angular + 1) * NUM_OF_ABC);
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

void ILP_NEP::update_potential(float* parameters, ParaMB& paramb, ANN& ann)
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

ILP_NEP::~ILP_NEP(void)
{
  // nothing
}

static __device__ __forceinline__ float calc_Tap(const float r_ij, const float Rcutinv)
{
  float Tap, r;

  r = r_ij * Rcutinv;
  Tap = Tap_coeff[7];
  Tap = Tap * r + Tap_coeff[6];
  Tap = Tap * r + Tap_coeff[5];
  Tap = Tap * r + Tap_coeff[4];
  Tap = Tap * r + Tap_coeff[3];
  Tap = Tap * r + Tap_coeff[2];
  Tap = Tap * r + Tap_coeff[1];
  Tap = Tap * r + Tap_coeff[0];

  return Tap;
}

// calculate the derivatives of long-range cutoff term
static __device__ __forceinline__ float calc_dTap(const float r_ij, const float Rcutinv)
{
  float dTap, r;
  
  r = r_ij * Rcutinv;
  dTap = 7.0f * Tap_coeff[7];
  dTap = dTap * r + 6.0f * Tap_coeff[6];
  dTap = dTap * r + 5.0f * Tap_coeff[5];
  dTap = dTap * r + 4.0f * Tap_coeff[4];
  dTap = dTap * r + 3.0f * Tap_coeff[3];
  dTap = dTap * r + 2.0f * Tap_coeff[2];
  dTap = dTap * r + Tap_coeff[1];
  dTap *= Rcutinv;

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
  const int *group_label,
  bool sublayer_flag[MAX_TYPE_ILP_NEP])
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
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

      // if material has sublayer, calc normal in sublayer (same type)
      if (group_label[n1] == group_label[n2] && d12sq < rcutsq && 
          (type1 == type2 || !sublayer_flag[type1]) && d12sq != 0) {
        // ilp_neighbor_list[count++ * number_of_particles + n1] = n2;
        neighptr[count++] = n2;
      }
    }

    // sort orders of neighbors
    if (sublayer_flag[type1]) {
      // init neighsort
      for (int ll = 0; ll < count; ++ll) {
        neighsort[ll] = neighptr[ll];
        check[ll] = neighptr[ll];
      }

      // select the first neighbor of atom n1
      if (count == MAX_ILP_NEIGHBOR_TMD) {
        neighsort[0] = neighptr[0];
        check[0] = -1;
      } else if (count < MAX_ILP_NEIGHBOR_TMD && count > 0) {
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
      } else if (count > MAX_ILP_NEIGHBOR_TMD) {
        printf("ERROR in ILP NEIGHBOR LIST\n");
        printf("\n===== ILP neighbor number[%d] is greater than 6 =====\n", count);
        exit(1);
      }

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

// calculate the normals and its derivatives for C B N
static __device__ void calc_normal_cbn(
  float (&vet)[MAX_ILP_NEIGHBOR_TMD][3],
  int cont,
  float (&normal)[3],
  float (&dnormdri)[3][3],
  float (&dnormal)[3][MAX_ILP_NEIGHBOR_TMD][3])
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

    //############################################################################################
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

// calculate the normals and its derivatives for TMDs
static __device__ void calc_normal_tmd(
  float (&vect)[MAX_ILP_NEIGHBOR_TMD][3],
  int cont,
  float (&normal)[3],
  float (&dnormdri)[3][3],
  float (&dnormal)[3][MAX_ILP_NEIGHBOR_TMD][3])
{
  int id, ip, m;
  float  dni[3];
  float  dnn[3][3], dpvdri[3][3];
  float Nave[3], pvet[MAX_ILP_NEIGHBOR_TMD][3], dpvet1[MAX_ILP_NEIGHBOR_TMD][3][3], dpvet2[MAX_ILP_NEIGHBOR_TMD][3][3], dNave[3][MAX_ILP_NEIGHBOR_TMD][3];

  float nninv;

  // initialize the arrays
  for (id = 0; id < 3; id++) {
    dni[id] = 0.0f;

    Nave[id] = 0.0f;
    for (ip = 0; ip < 3; ip++) {
      dpvdri[ip][id] = 0.0f;
      for (m = 0; m < MAX_ILP_NEIGHBOR_TMD; m++) {
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
        for (m = 0; m < MAX_ILP_NEIGHBOR_TMD; ++m) {
          dnormal[id][m][ip] = 0.0f;
        }
      }
    }
  } else if (cont > 1 && cont < MAX_ILP_NEIGHBOR_TMD) {
    for (int k = 0; k < cont - 1; ++k) {
      for (ip = 0; ip < 3; ++ip) {
        pvet[k][ip] = vect[k][modulo(ip + 1, 3)] * vect[k + 1][modulo(ip + 2, 3)] -
                vect[k][modulo(ip + 2, 3)] * vect[k + 1][modulo(ip + 1, 3)];
      }
      // dpvet1[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l
      // derivatives respect to atom l
      // dNik,x/drl
      dpvet1[k][0][0] = 0.0f;
      dpvet1[k][0][1] = vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet1[k][0][2] = -vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][1];
      // dNik,y/drl
      dpvet1[k][1][0] = -vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet1[k][1][1] = 0.0f;
      dpvet1[k][1][2] = vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][0];
      // dNik,z/drl
      dpvet1[k][2][0] = vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][1];
      dpvet1[k][2][1] = -vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][0];
      dpvet1[k][2][2] = 0.0f;

      // dpvet2[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l+1
      // derivatives respect to atom l+1
      // dNik,x/drl+1
      dpvet2[k][0][0] = 0.0f;
      dpvet2[k][0][1] = -vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet2[k][0][2] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][1];
      // dNik,y/drl+1
      dpvet2[k][1][0] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet2[k][1][1] = 0.0f;
      dpvet2[k][1][2] = -vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][0];
      // dNik,z/drl+1
      dpvet2[k][2][0] = -vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][1];
      dpvet2[k][2][1] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][0];
      dpvet2[k][2][2] = 0.0f;
    }

    // average the normal vectors by using the MAX_ILP_NEIGHBOR_TMD neighboring planes
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
    // dnn[m][id]: the derivative of nn respect to r[m][id], m=0,...MAX_ILP_NEIGHBOR_TMD-1; id=0,1,2
    // r[m][id]: the id's component of atom m
    for (m = 0; m < cont; m++) {
      for (id = 0; id < 3; id++) {
        dnn[m][id] = (Nave[0] * dNave[0][m][id] + Nave[1] * dNave[1][m][id] +
                      Nave[2] * dNave[2][m][id]) * nninv;
      }
    }
    // dnormal[i][id][m][ip]: the derivative of normal[i][id] respect to r[m][ip], id,ip=0,1,2.
    // for atom m, which is a neighbor atom of atom i, m = 0,...,MAX_ILP_NEIGHBOR_TMD-1
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
  } else if (cont == MAX_ILP_NEIGHBOR_TMD) {
    // derivatives of Ni[l] respect to the MAX_ILP_NEIGHBOR_TMD neighbors
    for (int k = 0; k < MAX_ILP_NEIGHBOR_TMD; ++k) {
      for (ip = 0; ip < 3; ++ip) {
        pvet[k][ip] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][modulo(ip + 1, 3)] *
                vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][modulo(ip + 2, 3)] -
            vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][modulo(ip + 2, 3)] *
                vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][modulo(ip + 1, 3)];
      }
      // dpvet1[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l
      // derivatives respect to atom l
      // dNik,x/drl
      dpvet1[k][0][0] = 0.0f;
      dpvet1[k][0][1] = vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet1[k][0][2] = -vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][1];
      // dNik,y/drl
      dpvet1[k][1][0] = -vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet1[k][1][1] = 0.0f;
      dpvet1[k][1][2] = vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][0];
      // dNik,z/drl
      dpvet1[k][2][0] = vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][1];
      dpvet1[k][2][1] = -vect[modulo(k + 1, MAX_ILP_NEIGHBOR_TMD)][0];
      dpvet1[k][2][2] = 0.0f;

      // dpvet2[k][l][ip]: the derivatve of the k (=0,...cont-1)th Nik respect to the ip component of atom l+1
      // derivatives respect to atom l+1
      // dNik,x/drl+1
      dpvet2[k][0][0] = 0.0f;
      dpvet2[k][0][1] = -vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet2[k][0][2] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][1];
      // dNik,y/drl+1
      dpvet2[k][1][0] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][2];
      dpvet2[k][1][1] = 0.0f;
      dpvet2[k][1][2] = -vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][0];
      // dNik,z/drl+1
      dpvet2[k][2][0] = -vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][1];
      dpvet2[k][2][1] = vect[modulo(k, MAX_ILP_NEIGHBOR_TMD)][0];
      dpvet2[k][2][2] = 0.0f;
    }

    // average the normal vectors by using the MAX_ILP_NEIGHBOR_TMD neighboring planes
    for (ip = 0; ip < 3; ++ip) {
      Nave[ip] = 0.0f;
      for (int k = 0; k < MAX_ILP_NEIGHBOR_TMD; ++k) {
        Nave[ip] += pvet[k][ip];
      }
      Nave[ip] /= MAX_ILP_NEIGHBOR_TMD;
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

    // derivatives of non-normalized normal vector, dNave:3xMAX_ILP_NEIGHBOR_TMDx3 array
    // dNave[id][m][ip]: the derivatve of the id component of Nave respect to the ip component of atom m
    for (id = 0; id < 3; ++id) {
      for (ip = 0; ip < 3; ++ip) {
        for (
            m = 0; m < MAX_ILP_NEIGHBOR_TMD;
            ++m) {    // sum of the derivatives of the mth and (m-1)th normal vector respect to the atom m
          dNave[id][m][ip] =
              (dpvet1[modulo(m, MAX_ILP_NEIGHBOR_TMD)][id][ip] + dpvet2[modulo(m - 1, MAX_ILP_NEIGHBOR_TMD)][id][ip]) / MAX_ILP_NEIGHBOR_TMD;
        }
      }
    }
    // derivatives of nn, dnn:MAX_ILP_NEIGHBOR_TMDx3 vector
    // dnn[m][id]: the derivative of nn respect to r[m][id], m=0,...MAX_ILP_NEIGHBOR_TMD-1; id=0,1,2
    // r[m][id]: the id's component of atom m
    for (m = 0; m < MAX_ILP_NEIGHBOR_TMD; ++m) {
      for (id = 0; id < 3; ++id) {
        dnn[m][id] =
            (Nave[0] * dNave[0][m][id] + Nave[1] * dNave[1][m][id] + Nave[2] * dNave[2][m][id]) *
            nninv;
      }
    }
    // dnormal[i][id][m][ip]: the derivative of normal[i][id] respect to r[m][ip], id,ip=0,1,2.
    // for atom m, which is a neighbor atom of atom i, m = 0,...,MAX_ILP_NEIGHBOR_TMD-1
    for (m = 0; m < MAX_ILP_NEIGHBOR_TMD; ++m) {
      for (id = 0; id < 3; ++id) {
        for (ip = 0; ip < 3; ++ip) {
          dnormal[id][m][ip] = dNave[id][m][ip] * nninv - Nave[id] * dnn[m][ip] * nninv * nninv;
        }
      }
    }
  } else {
    printf("\n===== ILP neighbor number[%d] is greater than 6 =====\n", cont);
    exit(1);
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
  double *g_potential,
  float *g_f12x,
  float *g_f12y,
  float *g_f12z,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh,
  bool sublayer_flag[MAX_TYPE_ILP_NEP])
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

    float delkix_half[MAX_ILP_NEIGHBOR_TMD] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float delkiy_half[MAX_ILP_NEIGHBOR_TMD] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float delkiz_half[MAX_ILP_NEIGHBOR_TMD] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // calculate the normal
    int cont = 0;
    float normal[3];
    float dnormdri[3][3];
    float dnormal[3][MAX_ILP_NEIGHBOR_TMD][3];

    float vet[MAX_ILP_NEIGHBOR_TMD][3];
    int id, ip, m;
    for (id = 0; id < 3; ++id) {
      normal[id] = 0.0f;
      for (ip = 0; ip < 3; ++ip) {
        dnormdri[ip][id] = 0.0f;
        for (m = 0; m < MAX_ILP_NEIGHBOR_TMD; ++m) {
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
    
    if (sublayer_flag[type1]) {
      calc_normal_tmd(vet, cont, normal, dnormdri, dnormal);
    } else {
      calc_normal_cbn(vet, cont, normal, dnormdri, dnormal);
    }

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

      if (r >= Rcut) {
        continue;
      }

      // calc att
      float Tap, dTap, rinv;
      float Rcutinv = 1.0f / Rcut;
      rinv = 1.0f / r;
      Tap = calc_Tap(r, Rcutinv);
      dTap = calc_dTap(r, Rcutinv);

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



// ----- NEP part -----
// nep find neighbor list
static __global__ void find_neighbor_list_large_box(
  ILP_NEP::ParaMB paramb,
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
  ILP_NEP::ParaMB paramb,
  ILP_NEP::ANN annmb,
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
  const bool is_polarizability,
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

    if (is_polarizability) {
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0_pol[t1],
        annmb.b0_pol[t1],
        annmb.w1_pol[t1],
        annmb.b1_pol,
        q,
        F,
        Fp);
      // Add the potential F for this atom to the diagonal of the virial
      g_virial[n1] = F;
      g_virial[n1 + N * 1] = F;
      g_virial[n1 + N * 2] = F;

      // Reset the potential and forces such that they
      // are zero for the next call to the model. The next call
      // is not used in the case of is_pol = True, but it doesn't
      // hurt to clean up.
      F = 0.0f;
      for (int d = 0; d < annmb.dim; ++d) {
        Fp[d] = 0.0f;
      }
    }

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
  ILP_NEP::ParaMB paramb,
  ILP_NEP::ANN annmb,
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
  const bool is_dipole,
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
      if (is_dipole) {
        // The dipole is proportional to minus the sum of the virials times r12
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_sxx -= r12_square * f21[0];
        s_syy -= r12_square * f21[1];
        s_szz -= r12_square * f21[2];
      } else {
        s_sxx += r12[0] * f21[0];
        s_syy += r12[1] * f21[1];
        s_szz += r12[2] * f21[2];
      }
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
  ILP_NEP::ParaMB paramb,
  ILP_NEP::ANN annmb,
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









