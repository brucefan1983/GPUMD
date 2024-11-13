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
