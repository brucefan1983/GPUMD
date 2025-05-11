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
#include <cstring>
// TODO
// #include <iomanip>


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
         strcmp(element, "Te") == 0;
}

#ifdef CODING
static __global__ void check_para_gpu(
  void* h_parambs,
  void* h_annmbs,
  const int num_nep)
{
  printf("\n\n========== CHECK PARAMETER BUFFER ==========\n");
  float* f_parambs = (float*)h_parambs;
  int* i_parambs = (int*)h_parambs;
  float* f_annmbs = (float*)h_annmbs;
  int* i_annmbs = (int*)h_annmbs;
  for (int i = 0; i < num_nep; ++i) {
    printf("---------- NEP %d HEAD PARAMETERS -----------\n", i);
    printf("use_typewise_cutoff               %8d\n", i_parambs[UTC     ]);
    printf("typewise_cutoff_radial_factor     %8f\n", f_parambs[TCRF    ]);
    printf("typewise_cutoff_angular_factor    %8f\n", f_parambs[TCAF    ]);
    printf("nep version                       %8d\n", i_parambs[VERSION ]);
    printf("rc_radial                         %8f\n", f_parambs[RCR     ]);
    printf("rc_angular                        %8f\n", f_parambs[RCA     ]);
    printf("rcinv_radial                      %8f\n", f_parambs[RCIR    ]);
    printf("rcinv_angular                     %8f\n", f_parambs[RCIA    ]);
    printf("MN_radial                         %8d\n", i_parambs[MNR     ]);
    printf("MN_angular                        %8d\n", i_parambs[MNA     ]);
    printf("n_max_radial                      %8d\n", i_parambs[NMAXR   ]);
    printf("n_max_angular                     %8d\n", i_parambs[NMAXA   ]);
    printf("L_max                             %8d\n", i_parambs[LMAX    ]);
    printf("dim_angular                       %8d\n", i_parambs[DIMA    ]);
    printf("num_L                             %8d\n", i_parambs[NUML    ]);
    printf("basis_size_radial                 %8d\n", i_parambs[BSR     ]);
    printf("basis_size_angular                %8d\n", i_parambs[BSA     ]);
    printf("num_types_sq                      %8d\n", i_parambs[NTS     ]);
    printf("num_c_radial                      %8d\n", i_parambs[NCR     ]);
    printf("num_types                         %8d\n", i_parambs[NT      ]);
    printf("ann dim                           %8d\n",  i_annmbs[ANNDIM  ]);
    printf("num_neurous1                      %8d\n",  i_annmbs[NNEUR   ]);
    printf("bias for output layer             %8f\n",  f_annmbs[OUTB1   ]);

    i_annmbs += H_ANN_OFFSET;
    f_annmbs += H_ANN_OFFSET;
    i_parambs += H_PAR_OFFSET;
    f_parambs += H_PAR_OFFSET;

  }

  f_parambs = (float*)h_parambs;
  i_parambs = (int*)h_parambs;
  f_annmbs = (float*)h_annmbs;
  i_annmbs = (int*)h_annmbs;
  for (int i = 0; i < num_nep; ++i) {
    printf("---------- NEP %d ANN PARAMETERS -----------\n", i);
    float* w0 = FLT_PTR((float*)h_annmbs + i * H_ANN_OFFSET + PTRW0);
    float* b0 = FLT_PTR((float*)h_annmbs + i * H_ANN_OFFSET + PTRB0);
    float* w1 = FLT_PTR((float*)h_annmbs + i * H_ANN_OFFSET + PTRW1);
    float* c  = FLT_PTR((float*)h_annmbs + i * H_ANN_OFFSET + PTRC);
    float* qs = FLT_PTR((float*)h_parambs + i * H_PAR_OFFSET + PTRQS); 
    int anndim = i_annmbs[ANNDIM];
    int nneu = i_annmbs[NNEUR];
    printf("# w0 type1\n");
    for (int j = 0; j < anndim * nneu; ++j) {
      printf("%15.7e\n", w0[j]);
    }

    printf("# b0 type1\n");
    for (int j = 0; j < nneu; ++j) {
      printf("%15.7e\n", b0[j]);
    }

    printf("# w1 type1\n");
    for (int j = 0; j < nneu; ++j) {
      printf("%15.7e\n", w1[j]);
    }

    printf("# c\n");
    int num_c = i_parambs[NTS] * ((i_parambs[NMAXR] + 1) * (i_parambs[BSR] + 1) + 
                                  (i_parambs[NMAXA] + 1) * (i_parambs[BSA] + 1));
    for (int j = 0; j < num_c; ++j) {
      printf("%15.7e\n", c[j]);
    }

    printf("# q_scaler\n");
    for (int j = 0; j < anndim; ++j) {
      printf("%15.7e\n", qs[j]);
    }

    i_annmbs += H_ANN_OFFSET;
    f_annmbs += H_ANN_OFFSET;
    i_parambs += H_PAR_OFFSET;
    f_parambs += H_PAR_OFFSET;
  }

  printf("========== CHECK PARAMETER BUFFER ==========\n\n");
}
#endif


ILP_NEP::ILP_NEP(FILE* fid_ilp, FILE* fid_nep_map, int num_types, int num_atoms)
{
#ifdef USE_TABLE
  printf("=============================================================================\n");
  printf("NEP+ILP potential doesn't support `USE_TABLE` and uses normal version of NEP.\n");
  printf("=============================================================================\n\n");
#endif
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
    sublayer_flag_cpu[n] = check_sublayer(atom_symbol);
    if (sublayer_flag_cpu[n]) {
      printf("(sublayer)");
    }
  }
  printf("\n");
  // cp sublayer flags to gpu
  sublayer_flag_gpu.resize(MAX_TYPE_ILP_NEP);
  sublayer_flag_gpu.copy_from_host(sublayer_flag_cpu);

  // read ILP group method
  PRINT_SCANF_ERROR(fscanf(fid_ilp, "%d", &ilp_group_method), 1, 
  "Reading error for ILP group method.");
  printf("Use group method %d to identify molecule for ILP.\n", ilp_group_method);
  PRINT_SCANF_ERROR(fscanf(fid_ilp, "%d", &ilp_sub_group_method), 1, 
  "Reading error for ILP group method.");
  printf("Use group method %d to identify molecule(sublayer) for ILP.\n", ilp_sub_group_method);

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
  max_nep_rc = 0.0;

  // init type map cpu
  type_map_cpu.resize(num_types * num_nep, -1);
  
  // read NEP parameter from each NEP file
  std::vector<std::vector<float>> all_ann_para({});     // save paras
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
    parambs[i].rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
    parambs[i].rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
    printf("    radial cutoff = %g A.\n", parambs[i].rc_radial);
    printf("    angular cutoff = %g A.\n", parambs[i].rc_angular);
    // save the max rc_radial, same to max rc
    max_nep_rc = max(max_nep_rc, parambs[i].rc_radial);

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
      parambs[i].typewise_cutoff_radial_factor = get_double_from_token(tokens[5], __FILE__, __LINE__);
      parambs[i].typewise_cutoff_angular_factor = get_double_from_token(tokens[6], __FILE__, __LINE__);
      if (parambs[i].typewise_cutoff_radial_factor > 0.0f) {
        parambs[i].use_typewise_cutoff = true;
      }
    }
//  #ifdef USE_TABLE
//    if (paramb.use_typewise_cutoff) {
//      PRINT_INPUT_ERROR("Cannot use tabulated radial functions with typewise cutoff.");
//    }
//  #endif

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
      parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    all_ann_para.push_back(parameters);
    for (int d = 0; d < annmbs[i].dim; ++d) {
      tokens = get_tokens(input);
      parambs[i].q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }

  }

  // check ilp rc and nep rc, make sure ilp's larger than nep's
  if (max_nep_rc > rc) {
    printf("ERROR: Please make sure cutoff of ILP is larger than it of NEP.\n");
    exit(1);
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
    printf("group %d of group method %d uses NEP %d.\n", i, nep_group_method, nep_i);
  }

#ifdef CODING
  printf("\n========== TYPE MAP: ILP --> NEP ==========\n");
  for (int i = 0; i < num_nep; ++i) {
    for (int j = 0; j < num_types; ++j) {
      printf("%d\t\t", type_map_cpu[j + i * num_types]);
    }
    printf("\n");
  }
  printf("========== TYPE MAP: ILP --> NEP ==========\n");
#endif
  // cp two maps to gpu
  nep_map.resize(num_nep_group);
  type_map.resize(num_types * num_nep);
  nep_map.copy_from_host(nep_map_cpu.data());
  type_map.copy_from_host(type_map_cpu.data());


  // initialize ilp neighbor lists and some temp vectors
  int max_neighbor_number = min(num_atoms, CUDA_MAX_NL_ILP_NEP_TMD);
  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * max_neighbor_number);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);

  // init ilp neighbor list
  ilp_data.ilp_NN.resize(num_atoms);
  ilp_data.ilp_NL.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.reduce_NL.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x.resize(num_atoms * max_neighbor_number);
  ilp_data.f12y.resize(num_atoms * max_neighbor_number);
  ilp_data.f12z.resize(num_atoms * max_neighbor_number);

  ilp_data.f12x_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.f12y_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.f12z_ilp_neigh.resize(num_atoms * MAX_ILP_NEIGHBOR_TMD);

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
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);

// #ifdef USE_TABLE
//   construct_table(parameters.data());
//   printf("    use tabulated radial functions to speed up.\n");
// #endif

  // updata nep parameters and ann parameters
  // calc buffer size
  int parambs_size = num_nep * (H_PAR_OFFSET) * SIZEOF_INT;
  int annmbs_size = num_nep * (H_ANN_OFFSET) * SIZEOF_INT;
  for (int i = 0; i < num_nep; ++i) {
    // add q_scaler and atomic_number buffers
    parambs_size += (annmbs[i].dim + parambs[i].num_types) * SIZEOF_INT;
    // add w0, b0, w1, c buffer
    annmbs_size += (annmbs[i].num_para - 1) * SIZEOF_INT;   // rm b1
  }
  int all_para_size = parambs_size + annmbs_size;

  // init buffers in cpu and gpu
  char* para_buffer_cpu = (char*) malloc(all_para_size);
  nep_data.para_buffer_gpu.resize(all_para_size);
  char* para_buffer_gpu = nep_data.para_buffer_gpu.data();

  // save head pointers
  h_parambs = para_buffer_gpu;
  h_annmbs = h_parambs + num_nep * H_PAR_OFFSET * SIZEOF_INT;

  // set parameters data in cpu buffer
  int* para_buf_w = (int*) para_buffer_cpu;     // a write ptr to cpu buffer
  int* para_buf_ptrw = para_buf_w;              // a write ptr to write the ptrs of para

  // parambs head
  for (int i = 0; i < num_nep; ++i) {
    int use_typewise_cutoff_int = parambs[i].use_typewise_cutoff;
    memcpy(para_buf_w + UTC    , &use_typewise_cutoff_int, SIZEOF_INT);
    memcpy(para_buf_w + TCRF   , &(parambs[i].typewise_cutoff_radial_factor), SIZEOF_INT);
    memcpy(para_buf_w + TCAF   , &(parambs[i].typewise_cutoff_angular_factor), SIZEOF_INT);
    memcpy(para_buf_w + VERSION, &(parambs[i].version), SIZEOF_INT);
    memcpy(para_buf_w + RCR    , &(parambs[i].rc_radial), SIZEOF_INT);
    memcpy(para_buf_w + RCA    , &(parambs[i].rc_angular), SIZEOF_INT);
    memcpy(para_buf_w + RCIR   , &(parambs[i].rcinv_radial), SIZEOF_INT);
    memcpy(para_buf_w + RCIA   , &(parambs[i].rcinv_angular), SIZEOF_INT);
    memcpy(para_buf_w + MNR    , &(parambs[i].MN_radial), SIZEOF_INT);
    memcpy(para_buf_w + MNA    , &(parambs[i].MN_angular), SIZEOF_INT);
    memcpy(para_buf_w + NMAXR  , &(parambs[i].n_max_radial), SIZEOF_INT);
    memcpy(para_buf_w + NMAXA  , &(parambs[i].n_max_angular), SIZEOF_INT);
    memcpy(para_buf_w + LMAX   , &(parambs[i].L_max), SIZEOF_INT);
    memcpy(para_buf_w + DIMA   , &(parambs[i].dim_angular), SIZEOF_INT);
    memcpy(para_buf_w + NUML   , &(parambs[i].num_L), SIZEOF_INT);
    memcpy(para_buf_w + BSR    , &(parambs[i].basis_size_radial), SIZEOF_INT);
    memcpy(para_buf_w + BSA    , &(parambs[i].basis_size_angular), SIZEOF_INT);
    memcpy(para_buf_w + NTS    , &(parambs[i].num_types_sq), SIZEOF_INT);
    memcpy(para_buf_w + NCR    , &(parambs[i].num_c_radial), SIZEOF_INT);
    memcpy(para_buf_w + NT     , &(parambs[i].num_types), SIZEOF_INT);
    para_buf_w += H_PAR_OFFSET;    // skip 2 pointers: PTRQS PTRAN
  }

  // annmbs head
  for (int i = 0; i < num_nep; ++i) {
    memcpy(para_buf_w + ANNDIM , &(annmbs[i].dim), SIZEOF_INT);
    memcpy(para_buf_w + NNEUR  , &(annmbs[i].num_neurons1), SIZEOF_INT);
    int b1_pos = 0;
    if (parambs[i].version == 3) {
      b1_pos = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
    } else if (parambs[i].version == 4) {
      b1_pos = (annmbs[i].dim + 2) * annmbs[i].num_neurons1 * parambs[i].num_types;
    } else if (parambs[i].version == 5) {
      b1_pos = ((annmbs[i].dim + 2) * annmbs[i].num_neurons1 + 1) * parambs[i].num_types;
    }
    memcpy(para_buf_w + OUTB1  , &(all_ann_para[i][b1_pos]), SIZEOF_INT);
    para_buf_w += H_ANN_OFFSET;  // skip 4 pointers: PTRC PTRW0 PTRB0 PTRW1 and an empty
  }
  
  // move gpu buffer pointer
  para_buffer_gpu += num_nep * (H_PAR_OFFSET + H_ANN_OFFSET) * SIZEOF_INT;

  // q_scaler
  for (int i = 0; i < num_nep; ++i) {
    // update pointer q_scaler
    memcpy(para_buf_ptrw + i * H_PAR_OFFSET + PTRQS, &para_buffer_gpu, SIZEOF_POINTER);

    int qs_offset = annmbs[i].dim;
    memcpy(para_buf_w, parambs[i].q_scaler, qs_offset * SIZEOF_INT);
    para_buf_w += qs_offset;
    para_buffer_gpu += qs_offset * SIZEOF_INT;
  }

  // atomic_numbers
  for (int i = 0; i < num_nep; ++i) {
    // update pointer atomic number
    memcpy(para_buf_ptrw + i * H_PAR_OFFSET + PTRAN, &para_buffer_gpu, SIZEOF_POINTER);

    int an_offset = parambs[i].num_types;
    memcpy(para_buf_w, parambs[i].atomic_numbers, an_offset * SIZEOF_INT);
    para_buf_w += an_offset;
    para_buffer_gpu += an_offset * SIZEOF_INT;
  }

  // move cpu para buffer ptr w to the annmb head
  para_buf_ptrw += num_nep * H_PAR_OFFSET;

  // w0
  for (int i = 0; i < num_nep; ++i) {
    // update pointer w0
    memcpy(para_buf_ptrw + i * H_ANN_OFFSET + PTRW0, &para_buffer_gpu, SIZEOF_POINTER);

    int w0_offset = annmbs[i].num_neurons1 * annmbs[i].dim;
    if (parambs[i].version == 3) {
      memcpy(para_buf_w, &(all_ann_para[i][0]), w0_offset * SIZEOF_INT);
      para_buf_w += w0_offset;
      para_buffer_gpu += w0_offset * SIZEOF_INT;
    } else if (parambs[i].version == 4) {
      int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
      for (int t = 0; t < parambs[i].num_types; ++t) {
        memcpy(para_buf_w, &(all_ann_para[i][t * t_offset]), w0_offset * SIZEOF_INT);
        para_buf_w += w0_offset;
      }
      para_buffer_gpu += parambs[i].num_types * w0_offset * SIZEOF_INT;
    } else if (parambs[i].version == 5) {
      int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1 + 1;
      for (int t = 0; t < parambs[i].num_types; ++t) {
        memcpy(para_buf_w, &(all_ann_para[i][t * t_offset]), w0_offset * SIZEOF_INT);
        para_buf_w += w0_offset;
      }
      para_buffer_gpu += parambs[i].num_types * w0_offset * SIZEOF_INT;
    }

  }

  // b0
  for (int i = 0; i < num_nep; ++i) {
    // update pointer b0
    memcpy(para_buf_ptrw + i * H_ANN_OFFSET + PTRB0, &para_buffer_gpu, SIZEOF_POINTER);

    int b0_offset = annmbs[i].num_neurons1;
    if (parambs[i].version == 3) {
      int b0_base = annmbs[i].num_neurons1 * annmbs[i].dim;
      memcpy(para_buf_w, &(all_ann_para[i][b0_base]), b0_offset * SIZEOF_INT);
      para_buf_w += b0_offset;
      para_buffer_gpu += b0_offset * SIZEOF_INT;
    } else if (parambs[i].version == 4) {
      int b0_base = annmbs[i].num_neurons1 * annmbs[i].dim;
      int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
      for (int t = 0; t < parambs[i].num_types; ++t) {
        memcpy(para_buf_w, &(all_ann_para[i][b0_base + t * t_offset]), b0_offset * SIZEOF_INT);
        para_buf_w += b0_offset;
      }
      para_buffer_gpu += parambs[i].num_types * b0_offset * SIZEOF_INT;
    } else if (parambs[i].version == 5) {
      int b0_base = annmbs[i].num_neurons1 * annmbs[i].dim;
      int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1 + 1;
      for (int t = 0; t < parambs[i].num_types; ++t) {
        memcpy(para_buf_w, &(all_ann_para[i][b0_base + t * t_offset]), b0_offset * SIZEOF_INT);
        para_buf_w += b0_offset;
      }
      para_buffer_gpu += parambs[i].num_types * b0_offset * SIZEOF_INT;
    }

  }

  // w1
  for (int i = 0; i < num_nep; ++i) {
    // update pointer w1
    memcpy(para_buf_ptrw + i * H_ANN_OFFSET + PTRW1, &para_buffer_gpu, SIZEOF_POINTER);

    int w1_offset = annmbs[i].num_neurons1;
    if (parambs[i].version == 3) {
      int w1_base = annmbs[i].num_neurons1 * (annmbs[i].dim + 1);
      memcpy(para_buf_w, &(all_ann_para[i][w1_base]), w1_offset * SIZEOF_INT);
      para_buf_w += w1_offset;
      para_buffer_gpu += w1_offset * SIZEOF_INT;
    } else if (parambs[i].version == 4) {
      int w1_base = annmbs[i].num_neurons1 * (annmbs[i].dim + 1);
      int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1;
      for (int t = 0; t < parambs[i].num_types; ++t) {
        memcpy(para_buf_w, &(all_ann_para[i][w1_base + t * t_offset]), w1_offset * SIZEOF_INT);
        para_buf_w += w1_offset;
      }
      para_buffer_gpu += parambs[i].num_types * w1_offset * SIZEOF_INT;
    } else if (parambs[i].version == 5) {
      int w1_base = annmbs[i].num_neurons1 * (annmbs[i].dim + 1);
      int t_offset = (annmbs[i].dim + 2) * annmbs[i].num_neurons1 + 1;
      ++w1_offset;
      for (int t = 0; t < parambs[i].num_types; ++t) {
        memcpy(para_buf_w, &(all_ann_para[i][w1_base + t * t_offset]), w1_offset * SIZEOF_INT);
        para_buf_w += w1_offset;
      }
      para_buffer_gpu += parambs[i].num_types * w1_offset * SIZEOF_INT;
    }

  }

  // c
  for (int i = 0; i < num_nep; ++i) {
    // update pointer c
    memcpy(para_buf_ptrw + i * H_ANN_OFFSET + PTRC, &para_buffer_gpu, SIZEOF_POINTER);

    int c_offset = annmbs[i].num_para - annmbs[i].num_para_ann;
    int c_base = annmbs[i].num_para_ann;
    memcpy(para_buf_w, &(all_ann_para[i][c_base]), c_offset * SIZEOF_INT);
    para_buf_w += c_offset;
    para_buffer_gpu += c_offset * SIZEOF_INT;
  }


  // cp parameters from cpu to gpu
  nep_data.para_buffer_gpu.copy_from_host(para_buffer_cpu, all_para_size);

  // free cpu buffer and set ptrs to null
  free(para_buffer_cpu);
  para_buffer_cpu = nullptr;
  para_buffer_gpu = nullptr;
  para_buf_w= nullptr;
  para_buf_ptrw = nullptr;

#ifdef CODING
//  check_para_gpu<<<1,1>>>(h_parambs, h_annmbs, num_nep);
//  GPU_CHECK_KERNEL
#endif
}

ILP_NEP::~ILP_NEP(void)
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
    Tap = Tap_coeff[7];
    for (int i = 6; i >= 0; --i) {
      Tap = Tap * r + Tap_coeff[i];
    }
  }

  // r = r_ij * Rcutinv;
  // Tap = Tap_coeff[7];
  // Tap = Tap * r + Tap_coeff[6];
  // Tap = Tap * r + Tap_coeff[5];
  // Tap = Tap * r + Tap_coeff[4];
  // Tap = Tap * r + Tap_coeff[3];
  // Tap = Tap * r + Tap_coeff[2];
  // Tap = Tap * r + Tap_coeff[1];
  // Tap = Tap * r + Tap_coeff[0];

  return Tap;
}

// calculate the derivatives of long-range cutoff term
static __device__ __forceinline__ float calc_dTap(const float r_ij, const float Rcutinv)
{
  float dTap, r;
  
  r = r_ij * Rcutinv;
  if (r >= 1.0f) {
    dTap = 0.0f;
  } else {
    dTap = 7.0f * Tap_coeff[7];
    for (int i = 6; i > 0; --i) {
      dTap = dTap * r + i * Tap_coeff[i];
    }
    dTap *= Rcutinv;
  }
  // r = r_ij * Rcutinv;
  // dTap = 7.0f * Tap_coeff[7];
  // dTap = dTap * r + 6.0f * Tap_coeff[6];
  // dTap = dTap * r + 5.0f * Tap_coeff[5];
  // dTap = dTap * r + 4.0f * Tap_coeff[4];
  // dTap = dTap * r + 3.0f * Tap_coeff[3];
  // dTap = dTap * r + 2.0f * Tap_coeff[2];
  // dTap = dTap * r + Tap_coeff[1];
  // dTap *= Rcutinv;

  return dTap;
}

// For ILP, the neighbor could not contain atoms in the same layer
static __global__ void gpu_find_neighbor_ON1_ilp_nep(
  const int* nep_map,
  const int* type_map,
  const int* group_label_nep,
  void* h_parambs,
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  int* NN_nep_radial,
  int* NL_nep_radial,
  int* NN_nep_angular,
  int* NL_nep_angular,
  const int* group_label_ilp,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const float ilp_cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[group_label_nep[n1]];
    float* paramb = (float*)h_parambs + nep_id * H_PAR_OFFSET;
    float rc_radial = paramb[RCR];
    float rc_angular = paramb[RCA];
    float rc_radial_sq = rc_radial * rc_radial;
    float rc_angular_sq = rc_angular * rc_angular;
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int ilp_count_diff = 0;   // ilp neighbor in different layer to calc energy
    int nep_count_radial = 0;
    int nep_count_angular = 0;
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            // neighbors in different layers
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              float d2 = (float)x12 * (float)x12 + (float)y12 * (float)y12 + (float)z12 * (float)z12;

              if (d2 > ilp_cutoff_square) {
                continue;
              }

              bool different_layer = group_label_ilp[n1] != group_label_ilp[n2];
              if (different_layer) {
                NL[ilp_count_diff++ * N + n1] = n2;
              } else if (d2 < rc_radial_sq) {
                NL_nep_radial[nep_count_radial++ * N + n1] = n2;

                if (d2 < rc_angular_sq) {
                  NL_nep_angular[nep_count_angular++ * N + n1] = n2;
                }
              }

            }
          }
        }
      }
    }
    NN[n1] = ilp_count_diff;
    NN_nep_radial[n1] = nep_count_radial;
    NN_nep_angular[n1] = nep_count_angular;
  }
}

void find_neighbor_ilp_nep(
  const int N1,
  const int N2,
  const int* nep_map,
  const int* type_map,
  const int* group_label_nep,
  void* h_parambs,
  float rc,
  Box& box,
  const int* group_label_ilp,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL,
  GPU_Vector<int>& NN_nep_radial,
  GPU_Vector<int>& NL_nep_radial,
  GPU_Vector<int>& NN_nep_angular,
  GPU_Vector<int>& NL_nep_angular)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_ilp_nep<<<grid_size, block_size>>>(
    nep_map,
    type_map,
    group_label_nep,
    h_parambs,
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    NN_nep_radial.data(),
    NL_nep_radial.data(),
    NN_nep_angular.data(),
    NL_nep_angular.data(),
    group_label_ilp,
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list_ilp<<<N, min(1024, MN), MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
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
  bool* sublayer_flag)
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
        return;
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
    } else if (count <= MAX_ILP_NEIGHBOR_CBN) {
      for (int jj = 0; jj < count; ++jj) {
        neighsort[jj] = neighptr[jj];
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
          dnormal[id][m][ip] = 0.0;
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
          dnormal[id][m][ip] = dn1[id][ip][m] * nninv - n1[id] * dnn[ip][m] * nninv * nninv;
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
          dnormal[id][m][ip] = dn1[id][ip][m] * nninv - n1[id] * dnn[ip][m] * nninv * nninv;
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
  float Vilp;
  double TSvdw_double, TSvdwinv_double;
  float TSvdwinv_float;
  float fpair, fsum;

  r2inv = 1.0f / rsq;
  r6inv = r2inv * r2inv * r2inv;
  r8inv = r2inv * r6inv;

  // TSvdw = 1.0 + exp(-d_Seff * r + d);
  // TSvdw = 1.0f + expf(-d_Seff * r + d);
  TSvdw_double = 1.0 + exp((double) (-d_Seff * r + d));
  // TSvdwinv = 1.0f / TSvdw;
  TSvdwinv_double = 1.0 / TSvdw_double;
  TSvdwinv_float = (float) TSvdwinv_double;
  Vilp = -C_6 * r6inv * TSvdwinv_float;

  // derivatives
  // fpair = -6.0 * C_6 * r8inv * TSvdwinv + \
  //   C_6 * d_Seff * (TSvdw - 1.0) * TSvdwinv * TSvdwinv * r8inv * r;
  // fpair = (-6.0f + d_Seff * (TSvdw - 1.0f) * TSvdwinv * r ) * C_6 * TSvdwinv * r8inv;
  fpair = (-6.0f + d_Seff * (1.0f - TSvdwinv_float) * r ) * C_6 * TSvdwinv_float * r8inv;
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
  bool* sublayer_flag)
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
      
      // if (n1 == 0)
      // printf("n2[%d] dx[%.16f] dy[%.16f] dz[%.16f] r[%.16f] rsq[%.16f] rinv[%.16f] tap[%.16f] att[%.16f]\n", 
      // n2, x12d, y12d, z12d, r, rsq, rinv, Tap, p2_vdW);
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


#ifdef CODING
static __device__ void check_ann(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  float* q)
{
  printf("N_d[%d] N_n[%d] b1[%15.7e]\n", N_des, N_neu, b1[0]);
  printf("### w0\n");
  for (int n = 0; n < N_neu; ++n) {
    for (int d = 0; d < N_des; ++d) {
      printf("%15.7e\n", w0[n * N_des + d]);
    }
  }
  printf("### b0\n");
  for (int n = 0; n < N_neu; ++n) {
    printf("%15.7e\n", b0[n]);
  }
  printf("### w1\n");
  for (int n = 0; n < N_neu; ++n) {
    printf("%15.7e\n", w1[n]);
  }
  printf("### q\n");
  for (int n = 0; n < N_des; ++n) {
    printf("%15.7e\n", q[n]);
  }
}
#endif

static __global__ void find_descriptor(
  const int* nep_map,
  const int* type_map,
  const int* labels,
  void* h_parambs,
  void* h_annmbs,
  const int total_types,
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
// #ifdef USE_TABLE
//   const float* __restrict__ g_gn_radial,
//   const float* __restrict__ g_gn_angular,
// #endif
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[labels[n1]];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float* paramb = (float*)h_parambs + nep_id * H_PAR_OFFSET;
    int* paramb_int = (int*) paramb;
    int type_offset = nep_id * total_types;
    int t1 = type_map[g_type[n1] + type_offset];
    float* annmb = (float*)h_annmbs + nep_id * H_ANN_OFFSET;
    int* atomic_numbers = INT_PTR(paramb + PTRAN);
    float* c = FLT_PTR(annmb + PTRC);
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

// #ifdef USE_TABLE
//       int index_left, index_right;
//       float weight_left, weight_right;
//       find_index_and_weight(
//         d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
//       int t12 = t1 * paramb.num_types + g_type[n2];
//       for (int n = 0; n <= paramb.n_max_radial; ++n) {
//         q[n] +=
//           g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
//             weight_left +
//           g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
//             weight_right;
//       }
// #else
      float fc12;
      int t2 = type_map[g_type[n2] + type_offset];
      // float rc = paramb.rc_radial;
      float rc = paramb[RCR];
      // if (paramb.use_typewise_cutoff) {
      //   rc = min(
      //     (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
      //      COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
      //       paramb.typewise_cutoff_radial_factor,
      //     rc);
      // }
      if (paramb_int[UTC]) {
        rc = min(
          (COVALENT_RADIUS[atomic_numbers[t1]] +
           COVALENT_RADIUS[atomic_numbers[t2]]) *
            paramb[TCRF],
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      // find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      // for (int n = 0; n <= paramb.n_max_radial; ++n) {
      //   float gn12 = 0.0f;
      //   for (int k = 0; k <= paramb.basis_size_radial; ++k) {
      //     int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
      //     c_index += t1 * paramb.num_types + t2;
      //     gn12 += fn12[k] * annmb.c[c_index];
      //   }
      //   q[n] += gn12;
      // }
      find_fn(paramb_int[BSR], rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb_int[NMAXR]; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb_int[BSR]; ++k) {
          int c_index = (n * (paramb_int[BSR] + 1) + k) * paramb_int[NTS];
          c_index += t1 * paramb_int[NT] + t2;
          gn12 += fn12[k] * c[c_index];
        }
        q[n] += gn12;
      }
// #endif
    }

    // get angular descriptors
    // for (int n = 0; n <= paramb.n_max_angular; ++n) {
    for (int n = 0; n <= paramb_int[NMAXA]; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        apply_mic(box, x12double, y12double, z12double);
        float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
// #ifdef USE_TABLE
//         int index_left, index_right;
//         float weight_left, weight_right;
//         find_index_and_weight(
//           d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
//         int t12 = t1 * paramb.num_types + g_type[n2];
//         float gn12 =
//           g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
//             weight_left +
//           g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
//             weight_right;
//         accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
// #else
        float fc12;
        int t2 = type_map[g_type[n2] + type_offset];
        // float rc = paramb.rc_angular;
        float rc = paramb[RCA];
        // if (paramb.use_typewise_cutoff) {
        //   rc = min(
        //     (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
        //      COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
        //       paramb.typewise_cutoff_angular_factor,
        //     rc);
        // }
        if (paramb_int[UTC]) {
          rc = min(
            (COVALENT_RADIUS[atomic_numbers[t1]] +
             COVALENT_RADIUS[atomic_numbers[t2]]) *
              paramb[TCAF],
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        // find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        find_fn(paramb_int[BSA], rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        // for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        //   int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        //   c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        //   gn12 += fn12[k] * annmb.c[c_index];
        // }
        // accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
        for (int k = 0; k <= paramb_int[BSA]; ++k) {
          int c_index = (n * (paramb_int[BSA] + 1) + k) * paramb_int[NTS];
          c_index += t1 * paramb_int[NT] + t2 + paramb_int[NCR];
          gn12 += fn12[k] * c[c_index];
        }
        accumulate_s(paramb_int[LMAX], d12, x12, y12, z12, gn12, s);
// #endif
      }
      // find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      find_q(paramb_int[LMAX], paramb_int[NUML], paramb_int[NMAXA] + 1, n, s, q + (paramb_int[NMAXR] + 1));
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    // nomalize descriptor
    // for (int d = 0; d < annmb.dim; ++d) {
    //   q[d] = q[d] * paramb.q_scaler[d];
    // }
    float* q_scaler = FLT_PTR(paramb + PTRQS);
    int ann_dim = *((int*)annmb + ANNDIM);
    for (int d = 0; d < ann_dim; ++d) {
      q[d] = q[d] * q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};


    // if (paramb.version == 5) {
    //   apply_ann_one_layer_nep5(
    //     annmb.dim,
    //     annmb.num_neurons1,
    //     annmb.w0[t1],
    //     annmb.b0[t1],
    //     annmb.w1[t1],
    //     annmb.b1,
    //     q,
    //     F,
    //     Fp);
    // } else {
    //   apply_ann_one_layer(
    //     annmb.dim,
    //     annmb.num_neurons1,
    //     annmb.w0[t1],
    //     annmb.b0[t1],
    //     annmb.w1[t1],
    //     annmb.b1,
    //     q,
    //     F,
    //     Fp);
    //   }
    int ann_num_neurons1 = *((int*)annmb + NNEUR);
    if (paramb_int[VERSION] == 3){
      apply_ann_one_layer(
        ann_dim,
        ann_num_neurons1,
        FLT_PTR(annmb + PTRW0),
        FLT_PTR(annmb + PTRB0),
        FLT_PTR(annmb + PTRW1),
        &annmb[OUTB1],
        q,
        F,
        Fp);
    } else if (paramb_int[VERSION] == 4) {
      apply_ann_one_layer(
        ann_dim,
        ann_num_neurons1,
        FLT_PTR(annmb + PTRW0) + t1 * ann_dim * ann_num_neurons1,
        FLT_PTR(annmb + PTRB0) + t1 * ann_num_neurons1,
        FLT_PTR(annmb + PTRW1) + t1 * ann_num_neurons1,
        &annmb[OUTB1],
        q,
        F,
        Fp);
    } else if (paramb_int[VERSION] == 5) {
      apply_ann_one_layer_nep5(
        ann_dim,
        ann_num_neurons1,
        FLT_PTR(annmb + PTRW0) + t1 * ann_dim * ann_num_neurons1,
        FLT_PTR(annmb + PTRB0) + t1 * ann_num_neurons1,
        FLT_PTR(annmb + PTRW1) + t1 * (ann_num_neurons1 + 1),
        &annmb[OUTB1],
        q,
        F,
        Fp);
    }
    g_pe[n1] += F;

    // for (int d = 0; d < annmb.dim; ++d) {
    for (int d = 0; d < ann_dim; ++d) {
      // g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
      g_Fp[d * N + n1] = Fp[d] * q_scaler[d];
    }

    // set ptrs to null
    paramb = nullptr;
    paramb_int = nullptr;
    annmb = nullptr;
    atomic_numbers = nullptr;
    q_scaler = nullptr;
    c = nullptr;
  }
}

static __global__ void find_force_radial(
  const int* nep_map,
  const int* type_map,
  const int* labels,
  void* h_parambs,
  void* h_annmbs,
  const int total_types,
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
// #ifdef USE_TABLE
//   const float* __restrict__ g_gnp_radial,
// #endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[labels[n1]];
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
    float* paramb = (float*)h_parambs + nep_id * H_PAR_OFFSET;
    int* paramb_int = (int*) paramb;
    float* annmb = (float*)h_annmbs + nep_id * H_ANN_OFFSET;
    int* atomic_numbers = INT_PTR(paramb + PTRAN);
    float* c = FLT_PTR(annmb + PTRC);
    int type_offset = nep_id * total_types;
    int t1 = type_map[g_type[n1] + type_offset];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = type_map[g_type[n2] + type_offset];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
// #ifdef USE_TABLE
//       int index_left, index_right;
//       float weight_left, weight_right;
//       find_index_and_weight(
//         d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
//       int t12 = t1 * paramb.num_types + t2;
//       int t21 = t2 * paramb.num_types + t1;
//       for (int n = 0; n <= paramb.n_max_radial; ++n) {
//         float gnp12 =
//           g_gnp_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
//             weight_left +
//           g_gnp_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
//             weight_right;
//         float gnp21 =
//           g_gnp_radial[(index_left * paramb.num_types_sq + t21) * (paramb.n_max_radial + 1) + n] *
//             weight_left +
//           g_gnp_radial[(index_right * paramb.num_types_sq + t21) * (paramb.n_max_radial + 1) + n] *
//             weight_right;
//         float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
//         float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
//         for (int d = 0; d < 3; ++d) {
//           f12[d] += tmp12 * r12[d];
//           f21[d] -= tmp21 * r12[d];
//         }
//       }
// #else
      float fc12, fcp12;
      // float rc = paramb.rc_radial;
      // if (paramb.use_typewise_cutoff) {
      //   rc = min(
      //     (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
      //      COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
      //       paramb.typewise_cutoff_radial_factor,
      //     rc);
      // }
      float rc = paramb[RCR];
      if (paramb_int[UTC]) {
        rc = min(
          (COVALENT_RADIUS[atomic_numbers[t1]] +
           COVALENT_RADIUS[atomic_numbers[t2]]) *
            paramb[TCRF],
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      // find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      find_fn_and_fnp(paramb_int[BSR], rcinv, d12, fc12, fcp12, fn12, fnp12);
      // for (int n = 0; n <= paramb.n_max_radial; ++n) {
      for (int n = 0; n <= paramb_int[NMAXR]; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        // for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        //   int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        //   gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
        //   gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
        // }
        for (int k = 0; k <= paramb_int[BSR]; ++k) {
          int c_index = (n * (paramb_int[BSR] + 1) + k) * paramb_int[NTS];
          gnp12 += fnp12[k] * c[c_index + t1 * paramb_int[NT] + t2];
          gnp21 += fnp12[k] * c[c_index + t2 * paramb_int[NT] + t1];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
// #endif
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

    // set ptrs to null
    paramb = nullptr;
    paramb_int = nullptr;
    annmb = nullptr;
    atomic_numbers = nullptr;
    c = nullptr;
  }
}

static __global__ void find_partial_force_angular(
  const int* nep_map,
  const int* type_map,
  const int* labels,
  void* h_parambs,
  void* h_annmbs,
  const int total_types,
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
// #ifdef USE_TABLE
//   const float* __restrict__ g_gn_angular,
//   const float* __restrict__ g_gnp_angular,
// #endif
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int nep_id = nep_map[labels[n1]];
    float* paramb = (float*)h_parambs + nep_id * H_PAR_OFFSET;
    int* paramb_int = (int*) paramb;
    float* annmb = (float*)h_annmbs + nep_id * H_ANN_OFFSET;
    int* atomic_numbers = INT_PTR(paramb + PTRAN);
    float* c = FLT_PTR(annmb + PTRC);

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    // for (int d = 0; d < paramb.dim_angular; ++d) {
    //   Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    // }
    // for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
    //   sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    // }
    for (int d = 0; d < paramb_int[DIMA]; ++d) {
      Fp[d] = g_Fp[(paramb_int[NMAXR] + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb_int[NMAXA] + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int type_offset = nep_id * total_types;
    int t1 = type_map[g_type[n1] + type_offset];
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
// #ifdef USE_TABLE
//       int index_left, index_right;
//       float weight_left, weight_right;
//       find_index_and_weight(
//         d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
//       int t12 = t1 * paramb.num_types + g_type[n2];
//       for (int n = 0; n <= paramb.n_max_angular; ++n) {
//         int index_left_all =
//           (index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
//         int index_right_all =
//           (index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
//         float gn12 =
//           g_gn_angular[index_left_all] * weight_left + g_gn_angular[index_right_all] * weight_right;
//         float gnp12 = g_gnp_angular[index_left_all] * weight_left +
//                       g_gnp_angular[index_right_all] * weight_right;
//         accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
//       }
// #else
      float fc12, fcp12;
      int t2 = type_map[g_type[n2] + type_offset];
      // float rc = paramb.rc_angular;
      // if (paramb.use_typewise_cutoff) {
      //   rc = min(
      //     (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
      //      COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
      //       paramb.typewise_cutoff_angular_factor,
      //     rc);
      // }
      float rc = paramb[RCA];
      if (paramb_int[UTC]) {
        rc = min(
          (COVALENT_RADIUS[atomic_numbers[t1]] +
           COVALENT_RADIUS[atomic_numbers[t2]]) *
            paramb[TCAF],
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      // find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      // for (int n = 0; n <= paramb.n_max_angular; ++n) {
      //   float gn12 = 0.0f;
      //   float gnp12 = 0.0f;
      //   for (int k = 0; k <= paramb.basis_size_angular; ++k) {
      //     int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
      //     c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
      //     gn12 += fn12[k] * annmb.c[c_index];
      //     gnp12 += fnp12[k] * annmb.c[c_index];
      //   }
      //   accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      // }
      find_fn_and_fnp(paramb_int[BSA], rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb_int[NMAXA]; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb_int[BSA]; ++k) {
          int c_index = (n * (paramb_int[BSA] + 1) + k) * paramb_int[NTS];
          c_index += t1 * paramb_int[NT] + t2 + paramb_int[NCR];
          gn12 += fn12[k] * c[c_index];
          gnp12 += fnp12[k] * c[c_index];
        }
        accumulate_f12(paramb_int[LMAX], paramb_int[NUML], n, paramb_int[NMAXA] + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }
// #endif
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }

    // set ptrs to null
    paramb = nullptr;
    paramb_int = nullptr;
    annmb = nullptr;
    atomic_numbers = nullptr;
    c = nullptr;
  }
}

static __global__ void gpu_find_force_many_body_nep(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f; // virial_stress_xx
  float s_sxy = 0.0f; // virial_stress_xy
  float s_sxz = 0.0f; // virial_stress_xz
  float s_syx = 0.0f; // virial_stress_yx
  float s_syy = 0.0f; // virial_stress_yy
  float s_syz = 0.0f; // virial_stress_yz
  float s_szx = 0.0f; // virial_stress_zx
  float s_szy = 0.0f; // virial_stress_zy
  float s_szz = 0.0f; // virial_stress_zz

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = offset * number_of_particles + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
      s_sxx += x12 * f21x;
      s_syy += y12 * f21y;
      s_szz += z12 * f21z;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
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

#define BLOCK_SIZE_FORCE_NEP 64
void find_properties_many_body_nep(
  Box& box,
  int N1,
  int N2,
  const int* NN,
  const int* NL,
  const float* f12x,
  const float* f12y,
  const float* f12z,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = position_per_atom.size() / 3;
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE_NEP + 1;

  gpu_find_force_many_body_nep<<<grid_size, BLOCK_SIZE_FORCE_NEP>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    NN,
    NL,
    f12x,
    f12y,
    f12z,
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}

static __global__ void init_f12(
  const int N,
  float *g_f12x_ilp,
  float *g_f12y_ilp,
  float *g_f12z_ilp,
  float *g_f12x_ilp_neigh,
  float *g_f12y_ilp_neigh,
  float *g_f12z_ilp_neigh,
  const int *NN) 
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    #pragma unroll
    for (int i = 0; i < MAX_ILP_NEIGHBOR_TMD; ++i) {
      g_f12x_ilp_neigh[n1 + N * i] = 0.0f;
      g_f12y_ilp_neigh[n1 + N * i] = 0.0f;
      g_f12z_ilp_neigh[n1 + N * i] = 0.0f;
    }

    for (int i = 0; i < NN[n1]; ++i) {
      g_f12x_ilp[n1 + N * i] = 0.0f;
      g_f12y_ilp[n1 + N * i] = 0.0f;
      g_f12z_ilp[n1 + N * i] = 0.0f;
    }


  }
}


// nep part of compute func

// define the pure virtual func
void ILP_NEP::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom)
{
  // nothing
}

#ifdef CODING
static __global__ void ppe(double* p, int N1, int N2) {

  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    printf("----- n1[%d] p[%lf]\n", n1, p[n1]);
  }
}
#endif

// TODO
// #define CHECK_NEIGHBOR 1
#define BLOCK_SIZE_ILP 128
//#define USE_FIXED_NEIGHBOR 1
#define UPDATE_TEMP 10
#define BIG_ILP_CUTOFF_SQUARE 50.0
// find force and related quantities
void ILP_NEP::compute_ilp(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom,
  std::vector<Group> &group)
{
#ifdef CODING
  double p_ilp = 0.0;
  double p_nep[10] = {0.0};
#endif

  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_ILP + 1;

  // get labels of ILP and nep
  const int *group_label_ilp = group[ilp_group_method].label.data();
  const int *group_sublabel_ilp = group[ilp_sub_group_method].label.data();
  const int *group_label_nep = group[nep_group_method].label.data();
  int* g_nep_map = nep_map.data();
  int* g_type_map = type_map.data();
  const int total_types = type_map_cpu.size() / num_nep;

// TODO
//   GPU_Vector<double> ilp_energy(2 * number_of_atoms, 0.0);

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor_ilp_nep(
      N1,
      N2,
      g_nep_map,
      g_type_map,
      group_label_nep,
      h_parambs,
      rc,
      box,
      group_label_ilp,
      type,
      position_per_atom,
      ilp_data.cell_count,
      ilp_data.cell_count_sum,
      ilp_data.cell_contents,
      ilp_data.NN,
      ilp_data.NL,
      nep_data.NN_radial,
      nep_data.NL_radial,
      nep_data.NN_angular,
      nep_data.NL_angular);
    
  gpu_sort_neighbor_list<<<number_of_atoms, max_MN_radial, max_MN_radial * sizeof(int)>>>(
    number_of_atoms, nep_data.NN_radial.data(), nep_data.NL_radial.data());
  GPU_CHECK_KERNEL

  gpu_sort_neighbor_list<<<number_of_atoms, max_MN_angular, max_MN_angular * sizeof(int)>>>(
    number_of_atoms, nep_data.NN_angular.data(), nep_data.NL_angular.data());
  GPU_CHECK_KERNEL


    build_reduce_neighbor_list<<<grid_size, BLOCK_SIZE_ILP>>>(
      number_of_atoms,
      N1,
      N2,
      ilp_data.NN.data(),
      ilp_data.NL.data(),
      ilp_data.reduce_NL.data());
  GPU_CHECK_KERNEL
#ifdef USE_FIXED_NEIGHBOR
  }
  num_calls %= UPDATE_TEMP;
#endif

  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + number_of_atoms;
  const double* z = position_per_atom.data() + number_of_atoms * 2;
  const int *NN = ilp_data.NN.data();
  const int *NL = ilp_data.NL.data();
  int *reduce_NL = ilp_data.reduce_NL.data();
  int *ilp_NL = ilp_data.ilp_NL.data();
  int *ilp_NN = ilp_data.ilp_NN.data();

  // find ILP neighbor list
  ILP_neighbor<<<grid_size, BLOCK_SIZE_ILP>>>(
    number_of_atoms, N1, N2, box, nep_data.NN_radial.data(), nep_data.NL_radial.data(), \
    type.data(), ilp_para, x, y, z, ilp_NN, \
    ilp_NL, group_sublabel_ilp, sublayer_flag_gpu.data());
  GPU_CHECK_KERNEL


// TODO
#ifdef CHECK_NEIGHBOR
  std::vector<int> cpu_nn_ilp_diff(number_of_atoms);
  std::vector<int> cpu_nl_ilp_diff(number_of_atoms * CUDA_MAX_NL_ILP_NEP_TMD);
  std::vector<int> cpu_nn_ilp_same(number_of_atoms);
  std::vector<int> cpu_nl_ilp_same(number_of_atoms * MAX_ILP_NEIGHBOR_TMD);
  ilp_data.NN.copy_to_host(cpu_nn_ilp_diff.data());
  ilp_data.NL.copy_to_host(cpu_nl_ilp_diff.data());
  ilp_data.ilp_NN.copy_to_host(cpu_nn_ilp_same.data());
  ilp_data.ilp_NL.copy_to_host(cpu_nl_ilp_same.data());

  std::ofstream output_file_ilp_nl("ilp_neighbor_list.out", std::ios_base::app);
  output_file_ilp_nl << "different layer NL" << std::endl;
  for (int i = 0; i < number_of_atoms; ++i) {
    output_file_ilp_nl << "atom[" << i << "] " << "NN[" << cpu_nn_ilp_diff[i] << "] ";
    for (int j = 0; j < cpu_nn_ilp_diff[i]; ++j) {
      output_file_ilp_nl << cpu_nl_ilp_diff[i + j * number_of_atoms] << " ";
    }
    output_file_ilp_nl << std::endl;
  }
  output_file_ilp_nl <<std::endl;

  output_file_ilp_nl << "same layer NL" << std::endl;
  for (int i = 0; i < number_of_atoms; ++i) {
    output_file_ilp_nl << "atom[" << i << "] " << "NN[" << cpu_nn_ilp_same[i] << "] ";
    for (int j = 0; j < cpu_nn_ilp_same[i]; ++j) {
      output_file_ilp_nl << cpu_nl_ilp_same[i + j * number_of_atoms] << " ";
    }
    output_file_ilp_nl << std::endl;
  }
  output_file_ilp_nl.close();
#endif

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

  // initialize partial force
  init_f12<<<grid_size, BLOCK_SIZE_ILP>>>(
    number_of_atoms,
    g_f12x, g_f12y, g_f12z, 
    g_f12x_ilp_neigh, g_f12y_ilp_neigh, g_f12z_ilp_neigh, NN);
  GPU_CHECK_KERNEL


// TODO
#ifdef CHECK_NEIGHBOR
  std::vector<int> cpu_nn_r(number_of_atoms);
  std::vector<int> cpu_nl_r(number_of_atoms * max_MN_radial);
  std::vector<int> cpu_nn_a(number_of_atoms);
  std::vector<int> cpu_nl_a(number_of_atoms * max_MN_angular);
  nep_data.NN_radial.copy_to_host(cpu_nn_r.data());
  nep_data.NL_radial.copy_to_host(cpu_nl_r.data());
  nep_data.NN_angular.copy_to_host(cpu_nn_a.data());
  nep_data.NL_angular.copy_to_host(cpu_nl_a.data());

  std::ofstream output_file("nep_neighbor_list.out", std::ios_base::app);
  output_file << "Radial" << std::endl;
  for (int i = 0; i < number_of_atoms; ++i) {
    output_file << "atom[" << i << "] " << "NN[" << cpu_nn_r[i] << "] ";
    for (int j = 0; j < cpu_nn_r[i]; ++j) {
      output_file << cpu_nl_r[i + j * number_of_atoms] << " ";
    }
    output_file << std::endl;
  }
  output_file <<std::endl;

  output_file << "Angular" << std::endl;
  for (int i = 0; i < number_of_atoms; ++i) {
    output_file << "atom[" << i << "] " << "NN[" << cpu_nn_a[i] << "] ";
    for (int j = 0; j < cpu_nn_a[i]; ++j) {
      output_file << cpu_nl_a[i + j * number_of_atoms] << " ";
    }
    output_file << std::endl;
  }
  output_file.close();
#endif


  gpu_find_force<<<grid_size, BLOCK_SIZE_ILP>>>(
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
    g_f12z_ilp_neigh,
    sublayer_flag_gpu.data());
  GPU_CHECK_KERNEL


// TODO
//   std::vector<double> cpu_ilp_f(3 * number_of_atoms);
//   force_per_atom.copy_to_host(cpu_ilp_f.data());
//   std::ofstream output_file_ilp_f("ilp_force.out", std::ios_base::app);
//   for (int i = 0; i < number_of_atoms; ++i) {
//     output_file_ilp_f << "atom[" << i << "] " << "fx[" << std::setprecision(12) << cpu_ilp_f[i] << "] "
//     << "fy[" << cpu_ilp_f[i + number_of_atoms] << "] " << "fz[" << cpu_ilp_f[i + 2 * number_of_atoms] 
//     << "]" << std::endl;
//   }
// 
// 
//   std::vector<double> cpu_ilp_e(2 * number_of_atoms);
//   ilp_energy.copy_to_host(cpu_ilp_e.data());
//   std::ofstream output_file_ilp_e("ilp_energy.out", std::ios_base::app);
//   for (int i = 0; i < number_of_atoms; ++i) {
//     output_file_ilp_e << "atom[" << i << "] " << "att[" << std::setprecision(12) << cpu_ilp_e[i] << "] "
//     << "rep[" << cpu_ilp_e[i + number_of_atoms] << "]" << std::endl;
//   }
//   output_file_ilp_e <<std::endl;

#ifdef CODING
  std::vector<double> ilp_tmp(number_of_atoms);
  potential_per_atom.copy_to_host(ilp_tmp.data());
  for (int i = 0; i < number_of_atoms; ++i) {
    p_ilp += ilp_tmp[i];
  }

#endif


  reduce_force_many_body<<<grid_size, BLOCK_SIZE_ILP>>>(
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


// TODO
  // force_per_atom.copy_to_host(cpu_ilp_f.data());
  // output_file_ilp_f << std::endl;
  // for (int i = 0; i < number_of_atoms; ++i) {
  //   output_file_ilp_f << "atom[" << i << "] " << "fx[" << std::setprecision(12) << cpu_ilp_f[i] << "] "
  //   << "fy[" << cpu_ilp_f[i + number_of_atoms] << "] " << "fz[" << cpu_ilp_f[i + 2 * number_of_atoms] 
  //   << "]" << std::endl;
  // }
  // output_file_ilp_f.close();

  // compute NEP
  const int BLOCK_SIZE_NEP = 64;
  const int N = type.size();
  const int grid_size_nep = (N2 - N1 - 1) / BLOCK_SIZE_NEP + 1;


  static int num_calls_n = 0;
  if (num_calls_n++ % 1000 == 0) {
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
    output_file << "Neighbor info at step " << num_calls_n - 1 << ": "
                << "radial(max=" << max_MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << max_MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor<<<grid_size_nep, BLOCK_SIZE_NEP>>>(
    g_nep_map,
    g_type_map,
    group_label_nep,
    h_parambs,
    h_annmbs,
    total_types,
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
// #ifdef USE_TABLE
//     nep_data.gn_radial.data(),
//     nep_data.gn_angular.data(),
// #endif
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  find_force_radial<<<grid_size_nep, BLOCK_SIZE_NEP>>>(
    g_nep_map,
    g_type_map,
    group_label_nep,
    h_parambs,
    h_annmbs,
    total_types,
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
// #ifdef USE_TABLE
//     nep_data.gnp_radial.data(),
// #endif
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size_nep, BLOCK_SIZE_NEP>>>(
    g_nep_map,
    g_type_map,
    group_label_nep,
    h_parambs,
    h_annmbs,
    total_types,
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
// #ifdef USE_TABLE
//     nep_data.gn_angular.data(),
//     nep_data.gnp_angular.data(),
// #endif
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body_nep(
    box,
    N1,
    N2,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL

#ifdef CODING
  std::vector<double> nep_tmp(number_of_atoms);
  potential_per_atom.copy_to_host(nep_tmp.data());
  for (int i = 0; i < number_of_atoms; ++i) {
    int nep_i = nep_map_cpu[group[nep_group_method].cpu_label[i]];
    p_nep[nep_i] += nep_tmp[i] - ilp_tmp[i];
  }

  printf("\n========== OUTPUT ENERGYS FOR DEBUG ==========\n");
  printf("ilp[%.12lf]\t\t", p_ilp);
  for (int i = 0; i < num_nep; ++i) {
    printf("nep%d[%.12lf]\t\t", i, p_nep[i]);
  }
  printf("\n========== OUTPUT ENERGYS FOR DEBUG ==========\n");

#endif


}



