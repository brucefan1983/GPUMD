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


ILP_NEP::ILP_NEP(FILE* fid_ilp, FILE* fid_nep_map, int num_types, int num_atoms)
{
  // read ILP elements
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP_NEP)) {
    PRINT_INPUT_ERROR("Incorrect type number of ILP parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid_ilp, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for ILP potential.");
    printf(" %s", atom_symbol);
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
    printf("    ANN = %d-%d-1.\n", annmbs[i].dim, annmbs[i].num_neurons1);

    // calculated parameters:
    // TODO
    rc = parambs[i].rc_radial; // largest cutoff
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
    nep_data.parameters.resize(annmbs[i].num_para);
    nep_data.parameters.copy_from_host(parameters.data());
    update_potential(nep_data.parameters.data(), annmbs[i]);
    for (int d = 0; d < annmbs[i].dim; ++d) {
      tokens = get_tokens(input);
      parambs[i].q_scaler[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }

  }

  // read nep map to identify the nep for each group
  int num_nep_group = 0;
  PRINT_SCANF_ERROR(fscanf(fid_nep_map, "%d", &num_nep_group), 1, 
  "Reading error for the number of nep group.");
  nep_map.resize(num_nep_group);
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
    nep_map[i] = nep_i;
    printf("group %d uses NEP %d.\n", i, nep_i);
  }


  // initialize neighbor lists and some temp vectors
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
  CHECK(gpuMemcpyToSymbol(Tap_coeff_tmd, h_tap_coeff, 8 * sizeof(float)));

  // set ilp_flag to 1
  ilp_flag = 1;
}