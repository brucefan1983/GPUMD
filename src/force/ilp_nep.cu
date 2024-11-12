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

  
  // read NEP parameter

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