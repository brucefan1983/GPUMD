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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "nep_charge.cuh"
#include "nep_charge_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP_Charge::initialize_dftd3()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  has_dftd3 = false;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "dftd3") {
        has_dftd3 = true;
        if (tokens.size() != 4) {
          std::cout << "dftd3 must have 3 parameters\n";
          exit(1);
        }
        std::string xc_functional = tokens[1];
        float rc_potential = get_double_from_token(tokens[2], __FILE__, __LINE__);
        float rc_coordination_number = get_double_from_token(tokens[3], __FILE__, __LINE__);
        dftd3.initialize(xc_functional, rc_potential, rc_coordination_number);
        break;
      }
    }
  }

  input_run.close();
}

NEP_Charge::NEP_Charge(const char* file_potential, const int num_atoms)
{
  #ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4_charge1") {
    zbl.enabled = false;
    charge_para.charge_mode = 1;
  } else if (tokens[0] == "nep4_zbl_charge1") {
    zbl.enabled = true;
    charge_para.charge_mode = 1;
  } else if (tokens[0] == "nep4_charge2") {
    zbl.enabled = false;
    charge_para.charge_mode = 2;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    zbl.enabled = true;
    charge_para.charge_mode = 2;
  } else if (tokens[0] == "nep4_charge3") {
    zbl.enabled = false;
    charge_para.charge_mode = 3;
  } else if (tokens[0] == "nep4_zbl_charge3") {
    zbl.enabled = true;
    charge_para.charge_mode = 3;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 charge models now."
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
    printf("Use the NEP4-Charge%d potential with %d atom type.\n", 
      charge_para.charge_mode, paramb.num_types);
  } else {
    printf("Use the NEP4-Charge%d potential with %d atom types.\n", 
      charge_para.charge_mode, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    paramb.atomic_numbers[n] = atomic_number - 1;
    printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be zbl rc_inner rc_outer." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      printf(
        "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
        zbl.rc_inner,
        zbl.rc_outer);
    }
  }

  // cutoff 4.2 3.7 80 47 1
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != 8) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                 "[radial_factor] [angular_factor] [zbl_factor].\n";
    exit(1);
  }
  paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
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
    paramb.typewise_cutoff_radial_factor = get_double_from_token(tokens[5], __FILE__, __LINE__);
    paramb.typewise_cutoff_angular_factor = get_double_from_token(tokens[6], __FILE__, __LINE__);
    paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[7], __FILE__, __LINE__);
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
  printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_para_ann = (annmb.dim + 3) * annmb.num_neurons1 * paramb.num_types + 2;
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
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data.parameters.resize(annmb.num_para);
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  // charge related parameters and data
  charge_para.alpha = float(PI) / paramb.rc_radial; // a good value
  charge_para.two_alpha_over_sqrt_pi = 2.0f * charge_para.alpha / sqrt(float(PI));
  charge_para.alpha_factor = 0.25f / (charge_para.alpha * charge_para.alpha);
  charge_para.A = erfc(float(PI)) / (paramb.rc_radial * paramb.rc_radial);
  charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-float(PI * PI)) / paramb.rc_radial;
  charge_para.B = - erfc(float(PI)) / paramb.rc_radial - charge_para.A * paramb.rc_radial;
  nep_data.kx.resize(charge_para.num_kpoints_max);
  nep_data.ky.resize(charge_para.num_kpoints_max);
  nep_data.kz.resize(charge_para.num_kpoints_max);
  nep_data.G.resize(charge_para.num_kpoints_max);
  nep_data.S_real.resize(charge_para.num_kpoints_max);
  nep_data.S_imag.resize(charge_para.num_kpoints_max);
  nep_data.D_real.resize(num_atoms);
  nep_data.charge.resize(num_atoms);
  nep_data.charge_derivative.resize(num_atoms * annmb.dim);

  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(num_atoms * annmb.dim);
  nep_data.sum_fxyz.resize(
    num_atoms * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cell_count.resize(num_atoms);
  nep_data.cell_count_sum.resize(num_atoms);
  nep_data.cell_contents.resize(num_atoms);
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);

#ifdef USE_TABLE
  construct_table(parameters.data());
  printf("    use tabulated radial functions to speed up.\n");
#endif

  initialize_dftd3();
}

NEP_Charge::~NEP_Charge(void)
{
  // nothing
}

void NEP_Charge::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * 2; // potential and charge
  }
  ann.sqrt_epsilon_inf = pointer;
  pointer += 1;
  ann.b1 = pointer;
  pointer += 1;

  ann.c = pointer;
}

#ifdef USE_TABLE
void NEP_Charge::construct_table(float* parameters)
{
  nep_data.gn_radial.resize(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  nep_data.gnp_radial.resize(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  nep_data.gn_angular.resize(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  nep_data.gnp_angular.resize(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  std::vector<float> gn_radial(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  std::vector<float> gnp_radial(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  std::vector<float> gn_angular(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  std::vector<float> gnp_angular(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  float* c_pointer = parameters + annmb.num_para_ann;
  construct_table_radial_or_angular(
    paramb.num_types,
    paramb.num_types_sq,
    paramb.n_max_radial,
    paramb.basis_size_radial,
    paramb.rc_radial,
    paramb.rcinv_radial,
    c_pointer,
    gn_radial.data(),
    gnp_radial.data());
  construct_table_radial_or_angular(
    paramb.num_types,
    paramb.num_types_sq,
    paramb.n_max_angular,
    paramb.basis_size_angular,
    paramb.rc_angular,
    paramb.rcinv_angular,
    c_pointer + paramb.num_c_radial,
    gn_angular.data(),
    gnp_angular.data());
  nep_data.gn_radial.copy_from_host(gn_radial.data());
  nep_data.gnp_radial.copy_from_host(gnp_radial.data());
  nep_data.gn_angular.copy_from_host(gn_angular.data());
  nep_data.gnp_angular.copy_from_host(gnp_angular.data());
}
#endif

static __global__ void find_neighbor_list_large_box(
  NEP_Charge::ParaMB paramb,
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
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
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
  float* g_charge,
  float* g_charge_derivative,
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
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    float charge = 0.0f;
    float charge_derivative[MAX_DIM] = {0.0f};

    apply_ann_one_layer_charge(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[t1],
      annmb.b0[t1],
      annmb.w1[t1],
      annmb.b1,
      q,
      F,
      Fp,
      charge,
      charge_derivative);

    g_pe[n1] += F;
    g_charge[n1] = charge;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
      g_charge_derivative[d * N + n1] = charge_derivative[d] * paramb.q_scaler[d];
    }
  }
}

static __global__ void zero_total_charge(const int N, float* g_charge)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      charge += g_charge[n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_charge[n] -= s_charge[0] / N;
    }
  }
}

static __global__ void find_force_charge_real_space_only(
  const int N,
  const NEP_Charge::Charge_Para charge_para,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  float* g_D_real)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
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
    float q1 = g_charge[n1];
    float s_pe = 0; // no self energy
    float D_real = 0; // no self energy

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float q2 = g_charge[n2];
      float qq = q1 * q2;
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * (erfc_r + charge_para.A * d12 + charge_para.B);
      s_pe += 0.5f * qq * (erfc_r + charge_para.A * d12 + charge_para.B);
      float f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 = -0.5f * K_C_SP * qq * (f2 * d12inv * d12inv - charge_para.A * d12inv);
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_D_real[n1] = K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

static __global__ void find_force_radial(
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
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
  const float* g_charge_derivative,
  const float* g_D_real,
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
        float tmp12 = (g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1]) * gnp12 * d12inv;
        float tmp21 = (g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * g_D_real[n2]) * gnp21 * d12inv;
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
        float tmp12 = (g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1]) * gnp12 * d12inv;
        float tmp21 = (g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * g_D_real[n2]) * gnp21 * d12inv;
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
  NEP_Charge::ParaMB paramb,
  NEP_Charge::ANN annmb,
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
  const float* g_charge_derivative,
  const float* g_D_real,
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
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1] 
      + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
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
        accumulate_f12(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
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
        accumulate_f12(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
#endif
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_force_ZBL(
  NEP_Charge::ParaMB paramb,
  const int N,
  const NEP_Charge::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
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
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = rc_outer * 0.5f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}

static void cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] =  a[1] * b [2] - a[2] * b [1];
  c[1] =  a[2] * b [0] - a[0] * b [2];
  c[2] =  a[0] * b [1] - a[1] * b [0];
}

static float get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

void NEP_Charge::find_k_and_G(const double* box)
{
  float a1[3] = {0.0f};
  float a2[3] = {0.0f};
  float a3[3] = {0.0f};
  float det = box[0] * (box[4] * box[8] - box[5] * box[7]) +
    box[1] * (box[5] * box[6] - box[3] * box[8]) +
    box[2] * (box[3] * box[7] - box[4] * box[6]);
  a1[0] = box[0];
  a1[1] = box[3];
  a1[2] = box[6];
  a2[0] = box[1];
  a2[1] = box[4];
  a2[2] = box[7];
  a3[0] = box[2];
  a3[1] = box[5];
  a3[2] = box[8];
  float b1[3] = {0.0f};
  float b2[3] = {0.0f};
  float b3[3] = {0.0f};
  cross_product(a2, a3, b1);
  cross_product(a3, a1, b2);
  cross_product(a1, a2, b3);

  const float two_pi = 6.2831853f;
  const float two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    b1[d] *= two_pi_over_det;
    b2[d] *= two_pi_over_det;
    b3[d] *= two_pi_over_det;
  }

  const float volume_k = two_pi * two_pi * two_pi / abs(det);
  int n1_max = charge_para.alpha * two_pi * get_area(b2, b3) / volume_k;
  int n2_max = charge_para.alpha * two_pi * get_area(b3, b1) / volume_k;
  int n3_max = charge_para.alpha * two_pi * get_area(b1, b2) / volume_k;
  float ksq_max = two_pi * two_pi * charge_para.alpha * charge_para.alpha;

  std::vector<float> cpu_kx;
  std::vector<float> cpu_ky;
  std::vector<float> cpu_kz;
  std::vector<float> cpu_G;

  std::uniform_real_distribution<double> rand_number(0.0, 1.0);
  double normalization_factor = 0.0;
  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = - n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = - n3_max; n3 <= n3_max; ++n3) {
        const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
        if (nsq > 0) {
          const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
          const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
          const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
          const float ksq = kx * kx + ky * ky + kz * kz;
          if (ksq < ksq_max) {
            double exp_factor = exp(-double(ksq) * charge_para.alpha_factor);
            normalization_factor += exp_factor;
            if (rand_number(rng) < exp_factor) {
              cpu_kx.emplace_back(kx);
              cpu_ky.emplace_back(ky);
              cpu_kz.emplace_back(kz);
              float G = abs(two_pi_over_det) / ksq;
              const float symmetry_factor = (n1 > 0) ? 2.0f : 1.0f;
              cpu_G.emplace_back(symmetry_factor * G);
            }
          }
        }
      }
    }
  }

  int num_kpoints = int(cpu_kx.size());
  for (int n = 0; n < num_kpoints; ++n) {
    cpu_G[n] *= normalization_factor / num_kpoints;
  }



  if (num_kpoints > charge_para.num_kpoints_max) {

      std::cout << "num_kpoints = " << num_kpoints << std::endl;
  std::cout << "normalization_factor = " << normalization_factor << std::endl;
  
    charge_para.num_kpoints_max = num_kpoints;
    nep_data.kx.resize(charge_para.num_kpoints_max);
    nep_data.ky.resize(charge_para.num_kpoints_max);
    nep_data.kz.resize(charge_para.num_kpoints_max);
    nep_data.G.resize(charge_para.num_kpoints_max);
    nep_data.S_real.resize(charge_para.num_kpoints_max);
    nep_data.S_imag.resize(charge_para.num_kpoints_max);
  }

  nep_data.kx.copy_from_host(cpu_kx.data(), num_kpoints);
  nep_data.ky.copy_from_host(cpu_ky.data(), num_kpoints);
  nep_data.kz.copy_from_host(cpu_kz.data(), num_kpoints);
  nep_data.G.copy_from_host(cpu_G.data(), num_kpoints);
}

static __global__ void find_structure_factor(
  const int num_kpoints_max,
  const int N1,
  const int N2,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  float* g_S_real,
  float* g_S_imag)
{
  int nk = blockIdx.x * blockDim.x + threadIdx.x;
  if (nk < num_kpoints_max) {
    float S_real = 0.0f;
    float S_imag = 0.0f;
    for (int n = N1; n < N2; ++n) {
      float kr = g_kx[nk] * float(g_x[n]) + g_ky[nk] * float(g_y[n]) + g_kz[nk] * float(g_z[n]);
      const float charge = g_charge[n];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      S_real += charge * cos_kr;
      S_imag -= charge * sin_kr;
    }
    g_S_real[nk] = S_real;
    g_S_imag[nk] = S_imag;
  }
}

static __global__ void find_force_charge_reciprocal_space(
  const int N,
  const int N1,
  const int N2,
  const int num_kpoints,
  const float alpha_factor,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    float temp_energy_sum = 0.0f;
    float temp_virial_sum[6] = {0.0f};
    float temp_force_sum[3] = {0.0f};
    float temp_D_real_sum = 0.0f;
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const float kx = g_kx[nk];
      const float ky = g_ky[nk];
      const float kz = g_kz[nk];
      const float kr = kx * g_x[n] + ky * g_y[n] + kz * g_z[n];
      const float G = g_G[nk];
      const float S_real = g_S_real[nk];
      const float S_imag = g_S_imag[nk];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
      const float GSS = G * (S_real * S_real + S_imag * S_imag);
      temp_energy_sum += GSS;
      const float alpha_k_factor = 2.0f * alpha_factor + 2.0f / (kx * kx + ky * ky + kz * kz);
      temp_virial_sum[0] += GSS * (1.0f - alpha_k_factor * kx * kx); // xx
      temp_virial_sum[1] += GSS * (1.0f - alpha_k_factor * ky * ky); // yy
      temp_virial_sum[2] += GSS * (1.0f - alpha_k_factor * kz * kz); // zz
      temp_virial_sum[3] -= GSS * (alpha_k_factor * kx * ky); // xy
      temp_virial_sum[4] -= GSS * (alpha_k_factor * ky * kz); // yz
      temp_virial_sum[5] -= GSS * (alpha_k_factor * kz * kx); // zx
      temp_D_real_sum += G * (S_real * cos_kr - S_imag * sin_kr);
      temp_force_sum[0] += kx * imag_term;
      temp_force_sum[1] += ky * imag_term;
      temp_force_sum[2] += kz * imag_term;
    }
    g_pe[n] += K_C_SP * temp_energy_sum / N;
    g_virial[n + 0 * N] += K_C_SP * temp_virial_sum[0] / N;
    g_virial[n + 1 * N] += K_C_SP * temp_virial_sum[1] / N;
    g_virial[n + 2 * N] += K_C_SP * temp_virial_sum[2] / N;
    g_virial[n + 3 * N] += K_C_SP * temp_virial_sum[3] / N;
    g_virial[n + 4 * N] += K_C_SP * temp_virial_sum[5] / N;
    g_virial[n + 5 * N] += K_C_SP * temp_virial_sum[4] / N;
    g_virial[n + 6 * N] += K_C_SP * temp_virial_sum[3] / N;
    g_virial[n + 7 * N] += K_C_SP * temp_virial_sum[5] / N;
    g_virial[n + 8 * N] += K_C_SP * temp_virial_sum[4] / N;
    g_D_real[n] = 2.0f * K_C_SP * temp_D_real_sum;
    const float charge_factor = K_C_SP * 2.0f * g_charge[n];
    g_fx[n] += charge_factor * temp_force_sum[0];
    g_fy[n] += charge_factor * temp_force_sum[1];
    g_fz[n] += charge_factor * temp_force_sum[2];
  }
}

static __global__ void find_force_charge_real_space(
  const int N,
  const NEP_Charge::Charge_Para charge_para,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  float* g_D_real)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
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
    float q1 = g_charge[n1];
    float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1; // self energy part
    float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi; // self energy part

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float q2 = g_charge[n2];
      float qq = q1 * q2;
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      s_pe += 0.5f * qq * erfc_r;
      float f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

// large box fo MD applications
void NEP_Charge::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const double rc_cell_list = 0.5 * rc;

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

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
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

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
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
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
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

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
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
    nep_data.charge.data(),
    nep_data.charge_derivative.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  // enforce charge neutrality
  zero_total_charge<<<1, 1024>>>(N, nep_data.charge.data());
  GPU_CHECK_KERNEL

  if (charge_para.charge_mode != 3) {
    find_k_and_G(box.cpu_h);
    find_structure_factor<<<(charge_para.num_kpoints_max - 1) / 64 + 1, 64>>>(
      charge_para.num_kpoints_max,
      N1,
      N2,
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.kx.data(),
      nep_data.ky.data(),
      nep_data.kz.data(),
      nep_data.S_real.data(),
      nep_data.S_imag.data());
    GPU_CHECK_KERNEL
    find_force_charge_reciprocal_space<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      charge_para.num_kpoints_max,
      charge_para.alpha_factor,
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.kx.data(),
      nep_data.ky.data(),
      nep_data.kz.data(),
      nep_data.G.data(),
      nep_data.S_real.data(),
      nep_data.S_imag.data(),
      nep_data.D_real.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }

  if (charge_para.charge_mode == 1) {
    find_force_charge_real_space<<<grid_size, BLOCK_SIZE>>>(
      N,
      charge_para,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  if (charge_para.charge_mode == 3) {
    find_force_charge_real_space_only<<<grid_size, BLOCK_SIZE>>>(
      N,
      charge_para,
      N1,
      N2,
      box,
      nep_data.NN_radial.data(),
      nep_data.NL_radial.data(),
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
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
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
#ifdef USE_TABLE
    nep_data.gnp_radial.data(),
#endif
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE>>>(
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
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
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
    false,
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP_Charge::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;
  GPU_Vector<int> NN_radial(type.size());
  GPU_Vector<int> NL_radial(size_x12);
  GPU_Vector<int> NN_angular(type.size());
  GPU_Vector<int> NL_angular(size_x12);
  GPU_Vector<float> r12(size_x12 * 6);

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    NN_radial.data(),
    NL_radial.data(),
    NN_angular.data(),
    NL_angular.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    NN_radial.copy_to_host(cpu_NN_radial.data());
    NN_angular.copy_to_host(cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < cpu_NN_radial[n]) {
        radial_actual = cpu_NN_radial[n];
      }
      if (angular_actual < cpu_NN_angular[n]) {
        angular_actual = cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    NN_radial.data(),
    NL_radial.data(),
    NN_angular.data(),
    NL_angular.data(),
    type.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
#ifdef USE_TABLE
    nep_data.gn_radial.data(),
    nep_data.gn_angular.data(),
#endif
    potential_per_atom.data(),
    nep_data.Fp.data(),
    nep_data.charge.data(),
    nep_data.charge_derivative.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  // enforce charge neutrality
  zero_total_charge<<<N, 1024>>>(N, nep_data.charge.data());
  GPU_CHECK_KERNEL

  if (charge_para.charge_mode != 3) {
    find_k_and_G(box.cpu_h);
    find_structure_factor<<<(charge_para.num_kpoints_max - 1) / 64 + 1, 64>>>(
      charge_para.num_kpoints_max,
      N1,
      N2,
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.kx.data(),
      nep_data.ky.data(),
      nep_data.kz.data(),
      nep_data.S_real.data(),
      nep_data.S_imag.data());
    GPU_CHECK_KERNEL

    find_force_charge_reciprocal_space<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      charge_para.num_kpoints_max,
      charge_para.alpha_factor,
      nep_data.charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.kx.data(),
      nep_data.ky.data(),
      nep_data.kz.data(),
      nep_data.G.data(),
      nep_data.S_real.data(),
      nep_data.S_imag.data(),
      nep_data.D_real.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }

  if (charge_para.charge_mode == 1) {
    find_force_charge_real_space_small_box<<<grid_size, BLOCK_SIZE>>>(
      N,
      charge_para,
      N1,
      N2,
      box,
      NN_radial.data(),
      NL_radial.data(),
      nep_data.charge.data(),
      r12.data(),
      r12.data() + size_x12,
      r12.data() + size_x12 * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  if (charge_para.charge_mode == 3) {
    find_force_charge_real_space_only_small_box<<<grid_size, BLOCK_SIZE>>>(
      N,
      charge_para,
      N1,
      N2,
      box,
      NN_radial.data(),
      NL_radial.data(),
      nep_data.charge.data(),
      r12.data(),
      r12.data() + size_x12,
      r12.data() + size_x12 * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    NN_radial.data(),
    NL_radial.data(),
    type.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
#ifdef USE_TABLE
    nep_data.gnp_radial.data(),
#endif
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    NN_angular.data(),
    NL_angular.data(),
    type.data(),
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.charge_derivative.data(),
    nep_data.D_real.data(),
    nep_data.sum_fxyz.data(),
#ifdef USE_TABLE
    nep_data.gn_angular.data(),
    nep_data.gnp_angular.data(),
#endif
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      NN_angular.data(),
      NL_angular.data(),
      type.data(),
      r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4,
      r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP_Charge::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * rc) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * rc) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * rc) {
    is_small_box = true;
  }

  if (is_small_box) {
    if (thickness_x > 10 * rc || thickness_y > 10 * rc || thickness_z > 10 * rc) {
      std::cout << "Error:\n"
                << "    The box has\n"
                << "        a thickness < 2.5 radial cutoffs in a periodic direction.\n"
                << "        and a thickness > 10 radial cutoffs in another direction.\n"
                << "    Please increase the periodic direction(s).\n";
      exit(1);
    }

    ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
    ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
    ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
    ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
    ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
    ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
    ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
    ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
    ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

    ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
    ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
    ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
    ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
    ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
    ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
    ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
    ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
    ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
    double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                 ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                 ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
    for (int n = 9; n < 18; n++) {
      ebox.h[n] /= det;
    }
  }

  return is_small_box;
}

void NEP_Charge::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (charge_para.charge_mode != 3) {
    if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
      PRINT_INPUT_ERROR("Cannot use non-periodic boundaries with K-space.");
    }
  }

  const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);
  if (is_small_box) {
    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  } else {
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

const GPU_Vector<int>& NEP_Charge::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP_Charge::get_NL_radial_ptr() { return nep_data.NL_radial; }

GPU_Vector<float>& NEP_Charge::get_charge_reference() { return nep_data.charge; }
