/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "nep.cuh"
#include "nep_kernels.cuh"
#include "nep_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/nep_io/compact_nep.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_io/nep_model_input.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/run_input.cuh"
#include <cstring>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP::initialize_dftd3(const RunInput& run_input)
{
  has_dftd3 = false;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty() && tokens[0] == "dftd3") {
      has_dftd3 = true;
      if (tokens.size() != 4) {
        std::cout << "dftd3 must have 3 parameters\n";
        exit(1);
      }
      std::string xc_functional = tokens[1];
      float rc_potential = get_double_from_token(tokens[2], __FILE__, __LINE__);
      float rc_coordination_number = get_double_from_token(tokens[3], __FILE__, __LINE__);
      dftd3.initialize(
        xc_functional,
        rc_potential,
        rc_coordination_number,
        get_first_potential_filename(run_input));
      break;
    }
  }
}

NEP::NEP(const char* file_potential, const int num_atoms, const RunInput& run_input)
  : NEP(file_potential, true)
{
  allocate_workspace(num_atoms);
  initialize_dftd3(run_input);
}

NEP::NEP(const char* file_potential, const bool verbose)
{
  NEP_Model_Input input(file_potential);

  std::vector<std::string> tokens = input.get_tokens();
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4") {
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_temperature") {
    paramb.model_type = 3;
  } else if (tokens[0] == "nep4_zbl_temperature") {
    paramb.model_type = 3;
    zbl.enabled = true;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 models now."
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
    if (verbose) printf("Use the NEP4 potential with %d atom type.\n", paramb.num_types);
  } else {
    if (verbose) printf("Use the NEP4 potential with %d atom types.\n", paramb.num_types);
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
    if (verbose) printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl
  if (zbl.enabled) {
    tokens = input.get_tokens();
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      if (paramb.model_type != 0) {
        PRINT_INPUT_ERROR("Flexible ZBL is only supported for model nep.");
      }
      zbl.flexible = true;
      if (verbose) printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        if (verbose) printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        if (verbose) printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  // cutoff
  tokens = input.get_tokens();
  if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
    std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
    exit(1);
  }
  if (tokens.size() == 5) {
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
    if (verbose) printf("    radial cutoff = %g A.\n", paramb.rc_radial[0]);
    if (verbose) printf("    angular cutoff = %g A.\n", paramb.rc_angular[0]);
  } else {
    if (verbose) printf("    cutoff = \n");
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
      if (verbose) printf("    (%g A, %g A)\n", paramb.rc_radial[n], paramb.rc_angular[n]);
    }
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
  }
  paramb.rc_radial_max_inv = 1.0f / paramb.rc_radial_max;

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);
  if (verbose) printf("    MN_radial = %d.\n", MN_radial);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  if (verbose) printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  if (verbose) printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  // n_max 10 8
  tokens = input.get_tokens();
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (verbose) printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  if (verbose) printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = input.get_tokens();
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (verbose) printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  if (verbose) printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = input.get_tokens();
  if (tokens.size() < 4) {
    std::cout << "This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] [has_q_233] [has_q_134]." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (verbose) printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  paramb.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) {
    paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  if (tokens.size() >= 6) {
    paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  }
  if (tokens.size() >= 7) {
    paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  }
  if (tokens.size() >= 8) {
    paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  }
  if (verbose) printf("    has_q_222 = %d.\n", paramb.has_q_222);
  if (verbose) printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  if (verbose) printf("    has_q_112 = %d.\n", paramb.has_q_112);
  if (verbose) printf("    has_q_123 = %d.\n", paramb.has_q_123);
  if (verbose) printf("    has_q_233 = %d.\n", paramb.has_q_233);
  if (verbose) printf("    has_q_134 = %d.\n", paramb.has_q_134);
  if (paramb.has_q_222) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_1111) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_112) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_123) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_233) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_134) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = input.get_tokens();
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
  if (verbose) printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);
  if (
    paramb.n_max_radial + 1 > MAX_NUM_N || paramb.n_max_angular + 1 > MAX_NUM_N ||
    paramb.basis_size_radial + 1 > MAX_NUM_N || paramb.basis_size_angular + 1 > MAX_NUM_N ||
    paramb.dim_angular > MAX_DIM_ANGULAR || annmb.dim > MAX_DIM) {
    PRINT_INPUT_ERROR("NEP model exceeds MAX_NUM_N, MAX_DIM_ANGULAR or MAX_DIM.");
  }

  // calculated parameters:
  rc = paramb.rc_radial_max; // largest cutoff
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  if (verbose) printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (verbose) printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;
  if (verbose) printf("    total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para + annmb.dim);
  for (int n = 0; n < annmb.num_para + annmb.dim; ++n) {
    tokens = input.get_tokens();
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  // refactor descriptor parameters memory
  std::vector<float> descriptor_parameters = input.get_descriptor_parameters_type_pair(
    parameters,
    annmb.num_para_ann,
    paramb.num_types,
    paramb.n_max_radial,
    paramb.n_max_angular,
    paramb.basis_size_radial,
    paramb.basis_size_angular);
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(parameters.data());
  nep_data.descriptor_parameters_type_pair.resize(num_para_descriptor);
  nep_data.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  annmb.c_type_pair = nep_data.descriptor_parameters_type_pair.data();
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;
  if (!input.energy_reference().empty()) {
    nep_data.energy_reference.resize(input.energy_reference().size());
    nep_data.energy_reference.copy_from_host(input.energy_reference().data());
    annmb.energy_reference = nep_data.energy_reference.data();
  }

  // flexible zbl potential parameters
  if (zbl.flexible) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = input.get_tokens();
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
}

void NEP::allocate_workspace(const int num_atoms)
{
  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  neighbor_manager.initialize(rc, num_atoms, paramb.MN_radial);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(static_cast<size_t>(num_atoms) * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(static_cast<size_t>(num_atoms) * annmb.dim);
  nep_data.sum_fxyz.resize(
    static_cast<size_t>(num_atoms) * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);
}

NEP::~NEP(void)
{
  // nothing
}

void NEP::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  pointer += 1;

  ann.c = pointer;
}

// large box fo MD applications
void NEP::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = type.size();

  neighbor_manager.update(
    box, 
    type, 
    position_per_atom);

  find_neighbor_lists_large_box(
    box,
    N,
    N1,
    N2,
    type.data(),
    position_per_atom.data(),
    neighbor_manager.get_candidate_NN().data(),
    neighbor_manager.get_candidate_NL().data(),
    nep_data,
    0);

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

  compute_large_box_from_neighbor_lists(
    box,
    N,
    N1,
    N2,
    type.data(),
    position_per_atom.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    nep_data,
    0);
}

// small box possibly used for active learning:
void NEP::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = type.size();

  find_neighbor_lists_small_box(
    box, ebox, N, N1, N2, type.data(), position_per_atom.data(), small_box_data, 0);

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
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

  compute_small_box_from_neighbor_lists(
    N,
    N1,
    N2,
    type.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    nep_data,
    small_box_data,
    0);
}

void NEP::find_neighbor_lists_small_box(
  const Box& box,
  const ExpandedBox& expanded_box,
  const int N,
  const int N1,
  const int N2,
  const int* type,
  const double* position,
  Small_Box_Data& small_box,
  gpuStream_t stream) const
{
  const int BLOCK_SIZE = 64;
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = N * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE, 0, stream>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    expanded_box,
    type,
    position,
    position + N,
    position + N * 2,
    small_box.NN_radial.data(),
    small_box.NL_radial.data(),
    small_box.NN_angular.data(),
    small_box.NL_angular.data(),
    small_box.r12.data(),
    small_box.r12.data() + size_x12,
    small_box.r12.data() + size_x12 * 2,
    small_box.r12.data() + size_x12 * 3,
    small_box.r12.data() + size_x12 * 4,
    small_box.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL
}

void NEP::compute_small_box_from_neighbor_lists(
  const int N,
  const int N1,
  const int N2,
  const int* type,
  double* potential,
  double* force,
  double* virial,
  NEP_Data& data,
  Small_Box_Data& small_box,
  gpuStream_t stream)
{
  const int BLOCK_SIZE = 64;
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = N * big_neighbor_size;

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE, 0, stream>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box.NN_radial.data(),
    small_box.NL_radial.data(),
    small_box.NN_angular.data(),
    small_box.NL_angular.data(),
    type,
    small_box.r12.data(),
    small_box.r12.data() + size_x12,
    small_box.r12.data() + size_x12 * 2,
    small_box.r12.data() + size_x12 * 3,
    small_box.r12.data() + size_x12 * 4,
    small_box.r12.data() + size_x12 * 5,
    potential,
    data.Fp.data(),
    virial,
    data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE, 0, stream>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box.NN_radial.data(),
    small_box.NL_radial.data(),
    type,
    small_box.r12.data(),
    small_box.r12.data() + size_x12,
    small_box.r12.data() + size_x12 * 2,
    data.Fp.data(),
    force,
    force + N,
    force + N * 2,
    virial);
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE, 0, stream>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box.NN_angular.data(),
    small_box.NL_angular.data(),
    type,
    small_box.r12.data() + size_x12 * 3,
    small_box.r12.data() + size_x12 * 4,
    small_box.r12.data() + size_x12 * 5,
    data.Fp.data(),
    data.sum_fxyz.data(),
    force,
    force + N,
    force + N * 2,
    virial);
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE, 0, stream>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box.NN_angular.data(),
      small_box.NL_angular.data(),
      type,
      small_box.r12.data() + size_x12 * 3,
      small_box.r12.data() + size_x12 * 4,
      small_box.r12.data() + size_x12 * 5,
      force,
      force + N,
      force + N * 2,
      virial,
      potential);
    GPU_CHECK_KERNEL
  }
}

bool NEP::get_expanded_box(const double rc, const Box& box, NEP::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * (rc + 1.0)) {
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

void NEP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial_max, box, ebox);
  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
        const int big_neighbor_size = 2000;
        const int size_x12 = current_num_atoms * big_neighbor_size;

        small_box_data.NN_radial.resize(current_num_atoms);
        small_box_data.NL_radial.resize(size_x12);
        small_box_data.NN_angular.resize(current_num_atoms);
        small_box_data.NL_angular.resize(size_x12);
        small_box_data.r12.resize(size_x12 * 6);
    }

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

static __global__ void find_descriptor(
  const float temperature,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
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
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1] = s[abc];
      }
    }

    // normalize descriptor
    q[annmb.dim - 1] = temperature;
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
    g_pe[n1] += double(F) + (annmb.energy_reference ? annmb.energy_reference[t1] : 0.0);

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

void NEP::compute_large_box(
  const float temperature,
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

  neighbor_manager.update(
    box, 
    type, 
    position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    neighbor_manager.get_candidate_NN().data(),
    neighbor_manager.get_candidate_NL().data(),
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

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    temperature,
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
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

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
    nep_data.sum_fxyz.data(),
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
void NEP::compute_small_box(
  const float temperature,
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
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
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
    temperature,
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
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
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
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
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

void NEP::compute(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial_max, box, ebox);

  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
        const int big_neighbor_size = 2000;
        const int size_x12 = current_num_atoms * big_neighbor_size;

        small_box_data.NN_radial.resize(current_num_atoms);
        small_box_data.NL_radial.resize(size_x12);
        small_box_data.NN_angular.resize(current_num_atoms);
        small_box_data.NL_angular.resize(size_x12);
        small_box_data.r12.resize(size_x12 * 6);
    }

    compute_small_box(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else {
    compute_large_box(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  }

  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

const GPU_Vector<int>& NEP::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP::get_NL_radial_ptr() { return nep_data.NL_radial; }
