/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

#include "nep_hessian_model.cuh"
#include "utilities/common.cuh"
#include "utilities/nep_parameters.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

[[noreturn]] void fail(const std::string& message)
{
  std::cout << message << std::endl;
  exit(1);
}

} // namespace

NEP_Hessian_Model::NEP_Hessian_Model(void) {}

NEP_Hessian_Model::~NEP_Hessian_Model(void) {}

void NEP_Hessian_Model::initialize(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    fail(std::string("Failed to open ") + file_potential);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    fail("The first line of the NEP model file should have at least 3 items.");
  }

  // Model class.  The Hessian supports the same model classes as class NEP.
  if (tokens[0] == "nep4") {
    paramb.model_type = 0;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.model_type = 0;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_temperature") {
    paramb.model_type = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl_temperature") {
    paramb.model_type = 3;
    zbl.enabled = true;
  } else {
    fail(tokens[0] + " is an unsupported NEP model. We only support NEP4 models now.");
  }

  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (static_cast<int>(tokens.size()) != 2 + paramb.num_types) {
    fail("The first line of the NEP model file should have " +
         std::to_string(paramb.num_types) + " atom symbols.");
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    if (atomic_number == 0) {
      fail("Unknown element symbol " + tokens[2 + n] + " in the NEP model file.");
    }
    zbl.atomic_numbers[n] = atomic_number;
  }

  // ZBL cutoffs.
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      fail("This line should be zbl rc_inner rc_outer [zbl_factor].");
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0.0f && zbl.rc_outer == 0.0f) {
      zbl.flexible = true;
    } else if (tokens.size() == 4) {
      paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
      paramb.use_typewise_cutoff_zbl = true;
    }
  }

  // Cutoffs.
  tokens = get_tokens(input);
  if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
    fail("cutoff should have 4 or num_types * 2 + 2 parameters.");
  }
  if (tokens.size() == 5) {
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
  } else {
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
    }
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
  }
  paramb.rc_radial_max_inv = 1.0f / paramb.rc_radial_max;

  const int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  const int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);
  if (MN_radial > 819) {
    fail("The maximum number of neighbors exceeds 819. Please reduce this value.");
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));

  // n_max n_max_radial n_max_angular
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    fail("This line should be n_max n_max_radial n_max_angular.");
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // basis_size basis_size_radial basis_size_angular
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    fail("This line should be basis_size basis_size_radial basis_size_angular.");
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] ...
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    fail("This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] "
         "[has_q_123] [has_q_233] [has_q_134].");
  }
  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.num_L = paramb.L_max;
  paramb.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  if (tokens.size() >= 6) paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  if (tokens.size() >= 7) paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  if (tokens.size() >= 8) paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  if (paramb.has_q_222) paramb.num_L += 1;
  if (paramb.has_q_1111) paramb.num_L += 1;
  if (paramb.has_q_112) paramb.num_L += 1;
  if (paramb.has_q_123) paramb.num_L += 1;
  if (paramb.has_q_233) paramb.num_L += 1;
  if (paramb.has_q_134) paramb.num_L += 1;

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN num_neurons 0
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    fail("This line should be ANN num_neurons 0.");
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  if (paramb.model_type == 3) {
    annmb.dim += 1;
  }

  // Calculated parameters.
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  const int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // Neural network and descriptor parameters.
  std::vector<float> parameters(annmb.num_para + annmb.dim);
  for (int n = 0; n < annmb.num_para + annmb.dim; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // Refactor the descriptor parameters into the [type_pair][basis] layout.
  std::vector<float> descriptor_parameters = get_descriptor_parameters_type_pair(
    parameters,
    annmb.num_para_ann,
    paramb.num_types,
    paramb.n_max_radial,
    paramb.n_max_angular,
    paramb.basis_size_radial,
    paramb.basis_size_angular);

  parameters_.resize(annmb.num_para + annmb.dim);
  parameters_.copy_from_host(parameters.data());
  descriptor_parameters_type_pair_.resize(num_para_descriptor);
  descriptor_parameters_type_pair_.copy_from_host(descriptor_parameters.data());

  update_potential(parameters_.data(), annmb);
  annmb.c_type_pair = descriptor_parameters_type_pair_.data();
  annmb.q_scaler = parameters_.data() + annmb.num_para;

  // Flexible ZBL parameters.
  if (zbl.flexible) {
    const int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }
}

void NEP_Hessian_Model::update_potential(float* parameters, NEP::ANN& ann)
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
