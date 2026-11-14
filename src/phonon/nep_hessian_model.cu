/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

#include "nep_hessian_model.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_io/nep_model_file.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

namespace {
// Atomic numbers are needed only for the ZBL pair potential.
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

} // namespace

void NEP_Hessian_Model::initialize(const char* file_potential)
{
  paramb = NEP::ParaMB{};
  annmb = NEP::ANN{};
  zbl = NEP::ZBL{};
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR((std::string("Cannot open potential file ") + file_potential).c_str());
  }
  NEP_Model_File model;
  std::string error;
  if (!model.read(input, error) || !model.validate_md(error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.name != "nep" && model.name != "nep_temperature") {
    PRINT_INPUT_ERROR("Analytic Hessian requires model nep or nep_temperature.");
  }
  paramb.model_type = model.name == "nep_temperature" ? 3 : 0;
  paramb.num_types = model.num_types;
  paramb.num_types_sq = model.num_types * model.num_types;
  paramb.rc_radial_max = 0.0f;
  for (int t = 0; t < model.num_types; ++t) {
    paramb.rc_radial[t] = model.rc_radial[t];
    paramb.rc_angular[t] = model.rc_angular[t];
    paramb.rc_radial_max = std::max(paramb.rc_radial_max, model.rc_radial[t]);
    const auto element = std::find(ELEMENTS, ELEMENTS + NUM_ELEMENTS, model.elements[t]);
    if (element == ELEMENTS + NUM_ELEMENTS) {
      PRINT_INPUT_ERROR(("Unknown NEP element: " + model.elements[t]).c_str());
    }
    zbl.atomic_numbers[t] = static_cast<int>(element - ELEMENTS) + 1;
  }
  paramb.rc_radial_max_inv = 1.0f / paramb.rc_radial_max;
  paramb.MN_radial = static_cast<int>(std::ceil(model.max_radial * 1.25));
  paramb.MN_angular = static_cast<int>(std::ceil(model.max_angular * 1.25));
  paramb.n_max_radial = model.n_max_radial;
  paramb.n_max_angular = model.n_max_angular;
  paramb.basis_size_radial = model.basis_size_radial;
  paramb.basis_size_angular = model.basis_size_angular;
  paramb.L_max = model.L_max;
  paramb.has_q_222 = 2 * model.has_q_222;
  paramb.has_q_1111 = model.has_q_1111;
  paramb.has_q_112 = model.has_q_112;
  paramb.has_q_123 = model.has_q_123;
  paramb.has_q_233 = model.has_q_233;
  paramb.has_q_134 = model.has_q_134;
  paramb.num_L = model.L_max + model.has_q_222 + model.has_q_1111 +
    model.has_q_112 + model.has_q_123 + model.has_q_233 + model.has_q_134;
  paramb.dim_angular = model.layout.dim_angular;
  paramb.num_c_radial = paramb.num_types_sq * (model.n_max_radial + 1) *
    (model.basis_size_radial + 1);

  zbl.enabled = model.enable_zbl;
  zbl.flexible = model.flexible_zbl;
  zbl.num_types = model.num_types;
  zbl.rc_inner = model.zbl_rc_inner;
  zbl.rc_outer = model.zbl_rc_outer;
  paramb.use_typewise_cutoff_zbl = model.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = model.typewise_cutoff_zbl_factor;
  std::copy(model.zbl_parameters.begin(), model.zbl_parameters.end(), zbl.para);

  annmb.dim = model.layout.dim;
  annmb.num_neurons1 = model.num_neurons1;
  annmb.num_para_ann = model.layout.number_of_variables_ann;
  annmb.num_para = model.layout.number_of_variables;
  parameters_.resize(model.parameters.size() + model.q_scaler.size());
  parameters_.copy_from_host(model.parameters.data(), model.parameters.size());
  parameters_.copy_from_host(model.q_scaler.data(), model.q_scaler.size(), annmb.num_para);
  float* pointer = parameters_.data();
  for (int t = 0; t < model.num_types; ++t) {
    annmb.w0[t] = pointer;
    pointer += annmb.num_neurons1 * annmb.dim;
    annmb.b0[t] = pointer;
    pointer += annmb.num_neurons1;
    annmb.w1[t] = pointer;
    pointer += annmb.num_neurons1;
  }
  annmb.b1 = pointer++;
  annmb.c_type_pair = pointer;
  annmb.q_scaler = parameters_.data() + annmb.num_para;
  // Element energy references are constants in the coordinates. Energy and
  // forces are obtained from Force; only ANN derivatives are evaluated here.
}
