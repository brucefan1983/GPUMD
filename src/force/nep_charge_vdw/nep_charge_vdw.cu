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
#include "nep_charge_vdw.cuh"
#include "nep_charge_vdw_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_model_file.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_vdw_parameters.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <cmath>
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

void NEP_Charge_VDW::check_ewald_pppm(const RunInput& run_input)
{
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty() && tokens[0] == "kspace") {
      if (tokens.size() < 2 || tokens.size() > 3) {
        PRINT_INPUT_ERROR("kspace requires ewald or pppm [spacing].");
      }
      if (tokens[1] == "ewald") {
        if (tokens.size() != 2) {
          PRINT_INPUT_ERROR("kspace ewald does not accept spacing.");
        }
        use_pppm = false;
      } else if (tokens[1] == "pppm") {
        use_pppm = true;
        if (
          tokens.size() == 3 && (!is_valid_real(tokens[2], &pppm_spacing) ||
                                 !(pppm_spacing >= 0.2 && pppm_spacing <= 2.0))) {
          PRINT_INPUT_ERROR("PPPM spacing must be a finite number between 0.2 and 2.0 A.");
        }
      } else {
        std::cout << "kspace method can only be ewald or pppm\n";
        exit(1);
      }
    }
  }
}

void NEP_Charge_VDW::check_need_bec(const RunInput& run_input)
{
  need_bec = false;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty()) {
      if (tokens[0] == "compute_dpdt") {
        need_bec = true;
        break;
      }

      if (tokens[0] == "dump_xyz" || tokens[0] == "dump_netcdf") {
        for (int n = 3; n < tokens.size(); ++n) {
          if (tokens[n] == "bec") {
            need_bec = true;
            break;
          }
        }
        if (need_bec) {
          break;
        }
      }

      if (tokens[0] == "add_efield") {
        if (
          tokens.size() == 4 || tokens.size() == 6 ||
          ((tokens.size() == 5 || tokens.size() == 7) && tokens.back() == "bec")) {
          need_bec = true;
          break;
        }
      }
    }
  }
}

static bool check_need_peratom_virial(const RunInput& run_input)
{
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (tokens.empty()) {
      continue;
    }
    if (
      tokens[0] == "compute_hac" || tokens[0] == "compute_hnemd" ||
      tokens[0] == "compute_hnemdec" || tokens[0] == "compute_shc" || tokens[0] == "compute_gkma" ||
      tokens[0] == "compute_hnema") {
      return true;
    }
    if (tokens[0] == "compute") {
      for (const auto& token : tokens) {
        if (token == "virial" || token == "jp") {
          return true;
        }
      }
    }
    if (tokens[0] == "dump_xyz" || tokens[0] == "dump_netcdf") {
      for (const auto& token : tokens) {
        if (token == "virial") {
          return true;
        }
      }
    }
  }
  return false;
}

NEP_Charge_VDW::NEP_Charge_VDW(
  const char* file_potential, const int num_atoms, const RunInput& run_input)
{
  for (const auto& line : run_input.lines()) {
    if (!line.tokens.empty() && line.tokens[0] == "dftd3") {
      PRINT_INPUT_ERROR("The dftd3 command is not supported for vdW NEP models.");
    }
  }

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    const std::string error = "Cannot open potential file " + std::string(file_potential) + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  NEP_Model_File model;
  std::string error;
  if (!model.read(input, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.name != "nep_vdw" && model.name != "qnep3_vdw") {
    PRINT_INPUT_ERROR("NEP_Charge_VDW requires a nep_vdw or qnep3_vdw model.");
  }
  paramb.has_charge = model.name == "qnep3_vdw";
  paramb.num_types = model.num_types;
  if (paramb.num_types > NUM_ELEMENTS) {
    PRINT_INPUT_ERROR("Too many atom types in the vdW NEP model.");
  }
  printf(
    "Use the %s potential with %d atom type%s.\n",
    model.name.c_str(),
    paramb.num_types,
    paramb.num_types == 1 ? "" : "s");

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (model.elements[n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    if (atomic_number == 0) {
      PRINT_INPUT_ERROR("vdW NEP models currently support elements from H to Pu.");
    }
    zbl.atomic_numbers[n] = atomic_number;
    annmb.c6_ref_sqrt[n] = c6_ref_sqrt[atomic_number - 1];
    printf("    type %d (%s with Z = %d).\n", n, model.elements[n].c_str(), atomic_number);
  }

  // ZBL settings and parameters are validated by the model reader.
  zbl.enabled = model.enable_zbl;
  zbl.flexible = model.flexible_zbl;
  zbl.rc_inner = model.zbl_rc_inner;
  zbl.rc_outer = model.zbl_rc_outer;
  zbl.num_types = paramb.num_types;
  paramb.use_typewise_cutoff_zbl = model.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = model.typewise_cutoff_zbl_factor;
  if (zbl.enabled) {
    if (zbl.flexible) {
      if (model.zbl_parameters.size() > sizeof(zbl.para) / sizeof(zbl.para[0])) {
        PRINT_INPUT_ERROR("Too many flexible ZBL parameters for NEP_Charge_VDW.");
      }
      std::copy(model.zbl_parameters.begin(), model.zbl_parameters.end(), zbl.para);
      printf("    has the flexible ZBL potential\n");
    } else if (paramb.use_typewise_cutoff_zbl) {
      printf(
        "    has the universal ZBL with typewise cutoff with a factor of %g.\n",
        paramb.typewise_cutoff_zbl_factor);
    } else {
      printf(
        "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
        zbl.rc_inner,
        zbl.rc_outer);
    }
  }

  // Keep each element's cutoffs and use the maxima for neighbor-list allocation.
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial_type[t] = model.rc_radial[t];
    paramb.rc_angular_type[t] = model.rc_angular[t];
    paramb.rc_radial = std::max(paramb.rc_radial, model.rc_radial[t]);
    printf(
      "    type %d: radial cutoff = %g A, angular cutoff = %g A.\n",
      t,
      model.rc_radial[t],
      model.rc_angular[t]);
  }
  if (
    model.max_radial < 1 || model.max_angular < 1 || model.max_radial > 819 ||
    model.max_angular > 819) {
    PRINT_INPUT_ERROR("MD neighbor counts must be between 1 and 819.");
  }
  printf("    MN_radial = %d.\n", model.max_radial);
  printf("    MN_angular = %d.\n", model.max_angular);
  paramb.MN_radial = int(ceil(model.max_radial * 1.25));
  paramb.MN_angular = int(ceil(model.max_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  paramb.n_max_radial = model.n_max_radial;
  paramb.n_max_angular = model.n_max_angular;
  paramb.basis_size_radial = model.basis_size_radial;
  paramb.basis_size_angular = model.basis_size_angular;
  paramb.L_max = model.L_max;
  paramb.has_q_222 = model.has_q_222;
  paramb.has_q_1111 = model.has_q_1111;
  paramb.has_q_112 = model.has_q_112;
  paramb.has_q_123 = model.has_q_123;
  paramb.has_q_233 = model.has_q_233;
  paramb.has_q_134 = model.has_q_134;
  paramb.dim_angular = model.layout.dim_angular;
  paramb.num_L = paramb.dim_angular / (paramb.n_max_angular + 1);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);
  printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  printf("    has_q_222 = %d.\n", paramb.has_q_222);
  printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  printf("    has_q_112 = %d.\n", paramb.has_q_112);
  printf("    has_q_123 = %d.\n", paramb.has_q_123);
  printf("    has_q_233 = %d.\n", paramb.has_q_233);
  printf("    has_q_134 = %d.\n", paramb.has_q_134);

  annmb.num_neurons1 = model.num_neurons1;
  annmb.dim = model.layout.dim;
  annmb.num_para_ann = model.layout.number_of_variables_ann;
  annmb.num_para = model.layout.number_of_variables;
  printf("    ANN = %d-%d-%d.\n", annmb.dim, annmb.num_neurons1, paramb.has_charge ? 3 : 2);
  printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  printf(
    "    number of descriptor parameters = %d.\n", model.layout.number_of_variables_descriptor);
  printf("    total number of parameters = %d.\n", annmb.num_para);

  rc = paramb.rc_radial;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // Native model files already store descriptor parameters in type-pair-major order.
  std::vector<float> parameters = model.parameters;
  parameters.insert(parameters.end(), model.q_scaler.begin(), model.q_scaler.end());
  nep_data.parameters.resize(parameters.size());
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;
  nep_data.energy_reference.resize(model.energy_reference.size());
  nep_data.energy_reference.copy_from_host(model.energy_reference.data());
  annmb.energy_reference = nep_data.energy_reference.data();

  // Training fixes alpha using the largest radial cutoff in the full model.
  // Removing inactive elements must not change this long-range model parameter.
  float full_rc_radial = paramb.rc_radial;
  const std::string original = get_original_nep_filename(file_potential);
  if (original != file_potential) {
    std::ifstream full_input(original);
    if (!full_input.is_open()) {
      const std::string error = "Cannot open original potential file " + original + ".";
      PRINT_INPUT_ERROR(error.c_str());
    }
    NEP_Model_File full_model;
    if (!full_model.read(full_input, error)) {
      PRINT_INPUT_ERROR(error.c_str());
    }
    full_rc_radial = *std::max_element(full_model.rc_radial.begin(), full_model.rc_radial.end());
  }

  // Both models use the same fixed alpha for dispersion.
  charge_para.alpha = float(PI) / full_rc_radial;
  // Keep alpha * spacing at or below pi / 6, independently of element compaction.
  pppm_spacing = std::fmin(1.0, full_rc_radial / 6.0);
  check_ewald_pppm(run_input);
  printf(
    "    use %s for %s.\n",
    use_pppm ? "PPPM" : "Ewald",
    paramb.has_charge ? "charge and dispersion" : "dispersion");
  if (use_pppm) {
    pppm.initialize(
      charge_para.alpha, paramb.has_charge, check_need_peratom_virial(run_input), pppm_spacing);
  } else if (paramb.has_charge) {
    ewald.initialize(charge_para.alpha);
  }
  if (paramb.has_charge) {
    check_need_bec(run_input);
    charge_para.two_alpha_over_sqrt_pi = 2.0f * charge_para.alpha / sqrt(float(PI));
    nep_data.D_real.resize(num_atoms);
    nep_data.charge.resize(num_atoms);
    nep_data.charge_derivative.resize(num_atoms * annmb.dim);
    if (need_bec) {
      nep_data.bec.resize(num_atoms * 9);
    }
  }
  nep_data.C6.resize(num_atoms);
  nep_data.C6_derivative.resize(num_atoms * annmb.dim);
  nep_data.D_C6.resize(num_atoms);

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
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);
  neighbor_manager.initialize(rc, num_atoms, paramb.MN_radial);
}

NEP_Charge_VDW::~NEP_Charge_VDW(void)
{
  // nothing
}

void NEP_Charge_VDW::update_potential(float* parameters, ANN& ann)
{
  const int num_outputs = paramb.has_charge ? 3 : 2;
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * num_outputs;
  }
  ann.sqrt_epsilon_inf = nullptr;
  if (paramb.has_charge) {
    ann.sqrt_epsilon_inf = pointer;
    pointer += 1;
  }
  ann.b1 = pointer;
  pointer += 1;

  ann.c = pointer;
}

static void cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

static float get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

void NEP_Charge_VDW::find_k_and_G(const double* box)
{
  // Alpha is fixed by the model; only a change in the physical box requires an update.
  if (num_kpoints > 0 && std::equal(box, box + 9, kpoint_box)) {
    return;
  }

  float h[9];
  std::copy(box, box + 9, h);
  const float det = h[0] * (h[4] * h[8] - h[5] * h[7]) + h[1] * (h[5] * h[6] - h[3] * h[8]) +
                    h[2] * (h[3] * h[7] - h[4] * h[6]);
  const float a1[3] = {h[0], h[3], h[6]};
  const float a2[3] = {h[1], h[4], h[7]};
  const float a3[3] = {h[2], h[5], h[8]};
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
  const float alpha = charge_para.alpha;
  const float alpha_factor = 0.25f / (alpha * alpha);
  const float volume_k = two_pi * two_pi * two_pi / std::abs(det);
  const int n1_max = alpha * two_pi * get_area(b2, b3) / volume_k;
  const int n2_max = alpha * two_pi * get_area(b3, b1) / volume_k;
  const int n3_max = alpha * two_pi * get_area(b1, b2) / volume_k;
  const float ksq_max = two_pi * two_pi * alpha * alpha;

  std::vector<float> cpu_kx;
  std::vector<float> cpu_ky;
  std::vector<float> cpu_kz;
  std::vector<float> cpu_G_vdw;
  std::vector<float> cpu_G_vdw_virial;

  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
        if (n1 == 0 && (n2 < 0 || (n2 == 0 && n3 < 0)))
          continue;
        const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
        const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
        const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
        const float ksq = kx * kx + ky * ky + kz * kz;
        if (ksq < ksq_max) {
          cpu_kx.emplace_back(kx);
          cpu_ky.emplace_back(ky);
          cpu_kz.emplace_back(kz);
          const float sqrt_pi = 1.77245385f;
          if (ksq == 0.0f) {
            // k=0 occurs once, unlike each nonzero +/-k pair.
            cpu_G_vdw.emplace_back(
              -std::abs(two_pi_over_det) * sqrt_pi * alpha * alpha * alpha / 12.0f);
            cpu_G_vdw_virial.emplace_back(0.0f);
          } else {
            const float b2 = ksq * alpha_factor;
            const float b = sqrt(b2);
            const float exp_b2 = exp(-b2);
            const float k = sqrt(ksq);
            const float erfc_b = erfc(b);
            const float prefactor = std::abs(two_pi_over_det) * sqrt_pi / 24.0f;
            const float c1 = -k * ksq * (sqrt_pi * erfc_b + (0.5f / b2 - 1.0f) * exp_b2 / b);
            const float c2 = 3.0f * k * (sqrt_pi * erfc_b - exp_b2 / b);
            cpu_G_vdw.emplace_back(prefactor * c1);
            cpu_G_vdw_virial.emplace_back(prefactor * c2);
          }
        }
      }
    }
  }

  num_kpoints = static_cast<int>(cpu_kx.size());
  if (num_kpoints > static_cast<int>(nep_data.kx.size())) {
    nep_data.kx.resize(num_kpoints);
    nep_data.ky.resize(num_kpoints);
    nep_data.kz.resize(num_kpoints);
    nep_data.G_vdw.resize(num_kpoints);
    nep_data.G_vdw_virial.resize(num_kpoints);
    nep_data.S_real.resize(num_kpoints);
    nep_data.S_imag.resize(num_kpoints);
  }
  nep_data.kx.copy_from_host(cpu_kx.data(), num_kpoints);
  nep_data.ky.copy_from_host(cpu_ky.data(), num_kpoints);
  nep_data.kz.copy_from_host(cpu_kz.data(), num_kpoints);
  nep_data.G_vdw.copy_from_host(cpu_G_vdw.data(), num_kpoints);
  nep_data.G_vdw_virial.copy_from_host(cpu_G_vdw_virial.data(), num_kpoints);
  std::copy(box, box + 9, kpoint_box);
}

static __global__ void find_neighbor_list_large_box(
  NEP_Charge_VDW::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  int t1 = g_type[n1];
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int count_radial = 0;
  int count_angular = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[n1 + N * i1];
    int t2 = g_type[n2];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    float rc_radial = (paramb.rc_radial_type[t1] + paramb.rc_radial_type[t2]) * 0.5f;
    float rc_angular = (paramb.rc_angular_type[t1] + paramb.rc_angular_type[t2]) * 0.5f;
    if (d12_square >= rc_radial * rc_radial) {
      continue;
    }
    g_NL_radial[count_radial++ * N + n1] = n2;
    if (d12_square < rc_angular * rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP_Charge_VDW::ParaMB paramb,
  NEP_Charge_VDW::ANN annmb,
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
  float* g_charge,
  float* g_charge_derivative,
  float* g_C6,
  float* g_C6_derivative,
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
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial_type[t1] + paramb.rc_radial_type[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c[c_index];
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
        float rc = (paramb.rc_angular_type[t1] + paramb.rc_angular_type[t2]) * 0.5f;
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
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max,
        paramb.has_q_222,
        paramb.has_q_1111,
        paramb.has_q_112,
        paramb.has_q_123,
        paramb.has_q_233,
        paramb.has_q_134,
        paramb.n_max_angular + 1,
        n,
        s,
        q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // normalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    float C6 = 0.0f;
    float C6_derivative[MAX_DIM] = {0.0f};

    if (paramb.has_charge) {
      float charge = 0.0f;
      float charge_derivative[MAX_DIM] = {0.0f};
      apply_ann_one_layer_charge_vdw(
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
        charge_derivative,
        C6,
        C6_derivative);
      g_charge[n1] = charge;
      for (int d = 0; d < annmb.dim; ++d) {
        g_charge_derivative[d * N + n1] = charge_derivative[d] * annmb.q_scaler[d];
      }
    } else {
      apply_ann_one_layer_vdw(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        C6,
        C6_derivative);
    }

    g_pe[n1] += double(F) + (annmb.energy_reference ? annmb.energy_reference[t1] : 0.0);
    const float SCALING_FACTOR = 0.1f;
    float C6_exp = annmb.c6_ref_sqrt[t1] * expf(C6 * SCALING_FACTOR);
    g_C6[n1] = C6_exp;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
      g_C6_derivative[d * N + n1] = C6_exp * SCALING_FACTOR * C6_derivative[d] * annmb.q_scaler[d];
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

// Chain rule correction: zero_total_charge shifted q by -mean(q),
// so D_real must be shifted by -mean(D_real) for consistent forces.
// Uses double accumulator for numerical precision.
static __global__ void zero_mean_D_real(const int N, float* g_D_real)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      sum += (double)g_D_real[n];
    }
  }
  s_sum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  float mean_D = (float)(s_sum[0] / N);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_D_real[n] -= mean_D;
    }
  }
}

static __global__ void find_bec_diagonal(const int N, const float* g_q, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_bec[n1 + N * 0] = g_q[n1];
    g_bec[n1 + N * 1] = 0.0f;
    g_bec[n1 + N * 2] = 0.0f;
    g_bec[n1 + N * 3] = 0.0f;
    g_bec[n1 + N * 4] = g_q[n1];
    g_bec[n1 + N * 5] = 0.0f;
    g_bec[n1 + N * 6] = 0.0f;
    g_bec[n1 + N * 7] = 0.0f;
    g_bec[n1 + N * 8] = g_q[n1];
  }
}

static __global__ void find_bec_radial(
  const NEP_Charge_VDW::ParaMB paramb,
  const NEP_Charge_VDW::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = (paramb.rc_radial_type[t1] + paramb.rc_radial_type[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      float bec_xx = 0.5f * (r12[0] * f12[0]);
      float bec_xy = 0.5f * (r12[0] * f12[1]);
      float bec_xz = 0.5f * (r12[0] * f12[2]);
      float bec_yx = 0.5f * (r12[1] * f12[0]);
      float bec_yy = 0.5f * (r12[1] * f12[1]);
      float bec_yz = 0.5f * (r12[1] * f12[2]);
      float bec_zx = 0.5f * (r12[2] * f12[0]);
      float bec_zy = 0.5f * (r12[2] * f12[1]);
      float bec_zz = 0.5f * (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_bec_angular(
  NEP_Charge_VDW::ParaMB paramb,
  NEP_Charge_VDW::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
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
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular_type[t1] + paramb.rc_angular_type[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222,
          paramb.has_q_1111,
          paramb.has_q_112,
          paramb.has_q_123,
          paramb.has_q_233,
          paramb.has_q_134,
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

      float bec_xx = 0.5f * (r12[0] * f12[0]);
      float bec_xy = 0.5f * (r12[0] * f12[1]);
      float bec_xz = 0.5f * (r12[0] * f12[2]);
      float bec_yx = 0.5f * (r12[1] * f12[0]);
      float bec_yy = 0.5f * (r12[1] * f12[1]);
      float bec_yz = 0.5f * (r12[1] * f12[2]);
      float bec_zx = 0.5f * (r12[2] * f12[0]);
      float bec_zy = 0.5f * (r12[2] * f12[1]);
      float bec_zz = 0.5f * (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void scale_bec(const int N, const float* sqrt_epsilon_inf, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

static __global__ void find_force_radial(
  NEP_Charge_VDW::ParaMB paramb,
  NEP_Charge_VDW::ANN annmb,
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
  const float* g_C6_derivative,
  const float* g_D_C6,
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
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;
      float rc = (paramb.rc_radial_type[t1] + paramb.rc_radial_type[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index_12 = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          int c_index_21 = get_c_index(
            t2 * paramb.num_types + t1, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index_12];
          gnp21 += fnp12[k] * annmb.c[c_index_21];
        }
        float tmp12 = g_Fp[n1 + n * N];
        if (paramb.has_charge) {
          tmp12 += g_charge_derivative[n1 + n * N] * g_D_real[n1];
        }
        tmp12 += g_C6_derivative[n1 + n * N] * g_D_C6[n1];
        float tmp21 = g_Fp[n2 + n * N];
        if (paramb.has_charge) {
          tmp21 += g_charge_derivative[n2 + n * N] * g_D_real[n2];
        }
        tmp21 += g_C6_derivative[n2 + n * N] * g_D_C6[n2];
        tmp12 *= gnp12 * d12inv;
        tmp21 *= gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
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
  NEP_Charge_VDW::ParaMB paramb,
  NEP_Charge_VDW::ANN annmb,
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
  const float* g_C6_derivative,
  const float* g_D_C6,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      float tmp = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
      if (paramb.has_charge) {
        tmp += g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
      }
      tmp += g_C6_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_C6[n1];
      Fp[d] = tmp;
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
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular_type[t1] + paramb.rc_angular_type[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222,
          paramb.has_q_1111,
          paramb.has_q_112,
          paramb.has_q_123,
          paramb.has_q_233,
          paramb.has_q_134,
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
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_force_ZBL(
  NEP_Charge_VDW::ParaMB paramb,
  const int N,
  const NEP_Charge_VDW::ZBL zbl,
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
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexible) {
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
          rc_inner = 0.0f;
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

// Remove the r=0 self interaction from the reciprocal-space charge energy.
// Its force and virial enter through D_real and the descriptor derivatives.
static __global__ void find_charge_self_energy(
  const NEP_Charge_VDW::Charge_Para charge_para,
  const int N1,
  const int N2,
  const float* g_charge,
  double* g_pe,
  float* g_D_real)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const float q1 = g_charge[n1];
    const float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1;
    const float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

static __global__ void find_structure_factor(
  const int num_kpoints,
  const int N1,
  const int N2,
  const float* g_C6,
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
  if (nk < num_kpoints) {
    float S_real = 0.0f;
    float S_imag = 0.0f;
    for (int n = N1; n < N2; ++n) {
      float kr = g_kx[nk] * float(g_x[n]) + g_ky[nk] * float(g_y[n]) + g_kz[nk] * float(g_z[n]);
      const float C6 = g_C6[n];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      S_real += C6 * cos_kr;
      S_imag -= C6 * sin_kr;
    }
    g_S_real[nk] = S_real;
    g_S_imag[nk] = S_imag;
  }
}

static __global__ void find_force_vdw_reciprocal_space(
  const int N,
  const int N1,
  const int N2,
  const int num_kpoints,
  const float alpha,
  const float* g_C6,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G_vdw,
  const float* g_G_vdw_virial,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_C6,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const float C6_1 = g_C6[n];
    const float alpha_squared = alpha * alpha;
    const float alpha_sixth = alpha_squared * alpha_squared * alpha_squared;
    double temp_energy_sum = 0.0;
    float temp_virial_sum[6] = {0.0f};
    float temp_force_sum[3] = {0.0f};
    float temp_D_C6_sum = 0.0f;
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const float kx = g_kx[nk];
      const float ky = g_ky[nk];
      const float kz = g_kz[nk];
      const float kr = kx * float(g_x[n]) + ky * float(g_y[n]) + kz * float(g_z[n]);
      const float G = g_G_vdw[nk];
      const float H = g_G_vdw_virial[nk];
      const float S_real = g_S_real[nk];
      const float S_imag = g_S_imag[nk];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
      const float SE = S_real * cos_kr - S_imag * sin_kr;
      const float GSE = G * SE;
      // Use the per-atom partition of MD Ewald; summing C6_1 * SE gives |S|^2.
      const float C6_GSE = C6_1 * GSE;
      const float C6_HSE = C6_1 * H * SE;
      temp_energy_sum += C6_GSE;
      temp_virial_sum[0] += C6_GSE - C6_HSE * kx * kx; // xx
      temp_virial_sum[1] += C6_GSE - C6_HSE * ky * ky; // yy
      temp_virial_sum[2] += C6_GSE - C6_HSE * kz * kz; // zz
      temp_virial_sum[3] -= C6_HSE * kx * ky;          // xy
      temp_virial_sum[4] -= C6_HSE * ky * kz;          // yz
      temp_virial_sum[5] -= C6_HSE * kz * kx;          // zx
      temp_D_C6_sum += GSE;
      temp_force_sum[0] += kx * imag_term;
      temp_force_sum[1] += ky * imag_term;
      temp_force_sum[2] += kz * imag_term;
    }
    // Remove the r=0 self interaction contained in the structure factor.
    g_pe[n] += temp_energy_sum + double(alpha_sixth) * C6_1 * C6_1 / 12.0;
    g_virial[n + 0 * N] += temp_virial_sum[0];
    g_virial[n + 1 * N] += temp_virial_sum[1];
    g_virial[n + 2 * N] += temp_virial_sum[2];
    g_virial[n + 3 * N] += temp_virial_sum[3];
    g_virial[n + 4 * N] += temp_virial_sum[5];
    g_virial[n + 5 * N] += temp_virial_sum[4];
    g_virial[n + 6 * N] += temp_virial_sum[3];
    g_virial[n + 7 * N] += temp_virial_sum[5];
    g_virial[n + 8 * N] += temp_virial_sum[4];
    g_D_C6[n] = 2.0f * temp_D_C6_sum + alpha_sixth * C6_1 / 6.0f;
    const float C6_factor = 2.0f * C6_1;
    g_fx[n] += C6_factor * temp_force_sum[0];
    g_fy[n] += C6_factor * temp_force_sum[1];
    g_fz[n] += C6_factor * temp_force_sum[2];
  }
}

void NEP_Charge_VDW::find_force_vdw(
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  find_structure_factor<<<(num_kpoints - 1) / 64 + 1, 64>>>(
    num_kpoints,
    N1,
    N2,
    nep_data.C6.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.kx.data(),
    nep_data.ky.data(),
    nep_data.kz.data(),
    nep_data.S_real.data(),
    nep_data.S_imag.data());
  GPU_CHECK_KERNEL

  find_force_vdw_reciprocal_space<<<(N2 - N1 - 1) / 64 + 1, 64>>>(
    N,
    N1,
    N2,
    num_kpoints,
    charge_para.alpha,
    nep_data.C6.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.kx.data(),
    nep_data.ky.data(),
    nep_data.kz.data(),
    nep_data.G_vdw.data(),
    nep_data.G_vdw_virial.data(),
    nep_data.S_real.data(),
    nep_data.S_imag.data(),
    nep_data.D_C6.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
}

void NEP_Charge_VDW::find_force_kspace(
  const int N,
  const Box& box,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (use_pppm) {
    // The local solver includes dispersion self energy and its amplitude derivative.
    pppm.find_force(
      N,
      N1,
      N2,
      box,
      nep_data.charge,
      nep_data.C6,
      position_per_atom,
      nep_data.D_real,
      nep_data.D_C6,
      force_per_atom,
      virial_per_atom,
      potential_per_atom);
  } else {
    if (paramb.has_charge) {
      ewald.find_force(
        N,
        N1,
        N2,
        box.cpu_h,
        nep_data.charge,
        position_per_atom,
        nep_data.D_real,
        force_per_atom,
        virial_per_atom,
        potential_per_atom);
    }
    find_k_and_G(box.cpu_h);
    find_force_vdw(N, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }

  if (paramb.has_charge) {
    const int BLOCK_SIZE = 64;
    const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;
    find_charge_self_energy<<<grid_size, BLOCK_SIZE>>>(
      charge_para,
      N1,
      N2,
      nep_data.charge.data(),
      potential_per_atom.data(),
      nep_data.D_real.data());
    GPU_CHECK_KERNEL

    // Chain rule correction: D_real -= mean(D_real)
    zero_mean_D_real<<<1, 1024>>>(N, nep_data.D_real.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_Charge_VDW::compute_large_box(
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

  neighbor_manager.update(box, type, position_per_atom);

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
    paramb.has_charge ? nep_data.charge.data() : nullptr,
    paramb.has_charge ? nep_data.charge_derivative.data() : nullptr,
    nep_data.C6.data(),
    nep_data.C6_derivative.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  if (paramb.has_charge) {
    // enforce charge neutrality
    zero_total_charge<<<1, 1024>>>(N, nep_data.charge.data());
    GPU_CHECK_KERNEL

    if (need_bec) {
      // get BEC (the diagonal part)
      find_bec_diagonal<<<grid_size, BLOCK_SIZE>>>(N, nep_data.charge.data(), nep_data.bec.data());
      GPU_CHECK_KERNEL

      // get BEC (radial descriptor part)
      find_bec_radial<<<grid_size, BLOCK_SIZE>>>(
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
        nep_data.charge_derivative.data(),
        nep_data.bec.data());
      GPU_CHECK_KERNEL

      // get BEC (angular descriptor part)
      find_bec_angular<<<grid_size, BLOCK_SIZE>>>(
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
        nep_data.charge_derivative.data(),
        nep_data.sum_fxyz.data(),
        nep_data.bec.data());
      GPU_CHECK_KERNEL

      scale_bec<<<grid_size, BLOCK_SIZE>>>(N, annmb.sqrt_epsilon_inf, nep_data.bec.data());
      GPU_CHECK_KERNEL
    }
  }

  find_force_kspace(N, box, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);

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
    paramb.has_charge ? nep_data.charge_derivative.data() : nullptr,
    paramb.has_charge ? nep_data.D_real.data() : nullptr,
    nep_data.C6_derivative.data(),
    nep_data.D_C6.data(),
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
    paramb.has_charge ? nep_data.charge_derivative.data() : nullptr,
    paramb.has_charge ? nep_data.D_real.data() : nullptr,
    nep_data.C6_derivative.data(),
    nep_data.D_C6.data(),
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

void NEP_Charge_VDW::compute_small_box(
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
    paramb.has_charge ? nep_data.charge.data() : nullptr,
    paramb.has_charge ? nep_data.charge_derivative.data() : nullptr,
    nep_data.C6.data(),
    nep_data.C6_derivative.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  if (paramb.has_charge) {
    // enforce charge neutrality
    zero_total_charge<<<1, 1024>>>(N, nep_data.charge.data());
    GPU_CHECK_KERNEL

    if (need_bec) {
      // get BEC (the diagonal part)
      find_bec_diagonal<<<grid_size, BLOCK_SIZE>>>(N, nep_data.charge.data(), nep_data.bec.data());
      GPU_CHECK_KERNEL

      // get BEC (radial descriptor part)
      find_bec_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
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
        nep_data.charge_derivative.data(),
        nep_data.bec.data());
      GPU_CHECK_KERNEL

      // get BEC (angular descriptor part)
      find_bec_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
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
        nep_data.charge_derivative.data(),
        nep_data.sum_fxyz.data(),
        nep_data.bec.data());
      GPU_CHECK_KERNEL

      scale_bec<<<grid_size, BLOCK_SIZE>>>(N, annmb.sqrt_epsilon_inf, nep_data.bec.data());
      GPU_CHECK_KERNEL
    }
  }

  find_force_kspace(N, box, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);

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
    paramb.has_charge ? nep_data.charge_derivative.data() : nullptr,
    paramb.has_charge ? nep_data.D_real.data() : nullptr,
    nep_data.C6_derivative.data(),
    nep_data.D_C6.data(),
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
    paramb.has_charge ? nep_data.charge_derivative.data() : nullptr,
    paramb.has_charge ? nep_data.D_real.data() : nullptr,
    nep_data.C6_derivative.data(),
    nep_data.D_C6.data(),
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

static bool get_expanded_box(const double rc, const Box& box, NEP_Charge_VDW::ExpandedBox& ebox)
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

void NEP_Charge_VDW::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
    PRINT_INPUT_ERROR("Cannot use non-periodic boundaries for vdW NEP models.");
  }

  const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);
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
}

const GPU_Vector<int>& NEP_Charge_VDW::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP_Charge_VDW::get_NL_radial_ptr() { return nep_data.NL_radial; }

GPU_Vector<float>& NEP_Charge_VDW::get_charge_reference() { return nep_data.charge; }

GPU_Vector<float>& NEP_Charge_VDW::get_bec_reference() { return nep_data.bec; }
