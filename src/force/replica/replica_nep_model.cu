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
NEP model of replica dynamics: a copy of the model reading and the small-box
evaluation of force/nep/nep.cu (the large-box kernels are in
replica_nep_kernels.cuh), kept separate from NEP so that changes for replica
dynamics do not affect ordinary NEP force evaluation. Keep it in sync with
force/nep/nep.cu by hand.

Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "replica_nep_model.cuh"
#include "replica_nep_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_io/nep_model_file.cuh"
#include "utilities/nep_utilities.cuh"
#include <cmath>
#include <cstddef>
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

NEP_Replica_Model::NEP_Replica_Model(const char* file_potential, const bool verbose)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    const std::string error = "Cannot open potential file " + std::string(file_potential) + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  NEP_Model_File model;
  std::string error;
  if (!model.read(input, error) || !model.validate_md(error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  if (model.name != "nep") {
    PRINT_INPUT_ERROR("Replica dynamics supports energy NEP and qNEP models only.");
  }
  paramb.num_types = model.num_types;

  if (paramb.num_types == 1) {
    if (verbose)
      printf("Use the NEP4 potential with %d atom type.\n", paramb.num_types);
  } else {
    if (verbose)
      printf("Use the NEP4 potential with %d atom types.\n", paramb.num_types);
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (model.elements[n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    if (verbose)
      printf(
        "    type %d (%s with Z = %d).\n", n, model.elements[n].c_str(), zbl.atomic_numbers[n]);
  }

  zbl.enabled = model.enable_zbl;
  zbl.flexible = model.flexible_zbl;
  paramb.use_typewise_cutoff_zbl = model.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = model.typewise_cutoff_zbl_factor;
  if (zbl.enabled) {
    zbl.rc_inner = model.zbl_rc_inner;
    zbl.rc_outer = model.zbl_rc_outer;
    if (zbl.flexible) {
      if (verbose)
        printf("    has the flexible ZBL potential\n");
    } else if (paramb.use_typewise_cutoff_zbl) {
      if (verbose)
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
    } else {
      if (verbose)
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
    }
  }

  if (model.has_multiple_cutoffs) {
    if (verbose)
      printf("    cutoff = \n");
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    paramb.rc_radial[n] = model.rc_radial[n];
    paramb.rc_angular[n] = model.rc_angular[n];
    if (model.has_multiple_cutoffs) {
      if (verbose)
        printf("    (%g A, %g A)\n", paramb.rc_radial[n], paramb.rc_angular[n]);
    }
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
  }
  if (!model.has_multiple_cutoffs) {
    if (verbose)
      printf("    radial cutoff = %g A.\n", paramb.rc_radial[0]);
    if (verbose)
      printf("    angular cutoff = %g A.\n", paramb.rc_angular[0]);
  }

  if (verbose)
    printf("    MN_radial = %d.\n", model.max_radial);
  if (verbose)
    printf("    MN_angular = %d.\n", model.max_angular);
  paramb.MN_radial = int(ceil(model.max_radial * 1.25));
  paramb.MN_angular = int(ceil(model.max_angular * 1.25));
  if (verbose)
    printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  if (verbose)
    printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  paramb.n_max_radial = model.n_max_radial;
  paramb.n_max_angular = model.n_max_angular;
  paramb.basis_size_radial = model.basis_size_radial;
  paramb.basis_size_angular = model.basis_size_angular;
  if (verbose)
    printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  if (verbose)
    printf("    n_max_angular = %d.\n", paramb.n_max_angular);
  if (verbose)
    printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  if (verbose)
    printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  paramb.L_max = model.L_max;
  paramb.has_q_222 = 2 * model.has_q_222;
  paramb.has_q_1111 = model.has_q_1111;
  paramb.has_q_112 = model.has_q_112;
  paramb.has_q_123 = model.has_q_123;
  paramb.has_q_233 = model.has_q_233;
  paramb.has_q_134 = model.has_q_134;
  paramb.num_L = paramb.L_max + model.has_q_222 + model.has_q_1111 + model.has_q_112 +
                 model.has_q_123 + model.has_q_233 + model.has_q_134;
  paramb.dim_angular = model.layout.dim_angular;
  if (verbose)
    printf("    l_max_3body = %d.\n", paramb.L_max);
  if (verbose)
    printf("    has_q_222 = %d.\n", paramb.has_q_222);
  if (verbose)
    printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  if (verbose)
    printf("    has_q_112 = %d.\n", paramb.has_q_112);
  if (verbose)
    printf("    has_q_123 = %d.\n", paramb.has_q_123);
  if (verbose)
    printf("    has_q_233 = %d.\n", paramb.has_q_233);
  if (verbose)
    printf("    has_q_134 = %d.\n", paramb.has_q_134);

  annmb.num_neurons1 = model.num_neurons1;
  annmb.dim = model.layout.dim;
  if (verbose)
    printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  rc = paramb.rc_radial_max;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
  annmb.num_para_ann = model.layout.number_of_variables_ann;
  annmb.num_para = model.layout.number_of_variables;
  const int num_para_descriptor = model.layout.number_of_variables_descriptor;
  if (verbose)
    printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  if (verbose)
    printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  if (verbose)
    printf("    total number of parameters = %d.\n", annmb.num_para);

  // Store ANN parameters, type-pair-major descriptor coefficients and scalers in one buffer.
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(model.parameters.data(), model.parameters.size());
  nep_data.parameters.copy_from_host(
    model.q_scaler.data(), model.q_scaler.size(), annmb.num_para);
  update_potential(nep_data.parameters.data(), annmb);
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;
  nep_data.energy_reference.resize(model.energy_reference.size());
  nep_data.energy_reference.copy_from_host(model.energy_reference.data());
  annmb.energy_reference = nep_data.energy_reference.data();

  if (zbl.flexible) {
    const int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      zbl.para[d] = model.zbl_parameters[d];
    }
    zbl.num_types = paramb.num_types;
  }
}

void NEP_Replica_Model::update_potential(float* parameters, ANN& ann)
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

  ann.c_type_pair = pointer;
}

void NEP_Replica_Model::find_neighbor_lists_small_box(
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

void NEP_Replica_Model::compute_small_box_from_neighbor_lists(
  const int N,
  const int N1,
  const int N2,
  const int* type,
  double* potential,
  double* force,
  double* virial,
  NEP_Replica_Data& data,
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

bool NEP_Replica_Model::get_expanded_box(const double rc, const Box& box, NEP_Replica_Model::ExpandedBox& ebox)
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

