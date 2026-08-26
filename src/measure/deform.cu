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
Deform the simulation box during a run.
------------------------------------------------------------------------------*/

#include "deform.cuh"
#include "integrate/integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

static __global__ void gpu_deform_atom(
  int N,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    g_x[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_y[i] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_z[i] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
  }
}

Deform::Deform(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "deform";
}

void Deform::parse(const char** param, int num_param)
{
  printf("Deform the box.\n");

  double value;
  if (num_param > 1 && is_valid_real(param[1], &value)) {
    use_legacy_format_ = true;
    parse_legacy(param, num_param);
  } else {
    parse_general(param, num_param);
  }
}

void Deform::parse_legacy(const char** param, int num_param)
{
  if (num_param != 5 && num_param != 7) {
    PRINT_INPUT_ERROR("Keyword 'deform' should have 4 or 6 parameters.");
  }

  if (!is_valid_real(param[1], &deform_rate_[0])) {
    PRINT_INPUT_ERROR("Deform rate should be a number.");
  }

  int offset = 0;
  if (num_param == 5) {
    deform_rate_[1] = deform_rate_[0];
    deform_rate_[2] = deform_rate_[0];
    printf("    strain rate is %g A / step.\n", deform_rate_[0]);
  } else {
    offset = 2;
    if (!is_valid_real(param[2], &deform_rate_[1])) {
      PRINT_INPUT_ERROR("Deform rate should be a number.");
    }
    if (!is_valid_real(param[3], &deform_rate_[2])) {
      PRINT_INPUT_ERROR("Deform rate should be a number.");
    }
    printf(
      "    strain rates are (%g, %g, %g) A / step.\n",
      deform_rate_[0],
      deform_rate_[1],
      deform_rate_[2]);
  }

  if (!is_valid_int(param[2 + offset], &deform_component_[0])) {
    PRINT_INPUT_ERROR("deform_x should be integer.\n");
  }
  if (!is_valid_int(param[3 + offset], &deform_component_[1])) {
    PRINT_INPUT_ERROR("deform_y should be integer.\n");
  }
  if (!is_valid_int(param[4 + offset], &deform_component_[2])) {
    PRINT_INPUT_ERROR("deform_z should be integer.\n");
  }

  if (deform_component_[0]) {
    printf("    apply strain in x direction.\n");
  }
  if (deform_component_[1]) {
    printf("    apply strain in y direction.\n");
  }
  if (deform_component_[2]) {
    printf("    apply strain in z direction.\n");
  }
}

void Deform::parse_general(const char** param, int num_param)
{
  if (num_param < 3 || num_param > 13 || num_param % 2 == 0) {
    PRINT_INPUT_ERROR(
      "The general form of keyword 'deform' should contain component-rate pairs.");
  }

  const char* component_name[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};

  for (int n = 1; n < num_param; n += 2) {
    int component = -1;
    for (int d = 0; d < 6; ++d) {
      if (strcmp(param[n], component_name[d]) == 0) {
        component = d;
        break;
      }
    }

    if (component < 0) {
      PRINT_INPUT_ERROR("Deform component should be xx, yy, zz, xy, xz, or yz.");
    }
    if (deform_component_[component]) {
      PRINT_INPUT_ERROR("The same deform component cannot be specified more than once.");
    }
    if (!is_valid_real(param[n + 1], &deform_rate_[component])) {
      PRINT_INPUT_ERROR("Deform rate should be a number.");
    }

    deform_component_[component] = 1;
    printf(
      "    deform %s at %g A / step.\n", component_name[component], deform_rate_[component]);
  }
}

int Deform::get_deform_x() const
{
  return deform_component_[0];
}

int Deform::get_deform_y() const
{
  return deform_component_[1];
}

int Deform::get_deform_z() const
{
  return deform_component_[2];
}

int Deform::get_deform_xy() const
{
  return deform_component_[3];
}

int Deform::get_deform_xz() const
{
  return deform_component_[4];
}

int Deform::get_deform_yz() const
{
  return deform_component_[5];
}

void Deform::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  box.set_is_orthogonal();

  if (
    !(integrate.type >= 0 && integrate.type <= 6) && integrate.type != 11 &&
    integrate.type != 12) {
    PRINT_INPUT_ERROR(
      "The current deform implementation only supports NVE, standard NVT, NPT-Berendsen, and NPT-SCR ensembles.");
  }

  if (deform_component_[0] && box.pbc_x != 1) {
    PRINT_INPUT_ERROR("The x direction must be periodic for xx deformation.");
  }
  if (deform_component_[1] && box.pbc_y != 1) {
    PRINT_INPUT_ERROR("The y direction must be periodic for yy deformation.");
  }
  if (deform_component_[2] && box.pbc_z != 1) {
    PRINT_INPUT_ERROR("The z direction must be periodic for zz deformation.");
  }
  if (deform_component_[3] && (box.pbc_x != 1 || box.pbc_y != 1)) {
    PRINT_INPUT_ERROR("The x and y directions must be periodic for xy deformation.");
  }
  if (deform_component_[4] && (box.pbc_x != 1 || box.pbc_z != 1)) {
    PRINT_INPUT_ERROR("The x and z directions must be periodic for xz deformation.");
  }
  if (deform_component_[5] && (box.pbc_y != 1 || box.pbc_z != 1)) {
    PRINT_INPUT_ERROR("The y and z directions must be periodic for yz deformation.");
  }

  if (use_legacy_format_ && !box.is_orthogonal) {
    PRINT_INPUT_ERROR("The legacy deform format only supports orthogonal boxes.");
  }

  if (integrate.type == 11 || integrate.type == 12) {
    if (integrate.num_target_pressure_components == 1) {
      PRINT_INPUT_ERROR("Deformation cannot be combined with isotropic NPT pressure control.");
    }
    if (integrate.num_target_pressure_components == 3) {
      if (deform_component_[3] || deform_component_[4] || deform_component_[5]) {
        PRINT_INPUT_ERROR("Shear deformation with NPT requires 6 target pressure components.");
      }
      if (!box.is_orthogonal) {
        PRINT_INPUT_ERROR("Deformation with 3-component NPT requires an orthogonal box.");
      }
    } else if (integrate.num_target_pressure_components == 6 && use_legacy_format_) {
      PRINT_INPUT_ERROR(
        "The legacy deform format cannot be combined with 6-component NPT pressure control.");
    }
  }
}

void Deform::post_integrate1(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  box.set_is_orthogonal();
  const bool has_shear = deform_component_[3] || deform_component_[4] || deform_component_[5];

  if (box.is_orthogonal && !has_shear) {
    double deformation_matrix[3][3] = {0.0};
    deformation_matrix[0][0] = 1.0;
    deformation_matrix[1][1] = 1.0;
    deformation_matrix[2][2] = 1.0;

    if (deform_component_[0]) {
      deformation_matrix[0][0] = (box.cpu_h[0] + deform_rate_[0]) / box.cpu_h[0];
    }
    if (deform_component_[1]) {
      deformation_matrix[1][1] = (box.cpu_h[4] + deform_rate_[1]) / box.cpu_h[4];
    }
    if (deform_component_[2]) {
      deformation_matrix[2][2] = (box.cpu_h[8] + deform_rate_[2]) / box.cpu_h[8];
    }

    double h_old[9];
    for (int i = 0; i < 9; ++i) {
      h_old[i] = box.cpu_h[i];
    }

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        double tmp = 0.0;
        for (int k = 0; k < 3; ++k) {
          tmp += deformation_matrix[r][k] * h_old[k * 3 + c];
        }
        box.cpu_h[r * 3 + c] = tmp;
      }
    }
    box.get_inverse();

    const int number_of_atoms = atom.number_of_atoms;
    gpu_deform_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      deformation_matrix[0][0],
      deformation_matrix[0][1],
      deformation_matrix[0][2],
      deformation_matrix[1][0],
      deformation_matrix[1][1],
      deformation_matrix[1][2],
      deformation_matrix[2][0],
      deformation_matrix[2][1],
      deformation_matrix[2][2],
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + number_of_atoms,
      atom.position_per_atom.data() + number_of_atoms * 2);
    GPU_CHECK_KERNEL
    return;
  }

  box.get_inverse();

  double h_old_inverse[9];
  double h_new[9];
  for (int i = 0; i < 9; ++i) {
    h_old_inverse[i] = box.cpu_h[i + 9];
    h_new[i] = box.cpu_h[i];
  }

  const int h_index[6] = {0, 4, 8, 1, 2, 5};
  for (int d = 0; d < 6; ++d) {
    if (deform_component_[d]) {
      h_new[h_index[d]] += deform_rate_[d];
    }
  }

  double deformation_matrix[3][3] = {0.0};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        deformation_matrix[r][c] += h_new[r * 3 + k] * h_old_inverse[k * 3 + c];
      }
    }
  }

  for (int i = 0; i < 9; ++i) {
    box.cpu_h[i] = h_new[i];
  }
  box.get_inverse();
  box.set_is_orthogonal();

  const int number_of_atoms = atom.number_of_atoms;
  gpu_deform_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    deformation_matrix[0][0],
    deformation_matrix[0][1],
    deformation_matrix[0][2],
    deformation_matrix[1][0],
    deformation_matrix[1][1],
    deformation_matrix[1][2],
    deformation_matrix[2][0],
    deformation_matrix[2][1],
    deformation_matrix[2][2],
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL
}

void Deform::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
}

void Deform::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
}
