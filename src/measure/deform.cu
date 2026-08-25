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

static __global__ void gpu_deform_orthogonal(
  const int number_of_atoms,
  const double scale_factor_x,
  const double scale_factor_y,
  const double scale_factor_z,
  double* x,
  double* y,
  double* z)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    x[i] *= scale_factor_x;
    y[i] *= scale_factor_y;
    z[i] *= scale_factor_z;
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

  if (num_param != 5 && num_param != 7) {
    PRINT_INPUT_ERROR("Keyword 'deform' should have 4 or 6 parameters.");
  }

  if (!is_valid_real(param[1], &deform_rate_[0])) {
    PRINT_INPUT_ERROR("Defrom rate should be a number.");
  }

  int offset = 0;
  if (num_param == 5) {
    deform_rate_[1] = deform_rate_[0];
    deform_rate_[2] = deform_rate_[0];
    printf("    strain rate is %g A / step.\n", deform_rate_[0]);
  } else {
    offset = 2;
    if (!is_valid_real(param[2], &deform_rate_[1])) {
      PRINT_INPUT_ERROR("Defrom rate should be a number.");
    }
    if (!is_valid_real(param[3], &deform_rate_[2])) {
      PRINT_INPUT_ERROR("Defrom rate should be a number.");
    }
    printf(
      "    strain rates are (%g, %g, %g) A / step.\n",
      deform_rate_[0],
      deform_rate_[1],
      deform_rate_[2]);
  }

  if (!is_valid_int(param[2 + offset], &deform_x_)) {
    PRINT_INPUT_ERROR("deform_x should be integer.\n");
  }
  if (!is_valid_int(param[3 + offset], &deform_y_)) {
    PRINT_INPUT_ERROR("deform_y should be integer.\n");
  }
  if (!is_valid_int(param[4 + offset], &deform_z_)) {
    PRINT_INPUT_ERROR("deform_z should be integer.\n");
  }

  if (deform_x_) {
    printf("    apply strain in x direction.\n");
  }
  if (deform_y_) {
    printf("    apply strain in y direction.\n");
  }
  if (deform_z_) {
    printf("    apply strain in z direction.\n");
  }
}

int Deform::get_deform_x() const
{
  return deform_x_;
}

int Deform::get_deform_y() const
{
  return deform_y_;
}

int Deform::get_deform_z() const
{
  return deform_z_;
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
  if (!box.is_orthogonal) {
    PRINT_INPUT_ERROR("The current deform implementation only supports orthogonal boxes.");
  }

  if (!(integrate.type >= 0 && integrate.type <= 6) && integrate.type != 11 && integrate.type != 12) {
    PRINT_INPUT_ERROR(
      "The current deform implementation only supports NVE, standard NVT, NPT-Berendsen, and NPT-SCR ensembles.");
  }

  if ((integrate.type == 11 || integrate.type == 12) && integrate.num_target_pressure_components != 3) {
    PRINT_INPUT_ERROR(
      "The current deform implementation can only be combined with orthogonal NPT pressure control.");
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
  double scale_factor[3] = {1.0, 1.0, 1.0};

  if (deform_x_) {
    scale_factor[0] = (box.cpu_h[0] + deform_rate_[0]) / box.cpu_h[0];
    box.cpu_h[0] *= scale_factor[0];
  }
  if (deform_y_) {
    scale_factor[1] = (box.cpu_h[4] + deform_rate_[1]) / box.cpu_h[4];
    box.cpu_h[4] *= scale_factor[1];
  }
  if (deform_z_) {
    scale_factor[2] = (box.cpu_h[8] + deform_rate_[2]) / box.cpu_h[8];
    box.cpu_h[8] *= scale_factor[2];
  }

  box.get_inverse();

  const int number_of_atoms = atom.number_of_atoms;
  gpu_deform_orthogonal<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    scale_factor[0],
    scale_factor[1],
    scale_factor[2],
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
