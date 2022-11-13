/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
Compute the cohesive energy curve with different deformations.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "force/force.cuh"
#include "minimize/minimize.cuh"
#include "minimize/minimizer_sd.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"

static void __global__ deform_position(
  const int N,
  const D cpu_d,
  const double* old_x,
  const double* old_y,
  const double* old_z,
  double* new_x,
  double* new_y,
  double* new_z)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < N) {
    new_x[n] = cpu_d.data[0] * old_x[n] + cpu_d.data[1] * old_y[n] + cpu_d.data[2] * old_z[n];
    new_y[n] = cpu_d.data[3] * old_x[n] + cpu_d.data[4] * old_y[n] + cpu_d.data[5] * old_z[n];
    new_z[n] = cpu_d.data[6] * old_x[n] + cpu_d.data[7] * old_y[n] + cpu_d.data[8] * old_z[n];
  }
}

void Cohesive::deform_box(
  const int N, const D& cpu_d, Box& old_box, Box& new_box, GPU_Vector<double>& position_per_atom)
{
  new_box.pbc_x = old_box.pbc_x;
  new_box.pbc_y = old_box.pbc_y;
  new_box.pbc_z = old_box.pbc_z;
  new_box.triclinic = old_box.triclinic;

  if (new_box.triclinic == 0) {
    new_box.cpu_h[0] = cpu_d.data[0] * old_box.cpu_h[0];
    new_box.cpu_h[1] = cpu_d.data[4] * old_box.cpu_h[1];
    new_box.cpu_h[2] = cpu_d.data[8] * old_box.cpu_h[2];
    for (int k = 0; k < 3; ++k) {
      new_box.cpu_h[k + 3] = new_box.cpu_h[k] * 0.5;
    }
  } else {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        double tmp = 0.0f;
        for (int k = 0; k < 3; ++k) {
          tmp += cpu_d.data[r * 3 + k] * old_box.cpu_h[k * 3 + c];
        }
        new_box.cpu_h[r * 3 + c] = tmp;
      }
    }
    new_box.get_inverse();
  }

  deform_position<<<(N - 1) / 128 + 1, 128>>>(
    N, cpu_d, position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, new_position_per_atom.data(),
    new_position_per_atom.data() + N, new_position_per_atom.data() + N * 2);
}

void Cohesive::parse(const char** param, int num_param, int type)
{
  if (type == 0) {
    parse_cohesive(param, num_param);
  } else if (type == 1) {
    parse_elastic(param, num_param);
  }
}

void Cohesive::parse_cohesive(const char** param, int num_param)
{
  printf("Compute cohesive energy.\n");
  if (num_param != 4) {
    PRINT_INPUT_ERROR("compute_cohesive should have 3 parameters.\n");
  }

  if (!is_valid_real(param[1], &start_factor)) {
    PRINT_INPUT_ERROR("start_factor should be a number.\n");
  }
  if (start_factor <= 0) {
    PRINT_INPUT_ERROR("start_factor should be positive.\n");
  }
  printf("    start_factor = %g.\n", start_factor);

  if (!is_valid_real(param[2], &end_factor)) {
    PRINT_INPUT_ERROR("end_factor should be a number.\n");
  }
  if (end_factor <= start_factor) {
    PRINT_INPUT_ERROR("end_factor should > start_factor.\n");
  }
  printf("    end_factor = %g.\n", end_factor);

  if (!is_valid_int(param[3], &num_points)) {
    PRINT_INPUT_ERROR("num_points should be an integer.\n");
  }
  if (num_points < 2) {
    PRINT_INPUT_ERROR("num_points should >= 2.\n");
  }
  printf("    num_points = %d.\n", num_points);

  delta_factor = (end_factor - start_factor) / (num_points - 1);
  deformation_type = 0; // deformation for cohesive
}

void Cohesive::parse_elastic(const char** param, int num_param)
{
  printf("Compute elastic constants.\n");
  if (num_param != 3) {
    PRINT_INPUT_ERROR("compute_elastic should have 2 parameters.\n");
  }

  if (!is_valid_real(param[1], &strain)) {
    PRINT_INPUT_ERROR("strain should be a number.\n");
  }
  if (strain <= 0) {
    PRINT_INPUT_ERROR("strain should > 0.\n");
  } else if (strain > 0.1) {
    PRINT_INPUT_ERROR("strain should <= 0.1.\n");
  }
  printf("    strain = %g.\n", strain);

  if (strcmp(param[2], "cubic") == 0) {
    printf("    crystal type = cubic.\n");
    deformation_type = 1; // deformation for cubic
    num_points = 5;       // 1 (original) + 4
  } else {
    PRINT_INPUT_ERROR("Invalid crystal type.");
  }
}

void Cohesive::allocate_memory(const int num_atoms)
{
  cpu_D.resize(num_points);
  cpu_potential_total.resize(num_points);
  cpu_potential_per_atom.resize(num_atoms);
  new_position_per_atom.resize(num_atoms * 3);
}

void Cohesive::compute_D()
{
  for (int n = 0; n < num_points; ++n) {
    for (int k = 0; k < 9; ++k) {
      cpu_D[n].data[k] = 0.0;
    }
  }
  if (deformation_type == 0) {
    for (int n = 0; n < num_points; ++n) {
      const double factor = start_factor + n * delta_factor;
      cpu_D[n].data[0] = factor;
      cpu_D[n].data[4] = factor;
      cpu_D[n].data[8] = factor;
    }
  } else if (deformation_type == 1) {
    cpu_D[0].data[0] = 1.0;
    cpu_D[0].data[4] = 1.0;
    cpu_D[0].data[8] = 1.0;
    cpu_D[1].data[0] = 1.0 + strain;
    cpu_D[1].data[4] = 1.0 + strain;
    cpu_D[1].data[8] = 1.0 + strain;
    cpu_D[2].data[0] = 1.0 - strain;
    cpu_D[2].data[4] = 1.0 - strain;
    cpu_D[2].data[8] = 1.0 - strain;
    cpu_D[3].data[0] = 1.0 + strain;
    cpu_D[3].data[4] = 1.0 - strain;
    cpu_D[3].data[8] = 1.0 / (1.0 - strain * strain);
    cpu_D[4].data[0] = 1.0;
    cpu_D[4].data[1] = strain;
    cpu_D[4].data[3] = strain;
    cpu_D[4].data[4] = 1.0;
    cpu_D[4].data[8] = 1.0 / (1.0 - strain * strain);
  }
}

void Cohesive::output(char* input_dir, Box& box)
{
  char file[200];
  strcpy(file, input_dir);
  if (deformation_type == 0) {
    strcat(file, "/cohesive.out");
  } else {
    strcat(file, "/elastic.out");
  }
  FILE* fid = my_fopen(file, "w");

  if (deformation_type == 0) {
    for (int n = 0; n < num_points; ++n) {
      const double factor = start_factor + delta_factor * n;
      fprintf(fid, "%15.7e%15.7e\n", factor, cpu_potential_total[n]);
    }
    printf("Cohesive energies have been computed.\n");
  } else if (deformation_type == 1) {
    const double volume = box.get_volume();
    for (int n = 1; n < num_points; ++n) {
      cpu_potential_total[n] = (cpu_potential_total[n] - cpu_potential_total[0]) /
                               (volume * strain * strain) * PRESSURE_UNIT_CONVERSION;
    }
    double C11 =
      (cpu_potential_total[1] + cpu_potential_total[2] + 6.0 * cpu_potential_total[3]) / 9.0;
    double C12 =
      (cpu_potential_total[1] + cpu_potential_total[2] - 3.0 * cpu_potential_total[3]) / 9.0;
    double C44 = cpu_potential_total[4] * 0.5;
    printf("\nThe elastic constants are:\n");
    printf("C11 = %g Gpa\n", C11);
    printf("C12 = %g GPa\n", C12);
    printf("C44 = %g GPa\n", C44);
    fprintf(fid, "C11 = %g Gpa\n", C11);
    fprintf(fid, "C12 = %g GPa\n", C12);
    fprintf(fid, "C44 = %g GPa\n", C44);
  }

  fclose(fid);
}

void Cohesive::compute(
  char* input_dir,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  Force& force)
{
  if (deformation_type == 1 && box.triclinic == 0) {
    PRINT_INPUT_ERROR("Please use triclinic box in xyz.in to compute elastic constants.");
  }
  const int num_atoms = potential_per_atom.size();
  allocate_memory(num_atoms);
  compute_D();

  for (int n = 0; n < num_points; ++n) {
    Box new_box;
    deform_box(num_atoms, cpu_D[n], box, new_box, position_per_atom);

    Minimizer_SD minimizer(num_atoms, 1000, 1.0e-5);
    minimizer.compute(
      force, new_box, new_position_per_atom, type, group, potential_per_atom, force_per_atom,
      virial_per_atom);

    potential_per_atom.copy_to_host(cpu_potential_per_atom.data());
    cpu_potential_total[n] = 0.0;
    for (int i = 0; i < num_atoms; ++i) {
      cpu_potential_total[n] += cpu_potential_per_atom[i];
    }
  }

  output(input_dir, box);
}
