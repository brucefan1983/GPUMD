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
Compute the cohesive energy curve with different deformations.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "force/force.cuh"
#include "minimize/minimize.cuh"
#include "minimize/minimizer_sd.cuh"
#include "model/box.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

static void __global__ deform_position(
  const int N,
  const double* old_inv,
  const double* new_h,
  const double* old_x,
  const double* old_y,
  const double* old_z,
  double* new_x,
  double* new_y,
  double* new_z)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < N) {
    double u = old_inv[0] * old_x[n] + old_inv[1] * old_y[n] + old_inv[2] * old_z[n];
    double v = old_inv[3] * old_x[n] + old_inv[4] * old_y[n] + old_inv[5] * old_z[n];
    double w = old_inv[6] * old_x[n] + old_inv[7] * old_y[n] + old_inv[8] * old_z[n];

    new_x[n] = new_h[0] * u + new_h[1] * v + new_h[2] * w;
    new_y[n] = new_h[3] * u + new_h[4] * v + new_h[5] * w;
    new_z[n] = new_h[6] * u + new_h[7] * v + new_h[8] * w;
  }
}

void Cohesive::deform_box(
  const int N,
  const D& cpu_d,
  Box& old_box,
  Box& new_box,
  GPU_Vector<double>& position_per_atom,
  const GPU_Vector<double>& old_box_inv)
{
  new_box.pbc_x = old_box.pbc_x;
  new_box.pbc_y = old_box.pbc_y;
  new_box.pbc_z = old_box.pbc_z;

  if (deformation_type == 0) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        new_box.cpu_h[r + c * 3] = cpu_d.data[r * 3 + c] * old_box.cpu_h[r + c * 3];
      }
    }
  } else {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        double tmp = 0.0f;
        for (int k = 0; k < 3; ++k) {
          tmp += cpu_d.data[r * 3 + k] * old_box.cpu_h[k * 3 + c];
        }
        new_box.cpu_h[r * 3 + c] = tmp + old_box.cpu_h[r * 3 + c];
      }
    }
  }

  new_box.get_inverse();
  new_box_h.copy_from_host(new_box.cpu_h, 9);

  deform_position<<<(N - 1) / 128 + 1, 128>>>(
    N,
    old_box_inv.data(),
    new_box_h.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    new_position_per_atom.data(),
    new_position_per_atom.data() + N,
    new_position_per_atom.data() + N * 2);
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

  if (!is_valid_int(param[3], &deform_d)) {
    PRINT_INPUT_ERROR("deform direction should be an integer.\n");
  }
  if (deform_d < 0 || deform_d > 6) {
    PRINT_INPUT_ERROR("deform direction should >=0 and <= 6.\n");
  }
  num_points = round((end_factor - start_factor) * 1000) + 1;
  printf("    num_points = %d.\n", num_points);

  const char* deform_mode[] = {"uniaxial", "uniaxial", "uniaxial", 
                               "biaxial", "biaxial", "biaxial", "triaxial"};
  const char* deform_dir[] = {"x", "y", "z", "xy", "yz", "xz", "xyz"};
  printf("    deform mode = %s - %s .\n", deform_mode[deform_d], deform_dir[deform_d]);

  delta_factor = 0.001; // (end_factor - start_factor) / (num_points - 1);
  deformation_type = 0; // deformation for cohesive
}

void Cohesive::parse_elastic(const char** param, int num_param)
{
  printf("Compute elastic constants.\n");
  if (num_param != 2) {
    PRINT_INPUT_ERROR("compute_elastic should have 1 parameter.\n");
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

  deformation_type = 1;
  num_points = 181;
}

void Cohesive::allocate_memory(const int num_atoms)
{
  cpu_D.resize(num_points);
  cpu_potential_total.resize(num_points);
  cpu_potential_per_atom.resize(num_atoms);
  new_position_per_atom.resize(num_atoms * 3);
  old_box_inv.resize(9);
  new_box_h.resize(9);
}

void Cohesive::compute_D()
{
  for (int n = 0; n < num_points; ++n) {
    for (int k = 0; k < 9; ++k) {
      cpu_D[n].data[k] = (deformation_type == 0) ? 1.0 : 0.0;
    }
  }

  if (deformation_type == 0) {
    for (int n = 0; n < num_points; ++n) {
      const double factor = start_factor + n * delta_factor;
      if (deform_d < 3) {
        for (int k = 3 * deform_d; k < 3 * deform_d + 3; ++k) {
          cpu_D[n].data[k] = factor;
        }
      } else if (deform_d > 2 && deform_d < 6) {
        for (int k = 3 * (deform_d - 3); k < 3 * (deform_d - 2) + 3; ++k) {
          int ki = k;
          if (ki > 8) {
            ki -= 9;
          }
          cpu_D[n].data[ki] = factor;
        }
      } else {
        for (int k = 0; k < 9; ++k) {
          cpu_D[n].data[k] = factor;
        }
      }
    }
  } else {
    int idx = 1;
    for (int i = 0; i < 9; ++i) {
      for (int j = i; j < 9; ++j) {
        for (int s1 : {-1, 1}) {
          for (int s2 : {-1, 1}) {
            cpu_D[idx].data[i] = s1 * strain;
            cpu_D[idx].data[j] = s2 * strain;
            idx++;
          }
        }
      }
    }
  }
}

void Cohesive::output(Box& box)
{
  FILE* fid = my_fopen(deformation_type == 0 ? "cohesive.out" : "elastic.out", "w");

  if (deformation_type == 0) {
    for (int n = 0; n < num_points; ++n) {
      const double factor = start_factor + delta_factor * n;
      fprintf(fid, "%15.7e%15.7e\n", factor, cpu_potential_total[n]);
    }
    printf("Cohesive energies have been computed.\n");
  } else {
    const double volume = box.get_volume();
    M.resize(180, std::vector<double>(81, 0.0));
    MTM.resize(81, std::vector<double>(81, 0.0));
    MTE.resize(81, 0.0);
    C81.resize(81);

    for (int n = 1; n < num_points; ++n) {
      int idx = 0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          double S_ij = cpu_D[n].data[i * 3 + j];
          for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
              double S_kl = cpu_D[n].data[k * 3 + l];
              M[n - 1][idx] = 0.5 * S_ij * S_kl;
              idx++;
            }
          }
        }
      }
    }

    for (int i = 0; i < 81; ++i) {
      for (int j = 0; j < 81; ++j) {
        for (int k = 0; k < 180; ++k) {
          MTM[i][j] += M[k][i] * M[k][j];
        }
      }
      MTM[i][i] += 1e-15;
    }

    for (int j = 0; j < 81; ++j) {
      for (int i = 0; i < 180; ++i) {
        double delta_E = cpu_potential_total[i + 1] - cpu_potential_total[0];
        MTE[j] += M[i][j] * delta_E;
      }
    }

    for (int i = 0; i < 81; ++i) {
      int max_j = i;
      double max_val = std::abs(MTM[i][i]);
      for (int j = i + 1; j < 81; ++j) {
        if (std::abs(MTM[j][i]) > max_val) {
          max_val = std::abs(MTM[j][i]);
          max_j = j;
        }
      }

      if (max_j != i) {
        std::swap(MTM[i], MTM[max_j]);
        std::swap(MTE[i], MTE[max_j]);
      }

      if (std::abs(MTM[i][i]) < 1e-20) {
        printf("Warning: Singular matrix at column %d\n", i);
        continue;
      }

      for (int k = i + 1; k < 81; ++k) {
        double factor = MTM[k][i] / MTM[i][i];
        for (int l = i; l < 81; ++l) {
          MTM[k][l] -= factor * MTM[i][l];
        }
        MTE[k] -= factor * MTE[i];
      }
    }

    for (int i = 80; i >= 0; --i) {
      double sum = MTE[i];
      for (int j = i + 1; j < 81; ++j) {
        sum -= MTM[i][j] * C81[j];
      }
      C81[i] = sum / MTM[i][i];
    }

    int voigt_idx[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {2, 0}, {0, 1}};
    for (int a = 0; a < 6; ++a) {
      int i = voigt_idx[a][0], j = voigt_idx[a][1];
      for (int b = 0; b < 6; ++b) {
        int k = voigt_idx[b][0], l = voigt_idx[b][1];
        C[a][b] = C81[27 * i + 9 * j + 3 * k + l] / volume * PRESSURE_UNIT_CONVERSION;
      }
    }

    printf("\nElastic Constants Matrix (GPa):\n");
    printf("        1         2         3         4         5         6\n");
    for (int i = 0; i < 6; i++) {
      printf("%d  ", i + 1);
      for (int j = 0; j < 6; j++) {
        printf("%8.3f  ", C[i][j]);
      }
      printf("\n");
    }

    fprintf(fid, "# Elastic Constants Matrix (GPa):\n");
    for (int i = 0; i < 6; i++) {
      for (int j = 0; j < 6; j++) {
        fprintf(fid, "%8.3f  ", C[i][j]);
      }
      fprintf(fid, "\n");
    }
    printf("Elastic Constants have been computed.\n");
  }

  fclose(fid);
}

void Cohesive::compute(
  Box& box,
  Atom& atom,
  std::vector<Group>& group,
  Force& force)
{
  allocate_memory(atom.number_of_atoms);
  compute_D();

  double old_inv[9];
  for (int i = 0; i < 9; ++i) {
    old_inv[i] = box.cpu_h[9 + i];
  }
  old_box_inv.copy_from_host(old_inv, 9);

  for (int n = 0; n < num_points; ++n) {
    Box new_box;
    deform_box(atom.number_of_atoms, cpu_D[n], box, new_box, atom.position_per_atom, old_box_inv);

    Minimizer_SD minimizer(-1, 0, atom.number_of_atoms, 1000, 1.0e-5);
    minimizer.compute(force, new_box, atom, new_position_per_atom, group);

    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom.data());
    cpu_potential_total[n] = 0.0;
    for (int i = 0; i < atom.number_of_atoms; ++i) {
      cpu_potential_total[n] += cpu_potential_per_atom[i];
    }
  }

  output(box);
}
