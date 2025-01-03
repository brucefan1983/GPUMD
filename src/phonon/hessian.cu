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
Use finite difference to calculate the hessian (force constants).
    H_ij^ab = [F_i^a(-) - F_i^a(+)] / [u_j^b(+) - u_j^b(-)]
Then calculate the dynamical matrices with different k points.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "force/force_constant.cuh"
#include "hessian.cuh"
#include "utilities/common.cuh"
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <vector>
#include <cstring>

void Hessian::compute(
  Force& force,
  Box& box,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  initialize(type.size());
  find_H(
    force,
    box,
    cpu_position_per_atom,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

  if (num_kpoints == 1) // currently for Alex's GKMA calculations
  {
    find_D(box, cpu_position_per_atom);
    find_eigenvectors();
  } else {
    find_dispersion(box, cpu_position_per_atom);
  }
}

void Hessian::read_basis(size_t N)
{
  FILE* fid = fopen("basis.in", "r");
  size_t count;
  count = fscanf(fid, "%zu", &num_basis);
  PRINT_SCANF_ERROR(count, 1, "Reading error for basis.in.");

  basis.resize(num_basis);
  mass.resize(num_basis);
  for (size_t m = 0; m < num_basis; ++m) {
    count = fscanf(fid, "%zu%lf", &basis[m], &mass[m]);
    PRINT_SCANF_ERROR(count, 2, "Reading error for basis.in.");
  }
  label.resize(N);
  for (size_t n = 0; n < N; ++n) {
    count = fscanf(fid, "%zu", &label[n]);
    PRINT_SCANF_ERROR(count, 1, "Reading error for basis.in.");
  }
  fclose(fid);
}

void Hessian::read_kpoints()
{
  FILE* fid = fopen("kpoints.in", "r");
  size_t count;
  count = fscanf(fid, "%zu", &num_kpoints);
  PRINT_SCANF_ERROR(count, 1, "Reading error for kpoints.in.");

  kpoints.resize(num_kpoints * 3);
  for (size_t m = 0; m < num_kpoints; ++m) {
    count = fscanf(fid, "%lf%lf%lf", &kpoints[m * 3 + 0], &kpoints[m * 3 + 1], &kpoints[m * 3 + 2]);
    PRINT_SCANF_ERROR(count, 3, "Reading error for kpoints.in.");
  }
  fclose(fid);
}

void Hessian::initialize(size_t N)
{
  read_basis(N);
  read_kpoints();
  size_t num_H = num_basis * N * 9;
  size_t num_D = num_basis * num_basis * 9 * num_kpoints;
  H.resize(num_H, 0.0);
  DR.resize(num_D, 0.0);
  if (num_kpoints > 1) // for dispersion calculation
  {
    DI.resize(num_D, 0.0);
  }
}

bool Hessian::is_too_far(
  const Box& box,
  const std::vector<double>& cpu_position_per_atom,
  const size_t n1,
  const size_t n2)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;
  double x12 = cpu_position_per_atom[n2] - cpu_position_per_atom[n1];
  double y12 =
    cpu_position_per_atom[n2 + number_of_atoms] - cpu_position_per_atom[n1 + number_of_atoms];
  double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2] -
               cpu_position_per_atom[n1 + number_of_atoms * 2];
  apply_mic(box, x12, y12, z12);
  double d12_square = x12 * x12 + y12 * y12 + z12 * z12;
  return (d12_square > (cutoff * cutoff));
}

void Hessian::find_H(
  Force& force,
  Box& box,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, cpu_position_per_atom, n1, n2)) {
        continue;
      }
      size_t offset = (nb * number_of_atoms + n2) * 9;
      find_H12(
        displacement,
        n1,
        n2,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom,
        force,
        H.data() + offset);
    }
  }
}

static void find_exp_ikr(
  const size_t n1,
  const size_t n2,
  const double* k,
  const Box& box,
  const std::vector<double>& cpu_position_per_atom,
  double& cos_kr,
  double& sin_kr)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;
  double x12 = cpu_position_per_atom[n2] - cpu_position_per_atom[n1];
  double y12 =
    cpu_position_per_atom[n2 + number_of_atoms] - cpu_position_per_atom[n1 + number_of_atoms];
  double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2] -
               cpu_position_per_atom[n1 + number_of_atoms * 2];
  apply_mic(box, x12, y12, z12);
  double kr = k[0] * x12 + k[1] * y12 + k[2] * z12;
  cos_kr = cos(kr);
  sin_kr = sin(kr);
}

void Hessian::output_D()
{
  FILE* fid = fopen("D.out", "w");
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t n1 = 0; n1 < num_basis * 3; ++n1) {
      for (size_t n2 = 0; n2 < num_basis * 3; ++n2) {
        // cuSOLVER requires column-major
        fprintf(fid, "%g ", DR[offset + n1 + n2 * num_basis * 3]);
      }
      if (num_kpoints > 1) {
        for (size_t n2 = 0; n2 < num_basis * 3; ++n2) {
          // cuSOLVER requires column-major
          fprintf(fid, "%g ", DI[offset + n1 + n2 * num_basis * 3]);
        }
      }
      fprintf(fid, "\n");
    }
  }
  fclose(fid);
}

void Hessian::find_omega(FILE* fid, size_t offset)
{
  size_t dim = num_basis * 3;
  std::vector<double> W(dim);
  eig_hermitian_QR(dim, DR.data() + offset, DI.data() + offset, W.data());
  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  for (size_t n = 0; n < dim; ++n) {
    fprintf(fid, "%g ", W[n] * natural_to_THz);
  }
  fprintf(fid, "\n");
}

void Hessian::find_omega_batch(FILE* fid)
{
  size_t dim = num_basis * 3;
  std::vector<double> W(dim * num_kpoints);
  eig_hermitian_Jacobi_batch(dim, num_kpoints, DR.data(), DI.data(), W.data());
  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * dim;
    for (size_t n = 0; n < dim; ++n) {
      fprintf(fid, "%g ", W[offset + n] * natural_to_THz);
    }
    fprintf(fid, "\n");
  }
}

void Hessian::find_dispersion(const Box& box, const std::vector<double>& cpu_position_per_atom)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;

  FILE* fid_omega2 = fopen("omega2.out", "w");
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t nb = 0; nb < num_basis; ++nb) {
      size_t n1 = basis[nb];
      size_t label_1 = label[n1];
      double mass_1 = mass[label_1];
      for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
        if (is_too_far(box, cpu_position_per_atom, n1, n2))
          continue;
        double cos_kr, sin_kr;
        find_exp_ikr(n1, n2, kpoints.data() + nk * 3, box, cpu_position_per_atom, cos_kr, sin_kr);

        size_t label_2 = label[n2];
        double mass_2 = mass[label_2];
        double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
        double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
        for (size_t a = 0; a < 3; ++a) {
          for (size_t b = 0; b < 3; ++b) {
            size_t a3b = a * 3 + b;
            size_t row = label_1 * 3 + a;
            size_t col = label_2 * 3 + b;
            // cuSOLVER requires column-major
            size_t index = offset + col * num_basis * 3 + row;
            DR[index] += H12[a3b] * cos_kr * mass_factor;
            DI[index] += H12[a3b] * sin_kr * mass_factor;
          }
        }
      }
    }
    if (num_basis > 10) {
      find_omega(fid_omega2, offset);
    } // > 32x32
  }
  output_D();
  if (num_basis <= 10) {
    find_omega_batch(fid_omega2);
  } // <= 32x32
  fclose(fid_omega2);
}

void Hessian::find_D(const Box& box, std::vector<double>& cpu_position_per_atom)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    size_t label_1 = label[n1];
    double mass_1 = mass[label_1];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, cpu_position_per_atom, n1, n2)) {
        continue;
      }

      size_t label_2 = label[n2];
      double mass_2 = mass[label_2];
      double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
      double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
      for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
          size_t a3b = a * 3 + b;
          size_t row = label_1 * 3 + a;
          size_t col = label_2 * 3 + b;
          // cuSOLVER requires column-major
          size_t index = col * num_basis * 3 + row;
          DR[index] += H12[a3b] * mass_factor;
        }
      }
    }
  }
}

void Hessian::find_eigenvectors()
{
  std::ofstream eigfile;
  eigfile.open("eigenvector.out", std::ios::out | std::ios::binary);

  size_t dim = num_basis * 3;
  std::vector<double> W(dim);
  std::vector<double> eigenvectors(dim * dim);
  eigenvectors_symmetric_Jacobi(dim, DR.data(), W.data(), eigenvectors.data());

  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);

  // output eigenvalues
  float om2;
  for (size_t n = 0; n < dim; n++) {
    om2 = (float)(W[n] * natural_to_THz);
    eigfile.write((char*)&om2, sizeof(float));
  }

  // output eigenvectors
  float eig;
  for (size_t col = 0; col < dim; col++) {
    for (size_t a = 0; a < 3; a++) {
      for (size_t b = 0; b < num_basis; b++) {
        size_t row = a + b * 3;
        // column-major order from cuSolver
        eig = (float)eigenvectors[row + col * dim];
        eigfile.write((char*)&eig, sizeof(float));
      }
    }
  }
  eigfile.close();
}

void Hessian::parse(const char** param, size_t num_param)
{
  if (num_param != 3) {
    PRINT_INPUT_ERROR("compute_phonon should have 2 parameters.\n");
  }
  // cutoff
  if (!is_valid_real(param[1], &cutoff)) {
    PRINT_INPUT_ERROR("cutoff for compute_phonon should be a number.\n");
  }
  if (cutoff <= 0) {
    PRINT_INPUT_ERROR("cutoff for compute_phonon should be positive.\n");
  }
  printf("Cutoff distance for compute_phonon = %g A.\n", cutoff);

  // displacement
  if (!is_valid_real(param[2], &displacement)) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be a number.\n");
  }
  if (displacement <= 0) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be positive.\n");
  }
  printf("displacement for compute_phonon = %g A.\n", displacement);
}
