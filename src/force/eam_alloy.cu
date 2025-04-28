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

#include "eam_alloy.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#define BLOCK_SIZE_FORCE 64

class CubicSpline
{
private:
  double x0;
  double h;
  int num_intervals;
  std::vector<double> a, b, c, d;

  static bool thomas_algorithm(
    const std::vector<double>& lower,
    const std::vector<double>& main_diag,
    const std::vector<double>& upper,
    const std::vector<double>& rhs,
    std::vector<double>& solution)
  {
    int n = main_diag.size();
    if (n == 0 || lower.size() != n - 1 || upper.size() != n - 1 || rhs.size() != n) {
      return false;
    }

    std::vector<double> new_main(n);
    std::vector<double> new_rhs(n);
    new_main[0] = main_diag[0];
    new_rhs[0] = rhs[0];

    if (new_main[0] == 0)
      return false;

    for (int i = 1; i < n; ++i) {
      double factor = lower[i - 1] / new_main[i - 1];
      new_main[i] = main_diag[i] - factor * upper[i - 1];
      new_rhs[i] = rhs[i] - factor * new_rhs[i - 1];
      if (new_main[i] == 0)
        return false;
    }

    solution[n - 1] = new_rhs[n - 1] / new_main[n - 1];
    for (int i = n - 2; i >= 0; --i) {
      solution[i] = (new_rhs[i] - upper[i] * solution[i + 1]) / new_main[i];
    }

    return true;
  }

public:
  CubicSpline(double x_start, double step, const std::vector<double>& y)
    : x0(x_start), h(step), num_intervals(y.size() - 1)
  {

    if (y.size() < 2) {
      throw std::invalid_argument("At least two points required for spline.");
    }
    if (h <= 0) {
      throw std::invalid_argument("Step size must be positive.");
    }

    int n = y.size();
    a.resize(num_intervals);
    b.resize(num_intervals);
    c.resize(num_intervals);
    d.resize(num_intervals);

    if (n == 2) {
      a[0] = y[0];
      b[0] = (y[1] - y[0]) / h;
      c[0] = 0.0;
      d[0] = 0.0;
      return;
    }

    std::vector<double> M(n, 0.0); // natural condition
    int num_unknowns = n - 2;
    std::vector<double> main_diag(num_unknowns, 4.0);
    std::vector<double> lower_diag(num_unknowns - 1, 1.0);
    std::vector<double> upper_diag(num_unknowns - 1, 1.0);
    std::vector<double> rhs(num_unknowns);

    for (int j = 0; j < num_unknowns; ++j) {
      int i = j + 1;
      rhs[j] = 6.0 * (y[i + 1] - 2 * y[i] + y[i - 1]) / (h * h);
    }

    std::vector<double> solution(num_unknowns);
    if (!thomas_algorithm(lower_diag, main_diag, upper_diag, rhs, solution)) {
      throw std::runtime_error("Failed to solve tridiagonal system.");
    }

    for (int j = 0; j < num_unknowns; ++j) {
      M[j + 1] = solution[j];
    }

    for (int i = 0; i < num_intervals; ++i) {
      double y_i = y[i];
      double y_next = y[i + 1];
      double M_i = M[i];
      double M_next = M[i + 1];

      a[i] = y_i;
      c[i] = M_i / 2.0;
      d[i] = (M_next - M_i) / (6.0 * h);
      b[i] = (y_next - y_i) / h - h * (2 * M_i + M_next) / 6.0;
    }
  }

  const std::vector<double>& get_a() const { return a; }
  const std::vector<double>& get_b() const { return b; }
  const std::vector<double>& get_c() const { return c; }
  const std::vector<double>& get_d() const { return d; }
};

__device__ double get_rho_and_F(
  double x,
  double x0,
  double h,
  int type,
  const double* a,
  const double* b,
  const double* c,
  const double* d,
  int num_intervals)
{

  int i = static_cast<int>((x - x0) / h);
  if (i >= num_intervals)
    i = num_intervals - 1;

  double dx = x - (x0 + i * h);
  int index = type * num_intervals + i;
  return a[index] + b[index] * dx + c[index] * dx * dx + d[index] * dx * dx * dx;
}

__device__ double get_rho_and_F_derivative(
  double x,
  double x0,
  double h,
  int type,
  const double* b,
  const double* c,
  const double* d,
  int num_intervals)
{

  int i = static_cast<int>((x - x0) / h);
  if (i >= num_intervals)
    i = num_intervals - 1;

  double dx = x - (x0 + i * h);
  int index = type * num_intervals + i;
  return b[index] + 2 * c[index] * dx + 3 * d[index] * dx * dx;
}

__device__ double get_phi(
  double x,
  double x0,
  double h,
  int i_type,
  int j_type,
  int Nelements,
  const double* a,
  const double* b,
  const double* c,
  const double* d,
  int num_intervals)
{

  int i = static_cast<int>((x - x0) / h);
  if (i >= num_intervals)
    i = num_intervals - 1;

  double dx = x - (x0 + i * h);
  int index = (i_type * Nelements + j_type) * num_intervals + i;
  return a[index] + b[index] * dx + c[index] * dx * dx + d[index] * dx * dx * dx;
}

__device__ double get_phi_derivative(
  double x,
  double x0,
  double h,
  int i_type,
  int j_type,
  int Nelements,
  const double* b,
  const double* c,
  const double* d,
  int num_intervals)
{

  int i = static_cast<int>((x - x0) / h);
  if (i >= num_intervals)
    i = num_intervals - 1;

  double dx = x - (x0 + i * h);
  int index = (i_type * Nelements + j_type) * num_intervals + i;
  return b[index] + 2 * c[index] * dx + 3 * d[index] * dx * dx;
}

EAMAlloy::EAMAlloy(const char* filename, const int number_of_atoms)
{

  initialize_eamalloy(filename, number_of_atoms);
  eam_data.NN.resize(number_of_atoms);
  eam_data.NL.resize(number_of_atoms * 400); // very safe for EAM
  eam_data.cell_count.resize(number_of_atoms);
  eam_data.cell_count_sum.resize(number_of_atoms);
  eam_data.cell_contents.resize(number_of_atoms);
  eam_data.d_F_rho_i_g.resize(number_of_atoms);
}

void EAMAlloy::initialize_eamalloy(const char* filename, const int number_of_atoms)
{

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + std::string(filename));
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  file.close();

  std::vector<std::string> data_words;
  for (size_t i = 3; i < lines.size(); ++i) {
    std::istringstream iss(lines[i]);
    std::string word;
    while (iss >> word) {
      data_words.push_back(word);
    }
  }

  size_t index = 0;
  eam_data.Nelements = std::stoi(data_words[index++]);
  eam_data.elements_list.assign(
    data_words.begin() + index, data_words.begin() + index + eam_data.Nelements);

  printf("Use %d-element EAM/Alloy potential with element(s): ", eam_data.Nelements);
  for (int i = 0; i < eam_data.Nelements; ++i) {
    printf("%s ", eam_data.elements_list[i].c_str());
  }
  printf("\n");

  index += eam_data.Nelements;
  eam_data.nrho = std::stoi(data_words[index++]);
  eam_data.drho = std::stod(data_words[index++]);
  eam_data.nr = std::stoi(data_words[index++]);
  eam_data.dr = std::stod(data_words[index++]);
  eam_data.rc = std::stod(data_words[index++]);
  eam_data.F_rho.resize(eam_data.Nelements * eam_data.nrho, 0.0);
  eam_data.rho_r.resize(eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.phi_r.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.F_rho_a.resize(eam_data.Nelements * eam_data.nrho, 0.0);
  eam_data.F_rho_b.resize(eam_data.Nelements * eam_data.nrho, 0.0);
  eam_data.F_rho_c.resize(eam_data.Nelements * eam_data.nrho, 0.0);
  eam_data.F_rho_d.resize(eam_data.Nelements * eam_data.nrho, 0.0);
  eam_data.rho_r_a.resize(eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.rho_r_b.resize(eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.rho_r_c.resize(eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.rho_r_d.resize(eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.phi_r_a.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.phi_r_b.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.phi_r_c.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.phi_r_d.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0);
  eam_data.atomic_number.resize(eam_data.Nelements, 0);
  eam_data.atomic_mass.resize(eam_data.Nelements, 0.0);
  eam_data.lattice_constant.resize(eam_data.Nelements, 0.0);
  eam_data.lattice_type.resize(eam_data.Nelements);

  for (int i = 0; i < eam_data.Nelements; ++i) {
    eam_data.atomic_number[i] = std::stoi(data_words[index++]);
    eam_data.atomic_mass[i] = std::stod(data_words[index++]);
    eam_data.lattice_constant[i] = std::stod(data_words[index++]);
    eam_data.lattice_type[i] = data_words[index++];

    for (int j = 0; j < eam_data.nrho; ++j) {
      eam_data.F_rho[i * eam_data.nrho + j] = std::stod(data_words[index++]);
    }

    for (int j = 0; j < eam_data.nr; ++j) {
      eam_data.rho_r[i * eam_data.nr + j] = std::stod(data_words[index++]);
    }
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j < eam_data.Nelements; ++j) {
      if (i >= j) {
        for (int k = 0; k < eam_data.nr; ++k) {
          size_t idx = (i * eam_data.Nelements + j) * eam_data.nr + k;
          eam_data.phi_r[idx] = std::stod(data_words[index++]);
        }
        if (i != j) {
          for (int k = 0; k < eam_data.nr; ++k) {
            size_t idx_ij = (i * eam_data.Nelements + j) * eam_data.nr + k;
            size_t idx_ji = (j * eam_data.Nelements + i) * eam_data.nr + k;
            eam_data.phi_r[idx_ji] = eam_data.phi_r[idx_ij];
          }
        }
      }
    }
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j < eam_data.Nelements; ++j) {
      for (int k = 1; k < eam_data.nr; ++k) {
        size_t idx = (i * eam_data.Nelements + j) * eam_data.nr + k;
        eam_data.phi_r[idx] /= k * eam_data.dr;
      }
      size_t idx0 = (i * eam_data.Nelements + j) * eam_data.nr;
      size_t idx1 = (i * eam_data.Nelements + j) * eam_data.nr + 1;
      eam_data.phi_r[idx0] = eam_data.phi_r[idx1];
    }
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    std::vector<double> y_sub(
      eam_data.F_rho.begin() + i * eam_data.nrho,
      eam_data.F_rho.begin() + i * eam_data.nrho + eam_data.nrho);
    auto sp = CubicSpline(0.0, eam_data.drho, y_sub);
    for (int j = 0; j < eam_data.nrho; ++j) {
      size_t idx = i * eam_data.nrho + j;
      eam_data.F_rho_a[idx] = sp.get_a()[j];
      eam_data.F_rho_b[idx] = sp.get_b()[j];
      eam_data.F_rho_c[idx] = sp.get_c()[j];
      eam_data.F_rho_d[idx] = sp.get_d()[j];
    }
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    std::vector<double> y_sub(
      eam_data.rho_r.begin() + i * eam_data.nr,
      eam_data.rho_r.begin() + i * eam_data.nr + eam_data.nr);
    auto sp = CubicSpline(0.0, eam_data.dr, y_sub);
    for (int j = 0; j < eam_data.nr; ++j) {
      size_t idx = i * eam_data.nr + j;
      eam_data.rho_r_a[idx] = sp.get_a()[j];
      eam_data.rho_r_b[idx] = sp.get_b()[j];
      eam_data.rho_r_c[idx] = sp.get_c()[j];
      eam_data.rho_r_d[idx] = sp.get_d()[j];
    }
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j < eam_data.Nelements; ++j) {
      std::vector<double> y_sub(
        eam_data.phi_r.begin() + (i * eam_data.Nelements + j) * eam_data.nr,
        eam_data.phi_r.begin() + (i * eam_data.Nelements + j) * eam_data.nr + eam_data.nr);
      auto sp = CubicSpline(0.0, eam_data.dr, y_sub);
      for (int k = 0; k < eam_data.nr; ++k) {
        size_t idx = (i * eam_data.Nelements + j) * eam_data.nr + k;
        eam_data.phi_r_a[idx] = sp.get_a()[k];
        eam_data.phi_r_b[idx] = sp.get_b()[k];
        eam_data.phi_r_c[idx] = sp.get_c()[k];
        eam_data.phi_r_d[idx] = sp.get_d()[k];
      }
    }
  }

  // Copy to GPU
  eam_data.F_rho_a_g.resize(eam_data.F_rho_a.size());
  eam_data.F_rho_b_g.resize(eam_data.F_rho_b.size());
  eam_data.F_rho_c_g.resize(eam_data.F_rho_c.size());
  eam_data.F_rho_d_g.resize(eam_data.F_rho_d.size());
  eam_data.rho_r_a_g.resize(eam_data.rho_r_a.size());
  eam_data.rho_r_b_g.resize(eam_data.rho_r_b.size());
  eam_data.rho_r_c_g.resize(eam_data.rho_r_c.size());
  eam_data.rho_r_d_g.resize(eam_data.rho_r_d.size());
  eam_data.phi_r_a_g.resize(eam_data.phi_r_a.size());
  eam_data.phi_r_b_g.resize(eam_data.phi_r_b.size());
  eam_data.phi_r_c_g.resize(eam_data.phi_r_c.size());
  eam_data.phi_r_d_g.resize(eam_data.phi_r_d.size());

  eam_data.F_rho_a_g.copy_from_host(eam_data.F_rho_a.data(), eam_data.F_rho_a.size());
  eam_data.F_rho_b_g.copy_from_host(eam_data.F_rho_b.data(), eam_data.F_rho_b.size());
  eam_data.F_rho_c_g.copy_from_host(eam_data.F_rho_c.data(), eam_data.F_rho_c.size());
  eam_data.F_rho_d_g.copy_from_host(eam_data.F_rho_d.data(), eam_data.F_rho_d.size());
  eam_data.rho_r_a_g.copy_from_host(eam_data.rho_r_a.data(), eam_data.rho_r_a.size());
  eam_data.rho_r_b_g.copy_from_host(eam_data.rho_r_b.data(), eam_data.rho_r_b.size());
  eam_data.rho_r_c_g.copy_from_host(eam_data.rho_r_c.data(), eam_data.rho_r_c.size());
  eam_data.rho_r_d_g.copy_from_host(eam_data.rho_r_d.data(), eam_data.rho_r_d.size());
  eam_data.phi_r_a_g.copy_from_host(eam_data.phi_r_a.data(), eam_data.phi_r_a.size());
  eam_data.phi_r_b_g.copy_from_host(eam_data.phi_r_b.data(), eam_data.phi_r_b.size());
  eam_data.phi_r_c_g.copy_from_host(eam_data.phi_r_c.data(), eam_data.phi_r_c.size());
  eam_data.phi_r_d_g.copy_from_host(eam_data.phi_r_d.data(), eam_data.phi_r_d.size());
}

EAMAlloy::~EAMAlloy(void)
{
  // nothing
}

static __global__ void find_force_eam_step1(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const int nr,
  const int nrho,
  const int Nelements,
  const double rc,
  const double dr,
  const double drho,
  const double* F_rho_a,
  const double* F_rho_b,
  const double* F_rho_c,
  const double* F_rho_d,
  const double* rho_r_a,
  const double* rho_r_b,
  const double* rho_r_c,
  const double* rho_r_d,
  const double* phi_r_a,
  const double* phi_r_b,
  const double* phi_r_c,
  const double* phi_r_d,
  double* d_F_rho_i,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
    int NN = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    const int i_type = g_type[n1];
    double rho = 0.0;
    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      if (d12 <= rc) {
        const int j_type = g_type[n2];
        g_pe[n1] +=
          get_phi(d12, 0.0, dr, i_type, j_type, Nelements, phi_r_a, phi_r_b, phi_r_c, phi_r_d, nr) *
          0.5;
        rho += get_rho_and_F(d12, 0.0, dr, j_type, rho_r_a, rho_r_b, rho_r_c, rho_r_d, nr);
      }
    }
    g_pe[n1] += get_rho_and_F(rho, 0.0, drho, i_type, F_rho_a, F_rho_b, F_rho_c, F_rho_d, nrho);
    d_F_rho_i[n1] =
      get_rho_and_F_derivative(rho, 0.0, drho, i_type, F_rho_b, F_rho_c, F_rho_d, nrho);
  }
}

static __global__ void find_force_eam_step2(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const int nr,
  const int nrho,
  const int Nelements,
  const double rc,
  const double dr,
  const double drho,
  const double* F_rho_a,
  const double* F_rho_b,
  const double* F_rho_c,
  const double* F_rho_d,
  const double* rho_r_a,
  const double* rho_r_b,
  const double* rho_r_c,
  const double* rho_r_d,
  const double* phi_r_a,
  const double* phi_r_b,
  const double* phi_r_c,
  const double* phi_r_d,
  double* d_F_rho_i,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
    int NN = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    const int i_type = g_type[n1];
    double Fp1 = d_F_rho_i[n1];
    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double xij = g_x[n2] - x1;
      double yij = g_y[n2] - y1;
      double zij = g_z[n2] - z1;
      apply_mic(box, xij, yij, zij);
      double r = sqrt(xij * xij + yij * yij + zij * zij);
      if (r <= rc) {
        const int j_type = g_type[n2];
        double Fp2 = d_F_rho_i[n2];
        double d_phi_r_i =
          get_phi_derivative(r, 0.0, dr, i_type, j_type, Nelements, phi_r_b, phi_r_c, phi_r_d, nr);
        double d_F_i =
          get_rho_and_F_derivative(r, 0.0, dr, j_type, rho_r_b, rho_r_c, rho_r_d, nr) * Fp1;
        double d_F_j =
          get_rho_and_F_derivative(r, 0.0, dr, i_type, rho_r_b, rho_r_c, rho_r_d, nr) * Fp2;

        double fij = d_phi_r_i + d_F_i + d_F_j;
        double fx = fij * xij / r;
        double fy = fij * yij / r;
        double fz = fij * zij / r;

        // save force
        g_fx[n1] += fx;
        g_fy[n1] += fy;
        g_fz[n1] += fz;
        double sxx = fx * xij * 0.5;
        double syy = fy * yij * 0.5;
        double szz = fz * zij * 0.5;
        double sxy = fx * yij * 0.5;
        double sxz = fx * zij * 0.5;
        double syz = fy * zij * 0.5;
        // save virial
        // xx xy xz    0 3 4
        // yx yy yz    6 1 5
        // zx zy zz    7 8 2
        g_virial[n1 + 0 * N] -= sxx;
        g_virial[n1 + 1 * N] -= syy;
        g_virial[n1 + 2 * N] -= szz;
        g_virial[n1 + 3 * N] -= sxy;
        g_virial[n1 + 4 * N] -= sxz;
        g_virial[n1 + 5 * N] -= syz;
        g_virial[n1 + 6 * N] -= sxy;
        g_virial[n1 + 7 * N] -= sxz;
        g_virial[n1 + 8 * N] -= syz;
      }
    }
  }
}

// Force evaluation wrapper
void EAMAlloy::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();

  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

  find_neighbor(
    N1,
    N2,
    eam_data.rc,
    box,
    type,
    position_per_atom,
    eam_data.cell_count,
    eam_data.cell_count_sum,
    eam_data.cell_contents,
    eam_data.NN,
    eam_data.NL);

  eam_data.d_F_rho_i_g.fill(0.0);
  find_force_eam_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    eam_data.NN.data(),
    eam_data.NL.data(),
    type.data(),
    eam_data.nr,
    eam_data.nrho,
    eam_data.Nelements,
    eam_data.rc,
    eam_data.dr,
    eam_data.drho,
    eam_data.F_rho_a_g.data(),
    eam_data.F_rho_b_g.data(),
    eam_data.F_rho_c_g.data(),
    eam_data.F_rho_d_g.data(),
    eam_data.rho_r_a_g.data(),
    eam_data.rho_r_b_g.data(),
    eam_data.rho_r_c_g.data(),
    eam_data.rho_r_d_g.data(),
    eam_data.phi_r_a_g.data(),
    eam_data.phi_r_b_g.data(),
    eam_data.phi_r_c_g.data(),
    eam_data.phi_r_d_g.data(),
    eam_data.d_F_rho_i_g.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_eam_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    eam_data.NN.data(),
    eam_data.NL.data(),
    type.data(),
    eam_data.nr,
    eam_data.nrho,
    eam_data.Nelements,
    eam_data.rc,
    eam_data.dr,
    eam_data.drho,
    eam_data.F_rho_a_g.data(),
    eam_data.F_rho_b_g.data(),
    eam_data.F_rho_c_g.data(),
    eam_data.F_rho_d_g.data(),
    eam_data.rho_r_a_g.data(),
    eam_data.rho_r_b_g.data(),
    eam_data.rho_r_c_g.data(),
    eam_data.rho_r_d_g.data(),
    eam_data.phi_r_a_g.data(),
    eam_data.phi_r_b_g.data(),
    eam_data.phi_r_c_g.data(),
    eam_data.phi_r_d_g.data(),
    eam_data.d_F_rho_i_g.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
}
