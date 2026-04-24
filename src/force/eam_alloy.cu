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

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define BLOCK_SIZE_FORCE 64

// LAMMPS-style Hermite cubic spline using centered finite-difference derivatives.
// Input: y has n values at positions r = 0, h, 2h, ..., (n-1)h
// Output: coefficients a, b, c, d (each length n) so that for interval m ∈ [0, n-2]:
//   f(r) = a[m] + b[m]*dx + c[m]*dx^2 + d[m]*dx^3, dx = r - m*h
// For m = n-1 the stored values give a linear extrapolation (c=d=0).
static void compute_lammps_spline(
  float h,
  const std::vector<float>& y,
  std::vector<float>& a,
  std::vector<float>& b,
  std::vector<float>& c,
  std::vector<float>& d)
{
  const int n = static_cast<int>(y.size());
  a.assign(n, 0.0f);
  b.assign(n, 0.0f);
  c.assign(n, 0.0f);
  d.assign(n, 0.0f);
  if (n < 2)
    return;

  // Derivative with respect to the normalized coordinate p = r/h (so dp per grid step = 1)
  std::vector<float> fp(n, 0.0f);

  fp[0] = y[1] - y[0];
  fp[n - 1] = y[n - 1] - y[n - 2];
  if (n >= 3) {
    fp[1] = 0.5f * (y[2] - y[0]);
    fp[n - 2] = 0.5f * (y[n - 1] - y[n - 3]);
  }
  for (int m = 2; m <= n - 3; ++m) {
    fp[m] = ((y[m - 2] - y[m + 2]) + 8.0f * (y[m + 1] - y[m - 1])) / 12.0f;
  }

  const float inv_h = 1.0f / h;
  const float inv_h2 = inv_h * inv_h;
  const float inv_h3 = inv_h2 * inv_h;

  for (int m = 0; m <= n - 2; ++m) {
    float dy = y[m + 1] - y[m];
    float B2 = 3.0f * dy - 2.0f * fp[m] - fp[m + 1];          // p^2 coefficient
    float B3 = fp[m] + fp[m + 1] - 2.0f * dy;                 // p^3 coefficient
    a[m] = y[m];
    b[m] = fp[m] * inv_h;
    c[m] = B2 * inv_h2;
    d[m] = B3 * inv_h3;
  }
  // Linear extrapolation slope at the last grid point (used only if ii is clamped to n-1).
  a[n - 1] = y[n - 1];
  b[n - 1] = fp[n - 1] * inv_h;
  c[n - 1] = 0.0f;
  d[n - 1] = 0.0f;
}

__device__ float get_cubic(
  int i,
  float x,
  float h,
  int type,
  const float* a,
  const float* b,
  const float* c,
  const float* d,
  int stride)
{
  float dx = x - (i * h);
  int index = type * stride + i;
  return a[index] + (b[index] + (c[index] + d[index] * dx) * dx) * dx;
}

__device__ float get_cubic_derivative(
  int i,
  float x,
  float h,
  int type,
  const float* b,
  const float* c,
  const float* d,
  int stride)
{
  float dx = x - (i * h);
  int index = type * stride + i;
  return b[index] + (2.0f * c[index] + 3.0f * d[index] * dx) * dx;
}

__device__ float get_pair(
  int i,
  float x,
  float h,
  int i_type,
  int j_type,
  int Nelements,
  const float* a,
  const float* b,
  const float* c,
  const float* d,
  int stride)
{
  float dx = x - (i * h);
  int index = (i_type * Nelements + j_type) * stride + i;
  return a[index] + (b[index] + (c[index] + d[index] * dx) * dx) * dx;
}

__device__ float get_pair_derivative(
  int i,
  float x,
  float h,
  int i_type,
  int j_type,
  int Nelements,
  const float* b,
  const float* c,
  const float* d,
  int stride)
{
  float dx = x - (i * h);
  int index = (i_type * Nelements + j_type) * stride + i;
  return b[index] + (2.0f * c[index] + 3.0f * d[index] * dx) * dx;
}

EAMAlloy::EAMAlloy(const char* filename, const int number_of_atoms)
{

  initialize_eamalloy(filename, number_of_atoms);
  eam_data.NN.resize(number_of_atoms);
  eam_data.NL.resize(number_of_atoms * 400); // very safe for EAM
  neighbor.initialize(eam_data.rc, number_of_atoms, 400);
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

  eam_data.F_rho.resize(eam_data.Nelements * eam_data.nrho, 0.0f);
  eam_data.rho_r.resize(eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.phi_r.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.F_rho_a.resize(eam_data.Nelements * eam_data.nrho, 0.0f);
  eam_data.F_rho_b.resize(eam_data.Nelements * eam_data.nrho, 0.0f);
  eam_data.F_rho_c.resize(eam_data.Nelements * eam_data.nrho, 0.0f);
  eam_data.F_rho_d.resize(eam_data.Nelements * eam_data.nrho, 0.0f);
  eam_data.rho_r_a.resize(eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.rho_r_b.resize(eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.rho_r_c.resize(eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.rho_r_d.resize(eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.phi_r_a.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.phi_r_b.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.phi_r_c.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0f);
  eam_data.phi_r_d.resize(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0f);

  // Section reader: reads n_values starting at the current line_idx, one line at a time.
  // After reading the required count, any leftover tokens on the last line are discarded
  // and line_idx advances to the next line. This matches LAMMPS's TextFileReader::next_dvector
  // convention, where each section starts on a new line.
  int line_idx = 5;
  auto read_section = [&](float* dst, int n_values) {
    int count = 0;
    while (count < n_values && line_idx < static_cast<int>(lines.size())) {
      std::istringstream iss(lines[line_idx]);
      std::string tok;
      while (iss >> tok && count < n_values) {
        try {
          dst[count] = std::stod(tok);
          count++;
        } catch (const std::invalid_argument&) {
          break;
        }
      }
      line_idx++;
    }
  };
  for (int i = 0; i < eam_data.Nelements; ++i) {
    // skip the per-element info line (atomic number, mass, etc.)
    line_idx++;
    read_section(eam_data.F_rho.data() + i * eam_data.nrho, eam_data.nrho);
    read_section(eam_data.rho_r.data() + i * eam_data.nr, eam_data.nr);
  }

  // read phi_r as r*phi(r) (LAMMPS z2r convention); each pair block starts on a new line.
  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j <= i; ++j) {
      read_section(
        eam_data.phi_r.data() + (i * eam_data.Nelements + j) * eam_data.nr, eam_data.nr);
      // mirror to upper triangle
      if (i != j) {
        for (int k = 0; k < eam_data.nr; ++k) {
          size_t idx_ij = (i * eam_data.Nelements + j) * eam_data.nr + k;
          size_t idx_ji = (j * eam_data.Nelements + i) * eam_data.nr + k;
          eam_data.phi_r[idx_ji] = eam_data.phi_r[idx_ij];
        }
      }
    }
  }

  // Hermite spline for F(rho).
  for (int i = 0; i < eam_data.Nelements; ++i) {
    std::vector<float> y_sub(
      eam_data.F_rho.begin() + i * eam_data.nrho,
      eam_data.F_rho.begin() + (i + 1) * eam_data.nrho);
    std::vector<float> ca, cb, cc, cd;
    compute_lammps_spline(eam_data.drho, y_sub, ca, cb, cc, cd);
    for (int j = 0; j < eam_data.nrho; ++j) {
      size_t idx = i * eam_data.nrho + j;
      eam_data.F_rho_a[idx] = ca[j];
      eam_data.F_rho_b[idx] = cb[j];
      eam_data.F_rho_c[idx] = cc[j];
      eam_data.F_rho_d[idx] = cd[j];
    }
  }

  // Hermite spline for rho(r).
  for (int i = 0; i < eam_data.Nelements; ++i) {
    std::vector<float> y_sub(
      eam_data.rho_r.begin() + i * eam_data.nr,
      eam_data.rho_r.begin() + (i + 1) * eam_data.nr);
    std::vector<float> ca, cb, cc, cd;
    compute_lammps_spline(eam_data.dr, y_sub, ca, cb, cc, cd);
    for (int j = 0; j < eam_data.nr; ++j) {
      size_t idx = i * eam_data.nr + j;
      eam_data.rho_r_a[idx] = ca[j];
      eam_data.rho_r_b[idx] = cb[j];
      eam_data.rho_r_c[idx] = cc[j];
      eam_data.rho_r_d[idx] = cd[j];
    }
  }

  // Hermite spline for r*phi(r) (stored as phi_r_* on GPU; divided by r at lookup).
  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j < eam_data.Nelements; ++j) {
      std::vector<float> y_sub(
        eam_data.phi_r.begin() + (i * eam_data.Nelements + j) * eam_data.nr,
        eam_data.phi_r.begin() + (i * eam_data.Nelements + j + 1) * eam_data.nr);
      std::vector<float> ca, cb, cc, cd;
      compute_lammps_spline(eam_data.dr, y_sub, ca, cb, cc, cd);
      for (int k = 0; k < eam_data.nr; ++k) {
        size_t idx = (i * eam_data.Nelements + j) * eam_data.nr + k;
        eam_data.phi_r_a[idx] = ca[k];
        eam_data.phi_r_b[idx] = cb[k];
        eam_data.phi_r_c[idx] = cc[k];
        eam_data.phi_r_d[idx] = cd[k];
      }
    }
  }

  // GPU memory copy
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
  const int* g_NN_global,
  const int* g_NL_global,
  int* g_NN_local,
  int* g_NL_local,
  const int* g_type,
  const int nr,
  const int nrho,
  const int Nelements,
  const float rc,
  const float dr,
  const float dr_inv,
  const float drho,
  const float drho_inv,
  const float* F_rho_a,
  const float* F_rho_b,
  const float* F_rho_c,
  const float* F_rho_d,
  const float* rho_r_a,
  const float* rho_r_b,
  const float* rho_r_c,
  const float* rho_r_d,
  const float* phi_r_a,
  const float* phi_r_b,
  const float* phi_r_c,
  const float* phi_r_d,
  float* d_F_rho_i,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
    int NN = g_NN_global[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    const int i_type = g_type[n1];
    float rho = 0.0f;
    int count_local = 0;

    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL_global[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      if (d12 <= rc) {
        const int j_type = g_type[n2];

        int ii = static_cast<int>(d12 * dr_inv);
        if (ii > nr - 2)
          ii = nr - 2;

        float z2 = get_pair(
          ii, d12, dr, i_type, j_type, Nelements, phi_r_a, phi_r_b, phi_r_c, phi_r_d, nr);
        float phi_ij = z2 / d12;
        g_pe[n1] += phi_ij * 0.5f;
        rho += get_cubic(ii, d12, dr, j_type, rho_r_a, rho_r_b, rho_r_c, rho_r_d, nr);
      }

      g_NL_local[count_local++ * N + n1] = n2;
    }

    g_NN_local[n1] = count_local;

    int jj = static_cast<int>(rho * drho_inv);
    if (jj > nrho - 2)
      jj = nrho - 2;
    if (jj < 0)
      jj = 0;

    g_pe[n1] += get_cubic(jj, rho, drho, i_type, F_rho_a, F_rho_b, F_rho_c, F_rho_d, nrho);
    d_F_rho_i[n1] =
      get_cubic_derivative(jj, rho, drho, i_type, F_rho_b, F_rho_c, F_rho_d, nrho);
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
  const float rc,
  const float dr,
  const float dr_inv,
  const float* F_rho_a,
  const float* F_rho_b,
  const float* F_rho_c,
  const float* F_rho_d,
  const float* rho_r_a,
  const float* rho_r_b,
  const float* rho_r_c,
  const float* rho_r_d,
  const float* phi_r_a,
  const float* phi_r_b,
  const float* phi_r_c,
  const float* phi_r_d,
  float* d_F_rho_i,
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
    float Fp1 = d_F_rho_i[n1];
    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float xij = g_x[n2] - x1;
      float yij = g_y[n2] - y1;
      float zij = g_z[n2] - z1;
      apply_mic(box, xij, yij, zij);
      float r = sqrt(xij * xij + yij * yij + zij * zij);
      if (r <= rc) {
        const int j_type = g_type[n2];
        float Fp2 = d_F_rho_i[n2];

        int ii = static_cast<int>(r * dr_inv);
        if (ii > nr - 2)
          ii = nr - 2;

        float rinv = 1.0f / r;
        float z2 = get_pair(
          ii, r, dr, i_type, j_type, Nelements, phi_r_a, phi_r_b, phi_r_c, phi_r_d, nr);
        float dz2_dr = get_pair_derivative(
          ii, r, dr, i_type, j_type, Nelements, phi_r_b, phi_r_c, phi_r_d, nr);
        float phi_ij = z2 * rinv;
        float d_phi_r_i = (dz2_dr - phi_ij) * rinv;  // dphi/dr

        float d_F_i =
          get_cubic_derivative(ii, r, dr, j_type, rho_r_b, rho_r_c, rho_r_d, nr) * Fp1;
        float d_F_j =
          get_cubic_derivative(ii, r, dr, i_type, rho_r_b, rho_r_c, rho_r_d, nr) * Fp2;

        float fij = d_phi_r_i + d_F_i + d_F_j;
        float fx = fij * xij * rinv;
        float fy = fij * yij * rinv;
        float fz = fij * zij * rinv;

        // save force
        g_fx[n1] += fx;
        g_fy[n1] += fy;
        g_fz[n1] += fz;
        float sxx = fx * xij * 0.5f;
        float syy = fy * yij * 0.5f;
        float szz = fz * zij * 0.5f;
        float sxy = fx * yij * 0.5f;
        float sxz = fx * zij * 0.5f;
        float syz = fy * zij * 0.5f;
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

  neighbor.find_neighbor_global(
    eam_data.rc,
    box,
    type,
    position_per_atom);

  eam_data.d_F_rho_i_g.fill(0.0f);
  find_force_eam_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    neighbor.NN.data(),
    neighbor.NL.data(),
    eam_data.NN.data(),
    eam_data.NL.data(),
    type.data(),
    eam_data.nr,
    eam_data.nrho,
    eam_data.Nelements,
    eam_data.rc,
    eam_data.dr,
    1.0f / eam_data.dr,
    eam_data.drho,
    1.0f / eam_data.drho,
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
    1.0f / eam_data.dr,
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
