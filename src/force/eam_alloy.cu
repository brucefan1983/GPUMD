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
#define BLOCK_SIZE_FORCE 256

// LAMMPS-style Hermite cubic spline using centered finite-difference derivatives.
// Fills `out[m] = (a, b, c, d)` so that on interval m ∈ [0, n-2]:
//   f(r) = a + b*dx + c*dx^2 + d*dx^3,  dx = r - m*h
// For m = n-1 we store (y[n-1], y'(n-1)/h, 0, 0) — a linear extrapolation guard.
static void compute_lammps_spline(float h, const std::vector<float>& y, std::vector<float4>& out)
{
  const int n = static_cast<int>(y.size());
  out.assign(n, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
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
    float B2 = 3.0f * dy - 2.0f * fp[m] - fp[m + 1]; // p^2 coefficient
    float B3 = fp[m] + fp[m + 1] - 2.0f * dy;        // p^3 coefficient
    out[m].x = y[m];
    out[m].y = fp[m] * inv_h;
    out[m].z = B2 * inv_h2;
    out[m].w = B3 * inv_h3;
  }
  out[n - 1].x = y[n - 1];
  out[n - 1].y = fp[n - 1] * inv_h;
}

// Spline coefficients are stored interleaved as float4 = (a, b, c, d) per (type, interval),
// so a single 16-byte read fetches all four polynomial coefficients (vs. four scattered
// 4-byte reads from separate arrays). Loaded via __ldg through the read-only cache.
__device__ inline float
spline_value(int i, float dx, int type, const float4* __restrict__ coef, int stride)
{
  float4 c = __ldg(&coef[type * stride + i]);
  return c.x + (c.y + (c.z + c.w * dx) * dx) * dx;
}

__device__ inline float
spline_deriv(int i, float dx, int type, const float4* __restrict__ coef, int stride)
{
  float4 c = __ldg(&coef[type * stride + i]);
  return c.y + (2.0f * c.z + 3.0f * c.w * dx) * dx;
}

__device__ inline float pair_value(
  int i,
  float dx,
  int i_type,
  int j_type,
  int Nelements,
  const float4* __restrict__ coef,
  int stride)
{
  float4 c = __ldg(&coef[(i_type * Nelements + j_type) * stride + i]);
  return c.x + (c.y + (c.z + c.w * dx) * dx) * dx;
}

// Combined value + derivative lookup for phi spline: one cache miss instead of two.
__device__ inline void pair_value_and_deriv(
  int i,
  float dx,
  int i_type,
  int j_type,
  int Nelements,
  const float4* __restrict__ coef,
  int stride,
  float& val,
  float& der)
{
  float4 c = __ldg(&coef[(i_type * Nelements + j_type) * stride + i]);
  val = c.x + (c.y + (c.z + c.w * dx) * dx) * dx;
  der = c.y + (2.0f * c.z + 3.0f * c.w * dx) * dx;
}

EAMAlloy::EAMAlloy(const char* filename, const int number_of_atoms, const int max_neighbor)
{
  initialize_eamalloy(filename, number_of_atoms);

  neighbor.initialize(eam_data.rc, number_of_atoms, 1);
  neighbor.NL.resize(static_cast<size_t>(number_of_atoms) * max_neighbor);
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

  // Host-side raw tabulated values (used only during initialization).
  std::vector<float> F_rho(eam_data.Nelements * eam_data.nrho, 0.0f);
  std::vector<float> rho_r(eam_data.Nelements * eam_data.nr, 0.0f);
  std::vector<float> phi_r(eam_data.Nelements * eam_data.Nelements * eam_data.nr, 0.0f);

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
    line_idx++; // skip per-element info line
    read_section(F_rho.data() + i * eam_data.nrho, eam_data.nrho);
    read_section(rho_r.data() + i * eam_data.nr, eam_data.nr);
  }

  // phi_r is stored as r*phi(r) (LAMMPS z2r convention); each pair block starts on a new line.
  // Lower-triangular (i >= j) blocks are read from the file and mirrored.
  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j <= i; ++j) {
      read_section(phi_r.data() + (i * eam_data.Nelements + j) * eam_data.nr, eam_data.nr);
      if (i != j) {
        for (int k = 0; k < eam_data.nr; ++k) {
          size_t idx_ij = (i * eam_data.Nelements + j) * eam_data.nr + k;
          size_t idx_ji = (j * eam_data.Nelements + i) * eam_data.nr + k;
          phi_r[idx_ji] = phi_r[idx_ij];
        }
      }
    }
  }

  // Build packed (a,b,c,d) coefficient tables and upload to GPU.
  std::vector<float4> F_rho_packed(eam_data.Nelements * eam_data.nrho);
  std::vector<float4> rho_r_packed(eam_data.Nelements * eam_data.nr);
  std::vector<float4> phi_r_packed(eam_data.Nelements * eam_data.Nelements * eam_data.nr);

  for (int i = 0; i < eam_data.Nelements; ++i) {
    std::vector<float> y(
      F_rho.begin() + i * eam_data.nrho, F_rho.begin() + (i + 1) * eam_data.nrho);
    std::vector<float4> coef;
    compute_lammps_spline(eam_data.drho, y, coef);
    std::copy(coef.begin(), coef.end(), F_rho_packed.begin() + i * eam_data.nrho);
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    std::vector<float> y(rho_r.begin() + i * eam_data.nr, rho_r.begin() + (i + 1) * eam_data.nr);
    std::vector<float4> coef;
    compute_lammps_spline(eam_data.dr, y, coef);
    std::copy(coef.begin(), coef.end(), rho_r_packed.begin() + i * eam_data.nr);
  }

  for (int i = 0; i < eam_data.Nelements; ++i) {
    for (int j = 0; j < eam_data.Nelements; ++j) {
      const size_t base = (i * eam_data.Nelements + j) * eam_data.nr;
      std::vector<float> y(phi_r.begin() + base, phi_r.begin() + base + eam_data.nr);
      std::vector<float4> coef;
      compute_lammps_spline(eam_data.dr, y, coef);
      std::copy(coef.begin(), coef.end(), phi_r_packed.begin() + base);
    }
  }

  eam_data.F_rho_g.resize(F_rho_packed.size());
  eam_data.rho_r_g.resize(rho_r_packed.size());
  eam_data.phi_r_g.resize(phi_r_packed.size());
  eam_data.F_rho_g.copy_from_host(F_rho_packed.data(), F_rho_packed.size());
  eam_data.rho_r_g.copy_from_host(rho_r_packed.data(), rho_r_packed.size());
  eam_data.phi_r_g.copy_from_host(phi_r_packed.data(), phi_r_packed.size());
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
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  const int* __restrict__ g_type,
  const int nr,
  const int nrho,
  const int Nelements,
  const float rc,
  const float dr,
  const float dr_inv,
  const float drho,
  const float drho_inv,
  const float4* __restrict__ F_rho_coef,
  const float4* __restrict__ rho_r_coef,
  const float4* __restrict__ phi_r_coef,
  float* d_F_rho_i,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;

  if (n1 < N2) {
    int NN = g_NN_global[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    const int i_type = g_type[n1];
    float rho = 0.0f;
    float pe_local = 0.0f;

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
        float dx = d12 - ii * dr;

        float z2 = pair_value(ii, dx, i_type, j_type, Nelements, phi_r_coef, nr);
        pe_local += (z2 / d12) * 0.5f;
        rho += spline_value(ii, dx, j_type, rho_r_coef, nr);
      }
    }

    // F(rho) lookup with LAMMPS-style p-clamping and linear extrapolation for rho > rhomax.
    // The cubic on each interval is only valid for dx ∈ [0, drho]; clamping prevents the
    // polynomial from blowing up when atoms get very close and rho exceeds the table.
    const float rhomax = (nrho - 1) * drho;
    int jj = static_cast<int>(rho * drho_inv);
    if (jj > nrho - 2)
      jj = nrho - 2;
    if (jj < 0)
      jj = 0;
    float dx_F = rho - jj * drho;
    if (dx_F > drho)
      dx_F = drho;
    if (dx_F < 0.0f)
      dx_F = 0.0f;
    float4 Fc = __ldg(&F_rho_coef[i_type * nrho + jj]);
    float F_val = Fc.x + (Fc.y + (Fc.z + Fc.w * dx_F) * dx_F) * dx_F;
    float fp_local = Fc.y + (2.0f * Fc.z + 3.0f * Fc.w * dx_F) * dx_F;
    if (rho > rhomax) {
      F_val += fp_local * (rho - rhomax);
    }
    g_pe[n1] += pe_local + F_val;
    d_F_rho_i[n1] = fp_local;
  }
}

static __global__ void find_force_eam_step2(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_type,
  const int nr,
  const int Nelements,
  const float rc,
  const float dr,
  const float dr_inv,
  const float4* __restrict__ rho_r_coef,
  const float4* __restrict__ phi_r_coef,
  const float* __restrict__ d_F_rho_i,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;

  if (n1 < N2) {
    int NN = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    const int i_type = g_type[n1];
    float Fp1 = __ldg(&d_F_rho_i[n1]);

    float fx_sum = 0.0f, fy_sum = 0.0f, fz_sum = 0.0f;
    float vxx = 0.0f, vyy = 0.0f, vzz = 0.0f;
    float vxy = 0.0f, vxz = 0.0f, vyz = 0.0f;

    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float xij = g_x[n2] - x1;
      float yij = g_y[n2] - y1;
      float zij = g_z[n2] - z1;
      apply_mic(box, xij, yij, zij);
      float r = sqrt(xij * xij + yij * yij + zij * zij);
      if (r <= rc) {
        const int j_type = __ldg(&g_type[n2]);
        float Fp2 = __ldg(&d_F_rho_i[n2]);

        int ii = static_cast<int>(r * dr_inv);
        if (ii > nr - 2)
          ii = nr - 2;
        float dx = r - ii * dr;

        float rinv = 1.0f / r;
        float z2, dz2_dr;
        pair_value_and_deriv(ii, dx, i_type, j_type, Nelements, phi_r_coef, nr, z2, dz2_dr);
        float phi_ij = z2 * rinv;
        float d_phi_r_i = (dz2_dr - phi_ij) * rinv;

        float d_F_i = spline_deriv(ii, dx, j_type, rho_r_coef, nr) * Fp1;
        float d_F_j = spline_deriv(ii, dx, i_type, rho_r_coef, nr) * Fp2;

        float fij = d_phi_r_i + d_F_i + d_F_j;
        float fx = fij * xij * rinv;
        float fy = fij * yij * rinv;
        float fz = fij * zij * rinv;

        fx_sum += fx;
        fy_sum += fy;
        fz_sum += fz;

        vxx -= fx * xij * 0.5f;
        vyy -= fy * yij * 0.5f;
        vzz -= fz * zij * 0.5f;
        vxy -= fx * yij * 0.5f;
        vxz -= fx * zij * 0.5f;
        vyz -= fy * zij * 0.5f;
      }
    }

    g_fx[n1] += fx_sum;
    g_fy[n1] += fy_sum;
    g_fz[n1] += fz_sum;
    g_virial[n1 + 0 * N] += vxx;
    g_virial[n1 + 1 * N] += vyy;
    g_virial[n1 + 2 * N] += vzz;
    g_virial[n1 + 3 * N] += vxy;
    g_virial[n1 + 4 * N] += vxz;
    g_virial[n1 + 5 * N] += vyz;
    g_virial[n1 + 6 * N] += vxy;
    g_virial[n1 + 7 * N] += vxz;
    g_virial[n1 + 8 * N] += vyz;
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

  neighbor.find_neighbor_global(eam_data.rc, box, type, position_per_atom);

  const float4* F_rho_coef = eam_data.F_rho_g.data();
  const float4* rho_r_coef = eam_data.rho_r_g.data();
  const float4* phi_r_coef = eam_data.phi_r_g.data();

  eam_data.d_F_rho_i_g.fill(0.0f);
  find_force_eam_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    neighbor.NN.data(),
    neighbor.NL.data(),
    type.data(),
    eam_data.nr,
    eam_data.nrho,
    eam_data.Nelements,
    eam_data.rc,
    eam_data.dr,
    1.0f / eam_data.dr,
    eam_data.drho,
    1.0f / eam_data.drho,
    F_rho_coef,
    rho_r_coef,
    phi_r_coef,
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
    neighbor.NN.data(),
    neighbor.NL.data(),
    type.data(),
    eam_data.nr,
    eam_data.Nelements,
    eam_data.rc,
    eam_data.dr,
    1.0f / eam_data.dr,
    rho_r_coef,
    phi_r_coef,
    eam_data.d_F_rho_i_g.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}
