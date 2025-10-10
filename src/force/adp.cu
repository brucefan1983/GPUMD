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
Angular Dependent Potential (ADP)

This extends the Embedded Atom Method (EAM) by incorporating angular forces
through dipole and quadrupole distortions of the local atomic environment.

Reference: Y. Mishin et al., Acta Materialia 53, 4029-4041 (2005)

Implemented by: Hongjian Chen (Hunan University), hjchen@hnu.edu.cn
------------------------------------------------------------------------------*/

#include "adp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <cctype>
#include <cmath>

#define BLOCK_SIZE_FORCE 128
constexpr int ADP_MAX_NEIGHBORS = 400;

constexpr int PBC_SHIFT_BIAS = 15;
constexpr int PBC_SHIFT_PAYLOAD_MASK = 0x7FFF;
constexpr int PBC_SHIFT_FLAG = 1 << 15;

__host__ __device__ __forceinline__ int encode_neighbor_shift(const int sx, const int sy, const int sz)
{
  if (sx == 0 && sy == 0 && sz == 0) {
    return 0;
  }
  const int bx = sx + PBC_SHIFT_BIAS;
  const int by = sy + PBC_SHIFT_BIAS;
  const int bz = sz + PBC_SHIFT_BIAS;
  return ((bx & 0x1F) | ((by & 0x1F) << 5) | ((bz & 0x1F) << 10)) | PBC_SHIFT_FLAG;
}

__host__ __device__ __forceinline__ void decode_neighbor_shift(const int code, int& sx, int& sy, int& sz)
{
  if (code == 0) {
    sx = sy = sz = 0;
    return;
  }
  const int payload = code & PBC_SHIFT_PAYLOAD_MASK;
  sx = ((payload & 0x1F) - PBC_SHIFT_BIAS);
  sy = (((payload >> 5) & 0x1F) - PBC_SHIFT_BIAS);
  sz = (((payload >> 10) & 0x1F) - PBC_SHIFT_BIAS);
}

__host__ __device__ __forceinline__ static int get_pair_index(int type1, int type2);

// Simple O(N^2) neighbor construction for small boxes (avoids duplicates with coarse cell lists)
static __global__ void build_neighbor_ON2(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const double rc2,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  int* __restrict__ NN,
  int* __restrict__ NL,
  int* __restrict__ shift_codes,
  const int max_neighbors)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];

  const double rc = sqrt(rc2);
  const double hx0 = box.cpu_h[0];
  const double hx1 = box.cpu_h[1];
  const double hx2 = box.cpu_h[2];
  const double hy0 = box.cpu_h[3];
  const double hy1 = box.cpu_h[4];
  const double hy2 = box.cpu_h[5];
  const double hz0 = box.cpu_h[6];
  const double hz1 = box.cpu_h[7];
  const double hz2 = box.cpu_h[8];

  int px = 0;
  if (box.pbc_x && box.thickness_x > 0.0) {
    double needed = rc / box.thickness_x - 0.5;
    if (needed > 0.0) {
      px = static_cast<int>(ceil(needed));
    } else {
      px = 1;
    }
  }
  int py = 0;
  if (box.pbc_y && box.thickness_y > 0.0) {
    double needed = rc / box.thickness_y - 0.5;
    if (needed > 0.0) {
      py = static_cast<int>(ceil(needed));
    } else {
      py = 1;
    }
  }
  int pz = 0;
  if (box.pbc_z && box.thickness_z > 0.0) {
    double needed = rc / box.thickness_z - 0.5;
    if (needed > 0.0) {
      pz = static_cast<int>(ceil(needed));
    } else {
      pz = 1;
    }
  }

  int count = 0;

  for (int n2 = 0; n2 < N; ++n2) {
    for (int iz = -pz; iz <= pz; ++iz) {
      for (int iy = -py; iy <= py; ++iy) {
        for (int ix = -px; ix <= px; ++ix) {
          if (ix == 0 && iy == 0 && iz == 0 && n2 == n1) continue;

          const double shift_x = ix * hx0 + iy * hx1 + iz * hx2;
          const double shift_y = ix * hy0 + iy * hy1 + iz * hy2;
          const double shift_z = ix * hz0 + iy * hz1 + iz * hz2;

          const double dx = (x[n2] + shift_x) - x1;
          const double dy = (y[n2] + shift_y) - y1;
          const double dz = (z[n2] + shift_z) - z1;
          const double d2 = dx * dx + dy * dy + dz * dz;

          if (d2 < rc2 && d2 > 1.0e-12) {
            if (count < max_neighbors) {
              NL[count * N + n1] = n2;
              shift_codes[count * N + n1] = encode_neighbor_shift(ix, iy, iz);
              ++count;
            }
          }
        }
      }
    }
  }

  NN[n1] = count;
}

ADP::ADP(const char* file_potential, const int number_of_atoms)
{
  initialize(file_potential, number_of_atoms);
}

ADP::~ADP(void) {}

void ADP::initialize(const char* file_potential, const int number_of_atoms)
{
  read_adp_file(file_potential);
  
  setup_spline();
  
  adp_data.inv_drho = (adp_data.drho != 0.0) ? 1.0 / adp_data.drho : 0.0;
  adp_data.inv_dr = (adp_data.dr != 0.0) ? 1.0 / adp_data.dr : 0.0;

  const int pair_table_size = adp_data.Nelements * adp_data.Nelements;
  if (pair_table_size > 0) {
    std::vector<int> pair_index_map(pair_table_size, 0);
    for (int i = 0; i < adp_data.Nelements; ++i) {
      for (int j = 0; j < adp_data.Nelements; ++j) {
        pair_index_map[i * adp_data.Nelements + j] = get_pair_index(i, j);
      }
    }
    adp_data.pair_index_map_g.resize(pair_table_size);
    adp_data.pair_index_map_g.copy_from_host(pair_index_map.data());
  }

  ensure_capacity(number_of_atoms);

  // Copy spline coefficients to GPU
  int total_rho_points = adp_data.Nelements * adp_data.nrho;
  int total_r_points = adp_data.Nelements * adp_data.nr;
  int total_pair_points = adp_data.Nelements * (adp_data.Nelements + 1) / 2 * adp_data.nr;
  
  adp_data.F_rho_a_g.resize(total_rho_points);
  adp_data.F_rho_b_g.resize(total_rho_points);
  adp_data.F_rho_c_g.resize(total_rho_points);
  adp_data.F_rho_d_g.resize(total_rho_points);
  
  adp_data.rho_r_a_g.resize(total_r_points);
  adp_data.rho_r_b_g.resize(total_r_points);
  adp_data.rho_r_c_g.resize(total_r_points);
  adp_data.rho_r_d_g.resize(total_r_points);
  
  adp_data.phi_r_a_g.resize(total_pair_points);
  adp_data.phi_r_b_g.resize(total_pair_points);
  adp_data.phi_r_c_g.resize(total_pair_points);
  adp_data.phi_r_d_g.resize(total_pair_points);
  
  adp_data.u_r_a_g.resize(total_pair_points);
  adp_data.u_r_b_g.resize(total_pair_points);
  adp_data.u_r_c_g.resize(total_pair_points);
  adp_data.u_r_d_g.resize(total_pair_points);
  
  adp_data.w_r_a_g.resize(total_pair_points);
  adp_data.w_r_b_g.resize(total_pair_points);
  adp_data.w_r_c_g.resize(total_pair_points);
  adp_data.w_r_d_g.resize(total_pair_points);
  
  // Copy spline data to GPU
  adp_data.F_rho_a_g.copy_from_host(adp_data.F_rho_a.data());
  adp_data.F_rho_b_g.copy_from_host(adp_data.F_rho_b.data());
  adp_data.F_rho_c_g.copy_from_host(adp_data.F_rho_c.data());
  adp_data.F_rho_d_g.copy_from_host(adp_data.F_rho_d.data());
  
  adp_data.rho_r_a_g.copy_from_host(adp_data.rho_r_a.data());
  adp_data.rho_r_b_g.copy_from_host(adp_data.rho_r_b.data());
  adp_data.rho_r_c_g.copy_from_host(adp_data.rho_r_c.data());
  adp_data.rho_r_d_g.copy_from_host(adp_data.rho_r_d.data());
  
  adp_data.phi_r_a_g.copy_from_host(adp_data.phi_r_a.data());
  adp_data.phi_r_b_g.copy_from_host(adp_data.phi_r_b.data());
  adp_data.phi_r_c_g.copy_from_host(adp_data.phi_r_c.data());
  adp_data.phi_r_d_g.copy_from_host(adp_data.phi_r_d.data());
  
  adp_data.u_r_a_g.copy_from_host(adp_data.u_r_a.data());
  adp_data.u_r_b_g.copy_from_host(adp_data.u_r_b.data());
  adp_data.u_r_c_g.copy_from_host(adp_data.u_r_c.data());
  adp_data.u_r_d_g.copy_from_host(adp_data.u_r_d.data());
  
  adp_data.w_r_a_g.copy_from_host(adp_data.w_r_a.data());
  adp_data.w_r_b_g.copy_from_host(adp_data.w_r_b.data());
  adp_data.w_r_c_g.copy_from_host(adp_data.w_r_c.data());
  adp_data.w_r_d_g.copy_from_host(adp_data.w_r_d.data());
}

void ADP::ensure_capacity(int number_of_atoms)
{
  if (number_of_atoms <= 0) {
    return;
  }

  auto ensure_int_capacity = [](GPU_Vector<int>& vec, int required) {
    if (vec.size() < required) {
      vec.resize(required);
    }
  };

  auto ensure_double_capacity = [](GPU_Vector<double>& vec, int required) {
    if (vec.size() < required) {
      vec.resize(required);
    }
  };

  ensure_int_capacity(adp_data.NN, number_of_atoms);
  ensure_int_capacity(adp_data.cell_count, number_of_atoms);
  ensure_int_capacity(adp_data.cell_count_sum, number_of_atoms);
  ensure_int_capacity(adp_data.cell_contents, number_of_atoms);

  const int neighbor_capacity = number_of_atoms * ADP_MAX_NEIGHBORS;
  if (adp_data.NL.size() < neighbor_capacity) {
    adp_data.NL.resize(neighbor_capacity);
  }
  if (adp_data.NL_shift.size() < neighbor_capacity) {
    adp_data.NL_shift.resize(neighbor_capacity);
  }

  ensure_double_capacity(adp_data.Fp, number_of_atoms);
  ensure_double_capacity(adp_data.mu, number_of_atoms * 3);
  ensure_double_capacity(adp_data.lambda, number_of_atoms * 6);
}

void ADP::read_adp_file(const char* file_potential)
{
  std::ifstream input_file(file_potential);
  if (!input_file.is_open()) {
    printf("Error: Cannot open ADP potential file: %s\n", file_potential);
    PRINT_INPUT_ERROR("Cannot open ADP potential file.");
  }
  
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input_file, line)) {
    lines.push_back(line);
  }
  input_file.close();
  
  // Parse all data words starting from line 4 (lines 1-3 are comments)
  std::vector<std::string> data_words;
  for (size_t i = 3; i < lines.size(); ++i) {
    std::istringstream iss(lines[i]);
    std::string word;
    while (iss >> word) {
      data_words.push_back(word);
    }
  }
  
  // Line 4: Nelements Element1 Element2 ... ElementN
  size_t index = 0;
  adp_data.Nelements = std::stoi(data_words[index++]);
  adp_data.elements_list.assign(
    data_words.begin() + index, data_words.begin() + index + adp_data.Nelements);
  index += adp_data.Nelements;
  
  // Line 5: Nrho, drho, Nr, dr, cutoff
  adp_data.nrho = std::stoi(data_words[index++]);
  adp_data.drho = std::stod(data_words[index++]);
  adp_data.nr = std::stoi(data_words[index++]);
  adp_data.dr = std::stod(data_words[index++]);
  adp_data.rc = std::stod(data_words[index++]);
  
  printf("Use %d-element ADP potential with element(s): ", adp_data.Nelements);
  for (int i = 0; i < adp_data.Nelements; ++i) {
    printf("%s ", adp_data.elements_list[i].c_str());
  }
  printf("\n");
  
  adp_data.F_rho.resize(adp_data.Nelements * adp_data.nrho);
  adp_data.rho_r.resize(adp_data.Nelements * adp_data.nr);
  
  // Read element-specific data
  for (int element = 0; element < adp_data.Nelements; element++) {
    // Skip: atomic number, mass, lattice constant, lattice type
    index += 4;
    
    // Read embedding function F(rho) for this element
    int base_rho = element * adp_data.nrho;
    for (int i = 0; i < adp_data.nrho; i++) {
      adp_data.F_rho[base_rho + i] = std::stod(data_words[index++]);
    }
    
    // Read density function rho(r) for this element
    int base_r = element * adp_data.nr;
    for (int i = 0; i < adp_data.nr; i++) {
      adp_data.rho_r[base_r + i] = std::stod(data_words[index++]);
    }
  }
  
  // Read pairwise interactions phi(r), u(r), w(r)
  int num_pairs = adp_data.Nelements * (adp_data.Nelements + 1) / 2;
  adp_data.phi_r.resize(num_pairs * adp_data.nr);
  adp_data.u_r.resize(num_pairs * adp_data.nr);
  adp_data.w_r.resize(num_pairs * adp_data.nr);
  
  int pair_index = 0;
  for (int i = 0; i < adp_data.Nelements; i++) {
    for (int j = 0; j <= i; j++) {
      int base_pair = pair_index * adp_data.nr;
      
      // Read phi(r) - pair potential (multiplied by r)
      for (int k = 0; k < adp_data.nr; k++) {
        adp_data.phi_r[base_pair + k] = std::stod(data_words[index++]);
      }
      
      pair_index++;
    }
  }
  
  // Read u(r) functions
  pair_index = 0;
  for (int i = 0; i < adp_data.Nelements; i++) {
    for (int j = 0; j <= i; j++) {
      int base_pair = pair_index * adp_data.nr;
      
      for (int k = 0; k < adp_data.nr; k++) {
        adp_data.u_r[base_pair + k] = std::stod(data_words[index++]);
      }
      
      pair_index++;
    }
  }
  
  // Read w(r) functions
  pair_index = 0;
  for (int i = 0; i < adp_data.Nelements; i++) {
    for (int j = 0; j <= i; j++) {
      int base_pair = pair_index * adp_data.nr;
      
      for (int k = 0; k < adp_data.nr; k++) {
        adp_data.w_r[base_pair + k] = std::stod(data_words[index++]);
      }
      
      pair_index++;
    }
  }
}

void ADP::setup_spline()
{
  const int total_rho_points = adp_data.Nelements * adp_data.nrho;
  adp_data.F_rho_a.resize(total_rho_points);
  adp_data.F_rho_b.resize(total_rho_points);
  adp_data.F_rho_c.resize(total_rho_points);
  adp_data.F_rho_d.resize(total_rho_points);
  
  const int total_r_points = adp_data.Nelements * adp_data.nr;
  adp_data.rho_r_a.resize(total_r_points);
  adp_data.rho_r_b.resize(total_r_points);
  adp_data.rho_r_c.resize(total_r_points);
  adp_data.rho_r_d.resize(total_r_points);
  
  const int total_pair_points = adp_data.Nelements * (adp_data.Nelements + 1) / 2 * adp_data.nr;
  adp_data.phi_r_a.resize(total_pair_points);
  adp_data.phi_r_b.resize(total_pair_points);
  adp_data.phi_r_c.resize(total_pair_points);
  adp_data.phi_r_d.resize(total_pair_points);
  
  adp_data.u_r_a.resize(total_pair_points);
  adp_data.u_r_b.resize(total_pair_points);
  adp_data.u_r_c.resize(total_pair_points);
  adp_data.u_r_d.resize(total_pair_points);
  
  adp_data.w_r_a.resize(total_pair_points);
  adp_data.w_r_b.resize(total_pair_points);
  adp_data.w_r_c.resize(total_pair_points);
  adp_data.w_r_d.resize(total_pair_points);
  
  calculate_spline(
    adp_data.F_rho.data(), adp_data.drho,
    adp_data.F_rho_a.data(), adp_data.F_rho_b.data(), adp_data.F_rho_c.data(), adp_data.F_rho_d.data(),
    adp_data.Nelements, adp_data.nrho);
  calculate_spline(
    adp_data.rho_r.data(), adp_data.dr,
    adp_data.rho_r_a.data(), adp_data.rho_r_b.data(), adp_data.rho_r_c.data(), adp_data.rho_r_d.data(),
    adp_data.Nelements, adp_data.nr);
  calculate_spline(
    adp_data.phi_r.data(), adp_data.dr,
    adp_data.phi_r_a.data(), adp_data.phi_r_b.data(), adp_data.phi_r_c.data(), adp_data.phi_r_d.data(),
    adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
  calculate_spline(
    adp_data.u_r.data(), adp_data.dr,
    adp_data.u_r_a.data(), adp_data.u_r_b.data(), adp_data.u_r_c.data(), adp_data.u_r_d.data(),
    adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
  calculate_spline(
    adp_data.w_r.data(), adp_data.dr,
    adp_data.w_r_a.data(), adp_data.w_r_b.data(), adp_data.w_r_c.data(), adp_data.w_r_d.data(),
    adp_data.Nelements * (adp_data.Nelements + 1) / 2, adp_data.nr);
}

__device__ __forceinline__ static void interpolate(
  const double* __restrict__ a,
  const double* __restrict__ b,
  const double* __restrict__ c,
  const double* __restrict__ d,
  const int index,
  const double x_frac,
  const double dx,
  double& y,
  double& yp)
{
  const double t = x_frac * dx;
  y = a[index] + t * (b[index] + t * (c[index] + t * d[index]));
  yp = b[index] + t * (2.0 * c[index] + 3.0 * t * d[index]);
}

__device__ __forceinline__ static double interpolate_value(
  const double* __restrict__ a,
  const double* __restrict__ b,
  const double* __restrict__ c,
  const double* __restrict__ d,
  const int index,
  const double t)
{
  return a[index] + t * (b[index] + t * (c[index] + t * d[index]));
}

void ADP::calculate_spline(
  const double* y, double dx,
  double* a, double* b, double* c, double* d,
  int n_functions, int n_points)
{
  for (int func = 0; func < n_functions; ++func) {
    int off = func * n_points;
    const double* yf = y + off;
    double* af = a + off;
    double* bf = b + off;
    double* cf = c + off;
    double* df = d + off;

    std::vector<double> s(n_points, 0.0);
    if (n_points >= 2) s[0] = yf[1] - yf[0];
    if (n_points >= 3) s[1] = 0.5 * (yf[2] - yf[0]);
    for (int i = 2; i <= n_points - 3; ++i) {
      s[i] = ((yf[i-2] - yf[i+2]) + 8.0 * (yf[i+1] - yf[i-1])) / 12.0;
    }
    if (n_points >= 3) s[n_points - 2] = 0.5 * (yf[n_points - 1] - yf[n_points - 3]);
    if (n_points >= 2) s[n_points - 1] = yf[n_points - 1] - yf[n_points - 2];

    for (int m = 0; m < n_points - 1; ++m) {
      double dy = yf[m+1] - yf[m];
      double c4 = 3.0 * dy - 2.0 * s[m] - s[m+1];
      double c3 = s[m] + s[m+1] - 2.0 * dy;
      af[m] = yf[m];
      bf[m] = s[m] / dx;
      cf[m] = c4 / (dx * dx);
      df[m] = c3 / (dx * dx * dx);
    }
    af[n_points-1] = yf[n_points-1];
    bf[n_points-1] = 0.0;
    cf[n_points-1] = 0.0;
    df[n_points-1] = 0.0;
  }
}

// Get pair index for element types (0-based) consistent with reading order (i loop outer, j<=i)
__host__ __device__ __forceinline__ static int get_pair_index(int type1, int type2)
{
  int a = (type1 >= type2) ? type1 : type2; // ensure a >= b
  int b = (type1 >= type2) ? type2 : type1;
  // sequence: (0,0)=0; (1,0)=1,(1,1)=2; (2,0)=3,(2,1)=4,(2,2)=5; index = a*(a+1)/2 + b
  return a * (a + 1) / 2 + b;
}

// Calculate density and dipole/quadruple terms
static __global__ void find_force_adp_step1(
  const int N,
  const int N1,
  const int N2,
  const int Nelements,
  const int nrho,
  const double drho,
  const double inv_drho,
  const int nr,
  const double dr,
  const double inv_dr,
  const double rc,
  const Box box,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_shift,
  const int* __restrict__ g_type,
  const int* __restrict__ pair_index_map,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* __restrict__ F_rho_a,
  const double* __restrict__ F_rho_b,
  const double* __restrict__ F_rho_c,
  const double* __restrict__ F_rho_d,
  const double* __restrict__ rho_r_a,
  const double* __restrict__ rho_r_b,
  const double* __restrict__ rho_r_c,
  const double* __restrict__ rho_r_d,
  const double* __restrict__ u_r_a,
  const double* __restrict__ u_r_b,
  const double* __restrict__ u_r_c,
  const double* __restrict__ u_r_d,
  const double* __restrict__ w_r_a,
  const double* __restrict__ w_r_b,
  const double* __restrict__ w_r_c,
  const double* __restrict__ w_r_d,
  double* __restrict__ g_Fp,
  double* __restrict__ g_mu,
  double* __restrict__ g_lambda,
  double* __restrict__ g_pe)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  const int NN = g_NN[n1];
  const int type1 = g_type[n1];

  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];

  const double hx0 = box.cpu_h[0];
  const double hx1 = box.cpu_h[1];
  const double hx2 = box.cpu_h[2];
  const double hy0 = box.cpu_h[3];
  const double hy1 = box.cpu_h[4];
  const double hy2 = box.cpu_h[5];
  const double hz0 = box.cpu_h[6];
  const double hz1 = box.cpu_h[7];
  const double hz2 = box.cpu_h[8];

  const double rc2 = rc * rc;
  const int* pair_row = pair_index_map + type1 * Nelements;

  double rho = 0.0;
  double mu_x = 0.0;
  double mu_y = 0.0;
  double mu_z = 0.0;
  double lambda_xx = 0.0;
  double lambda_yy = 0.0;
  double lambda_zz = 0.0;
  double lambda_xy = 0.0;
  double lambda_xz = 0.0;
  double lambda_yz = 0.0;

  for (int i1 = 0; i1 < NN; ++i1) {
    const int idx = i1 * N + n1;
    const int n2 = g_NL[idx];
    const int type2 = g_type[n2];
    const double x2 = g_x[n2];
    const double y2 = g_y[n2];
    const double z2 = g_z[n2];

    double delx = x1 - x2;
    double dely = y1 - y2;
    double delz = z1 - z2;

    if (g_shift) {
      const int code = g_shift[idx];
      int sx, sy, sz;
      decode_neighbor_shift(code, sx, sy, sz);
      delx -= sx * hx0 + sy * hx1 + sz * hx2;
      dely -= sx * hy0 + sy * hy1 + sz * hy2;
      delz -= sx * hz0 + sy * hz1 + sz * hz2;
    } else {
      apply_mic(box, delx, dely, delz);
    }

    const double d2 = delx * delx + dely * dely + delz * delz;
    if (d2 >= rc2 || d2 <= 1.0e-24) {
      continue;
    }

    const double inv_r = rsqrt(d2);
    const double d12 = d2 * inv_r;
    const double pp = d12 * inv_dr + 1.0;
    int m = static_cast<int>(pp);
    m = (m < 1) ? 1 : ((m > nr - 1) ? nr - 1 : m);
    const double frac = (pp - m > 1.0) ? 1.0 : (pp - m);
    const int ir = m - 1;
    const double interp_t = frac * dr;

    const int rho_offset = type2 * nr + ir;
    const int pair_offset = pair_row[type2] * nr + ir;

    rho += interpolate_value(rho_r_a, rho_r_b, rho_r_c, rho_r_d, rho_offset, interp_t);
    const double u_val = interpolate_value(u_r_a, u_r_b, u_r_c, u_r_d, pair_offset, interp_t);
    const double w_val = interpolate_value(w_r_a, w_r_b, w_r_c, w_r_d, pair_offset, interp_t);

    mu_x += u_val * delx;
    mu_y += u_val * dely;
    mu_z += u_val * delz;

    const double delx2 = delx * delx;
    const double dely2 = dely * dely;
    const double delz2 = delz * delz;
    lambda_xx += w_val * delx2;
    lambda_yy += w_val * dely2;
    lambda_zz += w_val * delz2;
    lambda_xy += w_val * delx * dely;
    lambda_xz += w_val * delx * delz;
    lambda_yz += w_val * dely * delz;
  }

  const int F_base = type1 * nrho;
  const double pp = rho * inv_drho + 1.0;
  int m = static_cast<int>(pp);
  m = (m < 1) ? 1 : ((m > nrho - 1) ? nrho - 1 : m);
  const double frac = (pp - m > 1.0) ? 1.0 : (pp - m);
  const int irho = m - 1;

  double F = 0.0;
  double Fp = 0.0;
  interpolate(
    F_rho_a,
    F_rho_b,
    F_rho_c,
    F_rho_d,
    F_base + irho,
    frac,
    drho,
    F,
    Fp);

  const double mu_squared = mu_x * mu_x + mu_y * mu_y + mu_z * mu_z;
  const double lambda_diagonal = lambda_xx * lambda_xx + lambda_yy * lambda_yy + lambda_zz * lambda_zz;
  const double lambda_offdiag = lambda_xy * lambda_xy + lambda_xz * lambda_xz + lambda_yz * lambda_yz;
  const double nu = lambda_xx + lambda_yy + lambda_zz;
  const double adp_energy = 0.5 * mu_squared + 0.5 * lambda_diagonal + lambda_offdiag - (nu * nu) / 6.0;

  g_pe[n1] += F + adp_energy;
  g_Fp[n1] = Fp;

  g_mu[n1] = mu_x;
  g_mu[n1 + N] = mu_y;
  g_mu[n1 + 2 * N] = mu_z;

  g_lambda[n1] = lambda_xx;
  g_lambda[n1 + N] = lambda_yy;
  g_lambda[n1 + 2 * N] = lambda_zz;
  g_lambda[n1 + 3 * N] = lambda_yz;
  g_lambda[n1 + 4 * N] = lambda_xz;
  g_lambda[n1 + 5 * N] = lambda_xy;
}

// Calculate forces
static __global__ void find_force_adp_step2(
  const int N,
  const int N1,
  const int N2,
  const int Nelements,
  const int nr,
  const double dr,
  const double inv_dr,
  const double rc,
  const Box box,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_shift,
  const int* __restrict__ g_type,
  const int* __restrict__ pair_index_map,
  const double* __restrict__ g_Fp,
  const double* __restrict__ g_mu,
  const double* __restrict__ g_lambda,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double* __restrict__ rho_r_a,
  const double* __restrict__ rho_r_b,
  const double* __restrict__ rho_r_c,
  const double* __restrict__ rho_r_d,
  const double* __restrict__ phi_r_a,
  const double* __restrict__ phi_r_b,
  const double* __restrict__ phi_r_c,
  const double* __restrict__ phi_r_d,
  const double* __restrict__ u_r_a,
  const double* __restrict__ u_r_b,
  const double* __restrict__ u_r_c,
  const double* __restrict__ u_r_d,
  const double* __restrict__ w_r_a,
  const double* __restrict__ w_r_b,
  const double* __restrict__ w_r_c,
  const double* __restrict__ w_r_d,
  double* __restrict__ g_fx,
  double* __restrict__ g_fy,
  double* __restrict__ g_fz,
  double* __restrict__ g_virial,
  double* __restrict__ g_pe)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  const int NN = g_NN[n1];
  const int type1 = g_type[n1];

  const double x1 = g_x[n1];
  const double y1 = g_y[n1];
  const double z1 = g_z[n1];

  const double hx0 = box.cpu_h[0];
  const double hx1 = box.cpu_h[1];
  const double hx2 = box.cpu_h[2];
  const double hy0 = box.cpu_h[3];
  const double hy1 = box.cpu_h[4];
  const double hy2 = box.cpu_h[5];
  const double hz0 = box.cpu_h[6];
  const double hz1 = box.cpu_h[7];
  const double hz2 = box.cpu_h[8];

  const double rc2 = rc * rc;
  const int* pair_row = pair_index_map + type1 * Nelements;

  const double* mu_x = g_mu;
  const double* mu_y = g_mu + N;
  const double* mu_z = g_mu + 2 * N;
  const double* lambda_xx = g_lambda;
  const double* lambda_yy = g_lambda + N;
  const double* lambda_zz = g_lambda + 2 * N;
  const double* lambda_yz = g_lambda + 3 * N;
  const double* lambda_xz = g_lambda + 4 * N;
  const double* lambda_xy = g_lambda + 5 * N;

  const double mu1_x = mu_x[n1];
  const double mu1_y = mu_y[n1];
  const double mu1_z = mu_z[n1];
  const double lambda1_xx = lambda_xx[n1];
  const double lambda1_yy = lambda_yy[n1];
  const double lambda1_zz = lambda_zz[n1];
  const double lambda1_yz = lambda_yz[n1];
  const double lambda1_xz = lambda_xz[n1];
  const double lambda1_xy = lambda_xy[n1];

  double fx = 0.0;
  double fy = 0.0;
  double fz = 0.0;
  double pe = 0.0;

  double s_sxx = 0.0;
  double s_syy = 0.0;
  double s_szz = 0.0;
  double s_sxy = 0.0;
  double s_sxz = 0.0;
  double s_syz = 0.0;
  double s_syx = 0.0;
  double s_szx = 0.0;
  double s_szy = 0.0;

  const double Fp1 = g_Fp[n1];
  const int rho1_base = type1 * nr;

  for (int i1 = 0; i1 < NN; ++i1) {
    const int idx = i1 * N + n1;
    const int n2 = g_NL[idx];
    const int type2 = g_type[n2];
    const double x2 = g_x[n2];
    const double y2 = g_y[n2];
    const double z2 = g_z[n2];

    double delx = x1 - x2;
    double dely = y1 - y2;
    double delz = z1 - z2;

    if (g_shift) {
      const int code = g_shift[idx];
      int sx, sy, sz;
      decode_neighbor_shift(code, sx, sy, sz);
      delx -= sx * hx0 + sy * hx1 + sz * hx2;
      dely -= sx * hy0 + sy * hy1 + sz * hy2;
      delz -= sx * hz0 + sy * hz1 + sz * hz2;
    } else {
      apply_mic(box, delx, dely, delz);
    }

    const double d2 = delx * delx + dely * dely + delz * delz;
    if (d2 >= rc2 || d2 <= 1.0e-24) {
      continue;
    }

    const double inv_r = rsqrt(d2);
    const double d12 = d2 * inv_r;
    const double pp = d12 * inv_dr + 1.0;
    int m = static_cast<int>(pp);
    m = (m < 1) ? 1 : ((m > nr - 1) ? nr - 1 : m);
    const double frac = (pp - m > 1.0) ? 1.0 : (pp - m);
    const int ir = m - 1;

    const int rho2_base = type2 * nr;
    const int pair_offset = pair_row[type2] * nr + ir;

    double rho_unused, rho1_deriv, rho2_deriv;
    interpolate(rho_r_a, rho_r_b, rho_r_c, rho_r_d, rho2_base + ir, frac, dr, rho_unused, rho1_deriv);
    interpolate(rho_r_a, rho_r_b, rho_r_c, rho_r_d, rho1_base + ir, frac, dr, rho_unused, rho2_deriv);

    double phi_rphi_val, phi_rphi_deriv, u_val, u_deriv, w_val, w_deriv;
    interpolate(phi_r_a, phi_r_b, phi_r_c, phi_r_d, pair_offset, frac, dr, phi_rphi_val, phi_rphi_deriv);
    interpolate(u_r_a, u_r_b, u_r_c, u_r_d, pair_offset, frac, dr, u_val, u_deriv);
    interpolate(w_r_a, w_r_b, w_r_c, w_r_d, pair_offset, frac, dr, w_val, w_deriv);

    const double phi_val = phi_rphi_val * inv_r;
    const double phi_deriv = phi_rphi_deriv * inv_r - phi_rphi_val * (inv_r * inv_r);

    const double Fp2 = g_Fp[n2];
    const double psip = Fp1 * rho1_deriv + Fp2 * rho2_deriv;
    const double scalar = -(phi_deriv + psip);

    const double mu2_x = mu_x[n2];
    const double mu2_y = mu_y[n2];
    const double mu2_z = mu_z[n2];

    const double lambda2_xx = lambda_xx[n2];
    const double lambda2_yy = lambda_yy[n2];
    const double lambda2_zz = lambda_zz[n2];
    const double lambda2_yz = lambda_yz[n2];
    const double lambda2_xz = lambda_xz[n2];
    const double lambda2_xy = lambda_xy[n2];

    const double delmux = mu1_x - mu2_x;
    const double delmuy = mu1_y - mu2_y;
    const double delmuz = mu1_z - mu2_z;
    const double trdelmu = delmux * delx + delmuy * dely + delmuz * delz;

    const double sumlamxx = lambda1_xx + lambda2_xx;
    const double sumlamyy = lambda1_yy + lambda2_yy;
    const double sumlamzz = lambda1_zz + lambda2_zz;
    const double sumlamyz = lambda1_yz + lambda2_yz;
    const double sumlamxz = lambda1_xz + lambda2_xz;
    const double sumlamxy = lambda1_xy + lambda2_xy;

    const double tradellam =
      sumlamxx * delx * delx + sumlamyy * dely * dely + sumlamzz * delz * delz +
      2.0 * sumlamxy * delx * dely + 2.0 * sumlamxz * delx * delz +
      2.0 * sumlamyz * dely * delz;
    const double nu = sumlamxx + sumlamyy + sumlamzz;
    const double w_scale = w_deriv * d12 + 2.0 * w_val;
    const double nu_w_third = (1.0 / 3.0) * nu * w_scale;
    const double w_val_2 = 2.0 * w_val;
    const double trdelmu_u_deriv_inv_r = trdelmu * u_deriv * inv_r;
    const double w_deriv_inv_r_tradellam = w_deriv * inv_r * tradellam;

    const double force_total_x = scalar * delx * inv_r - 
      (delmux * u_val + trdelmu_u_deriv_inv_r * delx +
       w_val_2 * (sumlamxx * delx + sumlamxy * dely + sumlamxz * delz) +
       w_deriv_inv_r_tradellam * delx - nu_w_third * delx);
    const double force_total_y = scalar * dely * inv_r - 
      (delmuy * u_val + trdelmu_u_deriv_inv_r * dely +
       w_val_2 * (sumlamxy * delx + sumlamyy * dely + sumlamyz * delz) +
       w_deriv_inv_r_tradellam * dely - nu_w_third * dely);
    const double force_total_z = scalar * delz * inv_r - 
      (delmuz * u_val + trdelmu_u_deriv_inv_r * delz +
       w_val_2 * (sumlamxz * delx + sumlamyz * dely + sumlamzz * delz) +
       w_deriv_inv_r_tradellam * delz - nu_w_third * delz);

    fx += force_total_x;
    fy += force_total_y;
    fz += force_total_z;
    pe += 0.5 * phi_val;

    const double half_vxx = 0.5 * delx * force_total_x;
    const double half_vyy = 0.5 * dely * force_total_y;
    const double half_vzz = 0.5 * delz * force_total_z;
    const double half_vxy = 0.5 * delx * force_total_y;
    const double half_vxz = 0.5 * delx * force_total_z;
    const double half_vyz = 0.5 * dely * force_total_z;

    s_sxx += half_vxx;
    s_syy += half_vyy;
    s_szz += half_vzz;
    s_sxy += half_vxy;
    s_sxz += half_vxz;
    s_syz += half_vyz;
    s_syx += half_vxy;
    s_szx += half_vxz;
    s_szy += half_vyz;
  }

  g_fx[n1] += fx;
  g_fy[n1] += fy;
  g_fz[n1] += fz;

  g_virial[n1 + 0 * N] += s_sxx;
  g_virial[n1 + 1 * N] += s_syy;
  g_virial[n1 + 2 * N] += s_szz;
  g_virial[n1 + 3 * N] += s_sxy;
  g_virial[n1 + 4 * N] += s_sxz;
  g_virial[n1 + 5 * N] += s_syz;
  g_virial[n1 + 6 * N] += s_syx;
  g_virial[n1 + 7 * N] += s_szx;
  g_virial[n1 + 8 * N] += s_szy;
  g_pe[n1] += pe;
}

void ADP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  ensure_capacity(number_of_atoms);
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
  const int* neighbor_shift_ptr = nullptr;
  const int* type_ptr = type.data();
  
  {
    int nbins[3];
    bool small_box = box.get_num_bins(0.5 * adp_data.rc, nbins);
    
    // Use O(N) cell list for normal boxes, O(N^2) brute force when rc > 0.5*box_size
    const int max_neighbors = number_of_atoms > 0 ? adp_data.NL.size() / number_of_atoms : ADP_MAX_NEIGHBORS;
    neighbor_shift_ptr = nullptr;
    
    if (!small_box) {
      find_neighbor(
        N1,
        N2,
        adp_data.rc,
        box,
        type,
        position_per_atom,
        adp_data.cell_count,
        adp_data.cell_count_sum,
        adp_data.cell_contents,
        adp_data.NN,
        adp_data.NL);
    } else {
      const double* gx = position_per_atom.data();
      const double* gy = position_per_atom.data() + number_of_atoms;
      const double* gz = position_per_atom.data() + number_of_atoms * 2;
      const int block = 256;
      const int grid = (N2 - N1 - 1) / block + 1;
      build_neighbor_ON2<<<grid, block>>>(
        box,
        number_of_atoms,
        N1,
        N2,
        adp_data.rc * adp_data.rc,
        gx, gy, gz,
        adp_data.NN.data(),
        adp_data.NL.data(),
        adp_data.NL_shift.data(),
        max_neighbors);
      GPU_CHECK_KERNEL
      neighbor_shift_ptr = adp_data.NL_shift.data();
    }
  }


  // Zero only internal accumulators needed for this step; top-level framework already zeroed force/potential/virial.
  adp_data.mu.fill(0.0);      // size 3N
  adp_data.lambda.fill(0.0);  // size 6N
  adp_data.Fp.fill(0.0);      // size N
  
  // Step 1: Calculate density, embedding energy, and angular terms
  find_force_adp_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    adp_data.Nelements,
    adp_data.nrho,
    adp_data.drho,
    adp_data.inv_drho,
    adp_data.nr,
    adp_data.dr,
    adp_data.inv_dr,
    adp_data.rc,
    box,
    adp_data.NN.data(),
    adp_data.NL.data(),
    neighbor_shift_ptr,
    type_ptr,
    adp_data.pair_index_map_g.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    adp_data.F_rho_a_g.data(),
    adp_data.F_rho_b_g.data(),
    adp_data.F_rho_c_g.data(),
    adp_data.F_rho_d_g.data(),
    adp_data.rho_r_a_g.data(),
    adp_data.rho_r_b_g.data(),
    adp_data.rho_r_c_g.data(),
    adp_data.rho_r_d_g.data(),
    adp_data.u_r_a_g.data(),
    adp_data.u_r_b_g.data(),
    adp_data.u_r_c_g.data(),
    adp_data.u_r_d_g.data(),
    adp_data.w_r_a_g.data(),
    adp_data.w_r_b_g.data(),
    adp_data.w_r_c_g.data(),
    adp_data.w_r_d_g.data(),
    adp_data.Fp.data(),
    adp_data.mu.data(),
    adp_data.lambda.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
  
  // Step 2: Calculate forces
  find_force_adp_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    adp_data.Nelements,
    adp_data.nr,
    adp_data.dr,
    adp_data.inv_dr,
    adp_data.rc,
    box,
    adp_data.NN.data(),
    adp_data.NL.data(),
    neighbor_shift_ptr,
    type_ptr,
    adp_data.pair_index_map_g.data(),
    adp_data.Fp.data(),
    adp_data.mu.data(),
    adp_data.lambda.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    adp_data.rho_r_a_g.data(),
    adp_data.rho_r_b_g.data(),
    adp_data.rho_r_c_g.data(),
    adp_data.rho_r_d_g.data(),
    adp_data.phi_r_a_g.data(),
    adp_data.phi_r_b_g.data(),
    adp_data.phi_r_c_g.data(),
    adp_data.phi_r_d_g.data(),
    adp_data.u_r_a_g.data(),
    adp_data.u_r_b_g.data(),
    adp_data.u_r_c_g.data(),
    adp_data.u_r_d_g.data(),
    adp_data.w_r_a_g.data(),
    adp_data.w_r_b_g.data(),
    adp_data.w_r_c_g.data(),
    adp_data.w_r_d_g.data(),
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
}
