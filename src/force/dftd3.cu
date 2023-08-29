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
DFT-D3:

[1] Stefan Grimme, Jens Antony, Stephan Ehrlich, Helge Krieg,
A consistent and accurate ab initio parametrization of density functional
dispersion correction (DFT-D) for the 94 elements H-Pu,
J. Chem. Phys. 132, 154104 (2010).

[2] Stefan Grimme, Stephan Ehrlich, Lars Goerigk,
Effect of the damping function in dispersion corrected density functional
theory,
J. Comput. Chem., 32, 1456 (2011).
------------------------------------------------------------------------------*/

#include "dftd3.cuh"
#include "dftd3para.cuh"
#include "model/box.cuh"
#include <iostream>
#include <string>
#include <vector>

namespace
{
const int MN = 10000; // maximum number of neighbors for one atom
const int NUM_ELEMENTS = 103;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

void __global__ find_dftd3_coordination_number_small_box(
  DFTD3::DFTD3_Para dftd3_para,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  float* g_cn)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int z1 = dftd3_para.atomic_number[g_type[n1]];
    float R_cov_1 = Bohr * covalent_radius[z1];
    float cn_temp = 0.0f;
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int z2 = dftd3_para.atomic_number[g_type[n2]];
      float R_cov_2 = Bohr * covalent_radius[z2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      cn_temp += 1.0f / (exp(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f)) + 1.0f);
    }
    g_cn[n1] = cn_temp;
  }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

void __global__ add_dftd3_force_small_box(
  DFTD3::DFTD3_Para dftd3_para,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_type,
  const float* g_c6_ref,
  const float* g_cn,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  double* g_potential,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  float* g_dc6_sum,
  float* g_dc8_sum)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int z1 = dftd3_para.atomic_number[g_type[n1]];
    int num_cn_1 = num_cn[z1];
    float dc6_sum = 0.0f;
    float dc8_sum = 0.0f;
    float s_potential = 0.0f;
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int z2 = dftd3_para.atomic_number[g_type[n2]];
      int z_small = z1, z_large = z2;
      if (z1 > z2) {
        z_small = z2;
        z_large = z1;
      }
      int z12 = z_small * max_elem - (z_small * (z_small - 1)) / 2 + (z_large - z_small);
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      float d12_4 = d12_2 * d12_2;
      float d12_6 = d12_4 * d12_2;
      float d12_8 = d12_6 * d12_2;
      float c6 = 0.0f;
      float dc6 = 0.0f;
      int num_cn_2 = num_cn[z2];
      if (num_cn_1 == 1 && num_cn_2 == 1) {
        c6 = g_c6_ref[z12 * max_cn2];
      } else {
        float W = 0.0f;
        float dW = 0.0f;
        float Z = 0.0f;
        float dZ = 0.0f;
        for (int i = 0; i < num_cn_1; ++i) {
          for (int j = 0; j < num_cn_2; ++j) {
            float diff_i = g_cn[n1] - cn_ref[z1 * max_cn + i];
            float diff_j = g_cn[n2] - cn_ref[z2 * max_cn + j];
            float L_ij = exp(-4.0f * (diff_i * diff_i + diff_j * diff_j));
            if (L_ij == 0.0f) {
              L_ij = 1.0e-37f;
            }
            W += L_ij;
            dW += L_ij * (-8.0f * diff_i);
            float c6_ref_ij = (z1 < z2) ? g_c6_ref[z12 * max_cn2 + i * max_cn + j]
                                        : g_c6_ref[z12 * max_cn2 + j * max_cn + i];
            Z += c6_ref_ij * L_ij;
            dZ += c6_ref_ij * L_ij * (-8.0f * diff_i);
          }
        }
        W = 1.0f / W;
        c6 = Z * W;
        dc6 = dZ * W - c6 * dW * W;
      }
      c6 *= HartreeBohr6;
      dc6 *= HartreeBohr6;

      float c8_over_c6 = 3.0f * r2r4[z1] * r2r4[z2] * Bohr2;
      float c8 = c6 * c8_over_c6;
      float damp = dftd3_para.a1 * sqrt(c8_over_c6) + dftd3_para.a2;
      float damp_2 = damp * damp;
      float damp_4 = damp_2 * damp_2;
      float damp_6 = 1.0f / (d12_6 + damp_4 * damp_2);
      float damp_8 = 1.0f / (d12_8 + damp_4 * damp_4);
      s_potential -= (dftd3_para.s6 * c6 * damp_6 + dftd3_para.s8 * c8 * damp_8) * 0.5f;
      float f2 = dftd3_para.s6 * c6 * 3.0f * d12_4 * (damp_6 * damp_6) +
                 dftd3_para.s8 * c8 * 4.0f * d12_6 * (damp_8 * damp_8);
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      atomicAdd(&g_virial[n2 + 0 * N], double(-r12[0] * f12[0]));
      atomicAdd(&g_virial[n2 + 1 * N], double(-r12[1] * f12[1]));
      atomicAdd(&g_virial[n2 + 2 * N], double(-r12[2] * f12[2]));
      atomicAdd(&g_virial[n2 + 3 * N], double(-r12[0] * f12[1]));
      atomicAdd(&g_virial[n2 + 4 * N], double(-r12[0] * f12[2]));
      atomicAdd(&g_virial[n2 + 5 * N], double(-r12[1] * f12[2]));
      atomicAdd(&g_virial[n2 + 6 * N], double(-r12[1] * f12[0]));
      atomicAdd(&g_virial[n2 + 7 * N], double(-r12[2] * f12[0]));
      atomicAdd(&g_virial[n2 + 8 * N], double(-r12[2] * f12[1]));
      dc6_sum += dc6 * dftd3_para.s6 * damp_6;
      dc8_sum += dc6 * c8_over_c6 * dftd3_para.s8 * damp_8;
    }
    g_potential[n1] += s_potential;
    g_dc6_sum[n1] = dc6_sum;
    g_dc8_sum[n1] = dc8_sum;
  }
}

void __global__ add_dftd3_force_extra_small_box(
  DFTD3::DFTD3_Para dftd3_para,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const float* g_dc6_sum,
  const float* g_dc8_sum,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int z1 = dftd3_para.atomic_number[g_type[n1]];
    float R_cov_1 = Bohr * covalent_radius[z1];
    float dc6_sum = g_dc6_sum[n1];
    float dc8_sum = g_dc8_sum[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int z2 = dftd3_para.atomic_number[g_type[n2]];
      float R_cov_2 = Bohr * covalent_radius[z2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      float d12 = sqrt(d12_2);
      float cn_exp_factor = exp(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f));
      float f2 = cn_exp_factor * 16.0f * (R_cov_1 + R_cov_2) * (dc6_sum + dc8_sum); // not 8.0f
      f2 /= (cn_exp_factor + 1.0f) * (cn_exp_factor + 1.0f) * d12 * d12_2;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      atomicAdd(&g_virial[n2 + 0 * N], double(-r12[0] * f12[0]));
      atomicAdd(&g_virial[n2 + 1 * N], double(-r12[1] * f12[1]));
      atomicAdd(&g_virial[n2 + 2 * N], double(-r12[2] * f12[2]));
      atomicAdd(&g_virial[n2 + 3 * N], double(-r12[0] * f12[1]));
      atomicAdd(&g_virial[n2 + 4 * N], double(-r12[0] * f12[2]));
      atomicAdd(&g_virial[n2 + 5 * N], double(-r12[1] * f12[2]));
      atomicAdd(&g_virial[n2 + 6 * N], double(-r12[1] * f12[0]));
      atomicAdd(&g_virial[n2 + 7 * N], double(-r12[2] * f12[0]));
      atomicAdd(&g_virial[n2 + 8 * N], double(-r12[2] * f12[1]));
    }
  }
}

bool get_expanded_box(const double rc, const Box& box, DFTD3::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * rc) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * rc) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * rc) {
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

    if (box.triclinic) {
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
    } else {
      ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
      ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
      ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
      ebox.h[3] = ebox.h[0] * 0.5;
      ebox.h[4] = ebox.h[1] * 0.5;
      ebox.h[5] = ebox.h[2] * 0.5;
    }
  }

  return is_small_box;
}

static __device__ void apply_mic_small_box(
  const Box& box, const DFTD3::ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
  if (box.triclinic == 0) {
    if (box.pbc_x == 1 && x12 < -ebox.h[3]) {
      x12 += ebox.h[0];
    } else if (box.pbc_x == 1 && x12 > +ebox.h[3]) {
      x12 -= ebox.h[0];
    }
    if (box.pbc_y == 1 && y12 < -ebox.h[4]) {
      y12 += ebox.h[1];
    } else if (box.pbc_y == 1 && y12 > +ebox.h[4]) {
      y12 -= ebox.h[1];
    }
    if (box.pbc_z == 1 && z12 < -ebox.h[5]) {
      z12 += ebox.h[2];
    } else if (box.pbc_z == 1 && z12 > +ebox.h[5]) {
      z12 -= ebox.h[2];
    }
  } else {
    double sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
    double sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
    double sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
    if (box.pbc_x == 1)
      sx12 -= nearbyint(sx12);
    if (box.pbc_y == 1)
      sy12 -= nearbyint(sy12);
    if (box.pbc_z == 1)
      sz12 -= nearbyint(sz12);
    x12 = ebox.h[0] * sx12 + ebox.h[1] * sy12 + ebox.h[2] * sz12;
    y12 = ebox.h[3] * sx12 + ebox.h[4] * sy12 + ebox.h[5] * sz12;
    z12 = ebox.h[6] * sx12 + ebox.h[7] * sy12 + ebox.h[8] * sz12;
  }
}

static __global__ void find_neighbor_list_small_box(
  const float rc_radial_sq,
  const float rc_angular_sq,
  const int N,
  const Box box,
  const DFTD3::ExpandedBox ebox,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = 0; n2 < N; ++n2) {
      for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
        for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
          for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }

            double delta[3];
            if (box.triclinic) {
              delta[0] = box.cpu_h[0] * ia + box.cpu_h[1] * ib + box.cpu_h[2] * ic;
              delta[1] = box.cpu_h[3] * ia + box.cpu_h[4] * ib + box.cpu_h[5] * ic;
              delta[2] = box.cpu_h[6] * ia + box.cpu_h[7] * ib + box.cpu_h[8] * ic;
            } else {
              delta[0] = box.cpu_h[0] * ia;
              delta[1] = box.cpu_h[1] * ib;
              delta[2] = box.cpu_h[2] * ic;
            }

            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(box, ebox, x12, y12, z12);

            float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
            if (distance_square < rc_radial_sq) {
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = float(x12);
              g_y12_radial[count_radial * N + n1] = float(y12);
              g_z12_radial[count_radial * N + n1] = float(z12);
              count_radial++;
            }
            if (distance_square < rc_angular_sq) {
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = float(x12);
              g_y12_angular[count_angular * N + n1] = float(y12);
              g_z12_angular[count_angular * N + n1] = float(z12);
              count_angular++;
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

std::string get_potential_file_name()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string potential_file_name;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "potential") {
        potential_file_name = tokens[1];
        break;
      }
    }
  }

  input_run.close();
  return potential_file_name;
}

void find_atomic_number(std::string& potential_file_name, int* atomic_number)
{
  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens[0].substr(0, 3) != "nep") {
    PRINT_INPUT_ERROR("DFTD3 only supports NEP models.");
  }

  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);

  if (tokens.size() != 2 + num_types) {
    std::cout << "The first line of the NEP model file should have " << num_types
              << " atom symbols." << std::endl;
    exit(1);
  }

  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number[n] = m;
        break;
      }
    }
    if (atomic_number[n] >= max_elem) {
      std::cout << "DFTD3 only supports elements from H to Pu." << std::endl;
      exit(1);
    }
  }

  input_potential.close();
}

} // namespace

void DFTD3::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = type.size();
  const int size_x12 = N * MN;
  if (NN_radial.size() == 0) {
    cn.resize(N);
    dc6_sum.resize(N);
    dc8_sum.resize(N);

    NN_radial.resize(N);
    NL_radial.resize(size_x12);
    NN_angular.resize(N);
    NL_angular.resize(size_x12);
    r12.resize(size_x12 * 6);
  }

  find_neighbor_list_small_box<<<(N - 1) / 64 + 1, 64>>>(
    rc_radial * rc_radial,
    rc_angular * rc_angular,
    N,
    box,
    ebox,
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    NN_radial.data(),
    NL_radial.data(),
    NN_angular.data(),
    NL_angular.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5);
  CUDA_CHECK_KERNEL

  find_dftd3_coordination_number_small_box<<<(N - 1) / 64 + 1, 64>>>(
    dftd3_para,
    N,
    NN_angular.data(),
    NL_angular.data(),
    type.data(),
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    cn.data());
  CUDA_CHECK_KERNEL

  add_dftd3_force_small_box<<<(N - 1) / 64 + 1, 64>>>(
    dftd3_para,
    N,
    NN_radial.data(),
    NL_radial.data(),
    type.data(),
    c6_ref.data(),
    cn.data(),
    r12.data(),
    r12.data() + size_x12,
    r12.data() + size_x12 * 2,
    potential_per_atom.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data(),
    dc6_sum.data(),
    dc8_sum.data());
  CUDA_CHECK_KERNEL

  add_dftd3_force_extra_small_box<<<(N - 1) / 64 + 1, 64>>>(
    dftd3_para,
    N,
    NN_angular.data(),
    NL_angular.data(),
    type.data(),
    dc6_sum.data(),
    dc8_sum.data(),
    r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4,
    r12.data() + size_x12 * 5,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  CUDA_CHECK_KERNEL
}

void DFTD3::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(rc_radial, box, ebox);

  if (is_small_box) {
    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  } else {
    std::cout << "large box version of DFTD3 has not been implemented yet.\n";
    exit(1);
  }
}

void DFTD3::initialize(
  std::string& xc_functional, const float rc_potential, const float rc_coordination_number)
{
  rc_radial = rc_potential;
  rc_angular = rc_coordination_number;
  if (xc_functional == "pbe" || xc_functional == "PBE") {
    dftd3_para.s6 = 1.0f;
    dftd3_para.s8 = 0.78750f;
    dftd3_para.a1 = 0.42890f;
    dftd3_para.a2 = 4.4407f * Bohr;
  } else {
    std::cout << "We only support the PBE functional for the time being.\n" << std::endl;
    exit(1);
  }

  c6_ref.resize(111625);
  c6_ref.copy_from_host(c6_ref_cpu);

  std::string potential_file_name = get_potential_file_name();
  find_atomic_number(potential_file_name, dftd3_para.atomic_number);
}