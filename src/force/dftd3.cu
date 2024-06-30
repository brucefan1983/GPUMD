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
#include "neighbor.cuh"
#include "utilities/common.cuh"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

namespace
{
const int MN = 10000; // maximum number of neighbors for one atom
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu"};

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
            W += L_ij;
            dW += L_ij * (-8.0f * diff_i);
            float c6_ref_ij = (z1 < z2) ? g_c6_ref[z12 * max_cn2 + i * max_cn + j]
                                        : g_c6_ref[z12 * max_cn2 + j * max_cn + i];
            Z += c6_ref_ij * L_ij;
            dZ += c6_ref_ij * L_ij * (-8.0f * diff_i);
          }
        }

        if (W < 1.0e-30f) {
          int i = num_cn_1 - 1;
          int j = num_cn_2 - 1;
          c6 = (z1 < z2) ? g_c6_ref[z12 * max_cn2 + i * max_cn + j]
                         : g_c6_ref[z12 * max_cn2 + j * max_cn + i];
        } else {
          W = 1.0f / W;
          c6 = Z * W;
          dc6 = dZ * W - c6 * dW * W;
        }
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

__device__ int find_neighbor_cell(
  int cell_id,
  int cell_id_x,
  int cell_id_y,
  int cell_id_z,
  int nx,
  int ny,
  int nz,
  int xx,
  int yy,
  int zz)
{
  int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
  if (cell_id_x + xx < 0)
    neighbor_cell += nx;
  if (cell_id_x + xx >= nx)
    neighbor_cell -= nx;
  if (cell_id_y + yy < 0)
    neighbor_cell += ny * nx;
  if (cell_id_y + yy >= ny)
    neighbor_cell -= ny * nx;
  if (cell_id_z + zz < 0)
    neighbor_cell += nz * ny * nx;
  if (cell_id_z + zz >= nz)
    neighbor_cell -= nz * ny * nx;

  return neighbor_cell;
}

__global__ void find_dftd3_coordination_number_large_box(
  const DFTD3::DFTD3_Para dftd3_para,
  const float rc,
  const int N,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* g_cell_count,
  const int* g_cell_count_sum,
  const int* g_cell_contents,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  float* g_cn)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  int atomic_number_1 = dftd3_para.atomic_number[g_type[n1]];
  float R_cov_1 = Bohr * covalent_radius[atomic_number_1];
  float cn_temp = 0.0f;
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(box, x1, y1, z1, 2.0f / rc, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;
  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell =
          find_neighbor_cell(cell_id, cell_id_x, cell_id_y, cell_id_z, nx, ny, nz, xx, yy, zz);
        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];
          if (n1 == n2) {
            continue;
          }
          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float r12[3] = {float(x12double), float(y12double), float(z12double)};
          float d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          if (d12_2 < rc * rc) {
            int atomic_number_2 = dftd3_para.atomic_number[g_type[n2]];
            float R_cov_2 = Bohr * covalent_radius[atomic_number_2];
            float d12 = sqrt(d12_2);
            cn_temp += 1.0f / (exp(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f)) + 1.0f);
          }
        }
      }
    }
  }
  g_cn[n1] = cn_temp;
}

__global__ void find_dftd3_force_large_box(
  const DFTD3::DFTD3_Para dftd3_para,
  const float rc,
  const int N,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* g_cell_count,
  const int* g_cell_count_sum,
  const int* g_cell_contents,
  const float* g_c6_ref,
  const float* g_cn,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  double* g_potential,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  float* g_dc6_sum,
  float* g_dc8_sum)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  int atomic_number_1 = dftd3_para.atomic_number[g_type[n1]];
  int num_cn_1 = num_cn[atomic_number_1];
  float dc6_sum = 0.0f;
  float dc8_sum = 0.0f;
  float s_pe = 0.0f;
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(box, x1, y1, z1, 2.0f / rc, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;
  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell =
          find_neighbor_cell(cell_id, cell_id_x, cell_id_y, cell_id_z, nx, ny, nz, xx, yy, zz);
        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];
          if (n1 == n2) {
            continue;
          }
          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float r12[3] = {float(x12double), float(y12double), float(z12double)};
          float d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          if (d12_2 < rc * rc) {
            int atomic_number_2 = dftd3_para.atomic_number[g_type[n2]];
            int z_small = atomic_number_1, z_large = atomic_number_2;
            if (atomic_number_1 > atomic_number_2) {
              z_small = atomic_number_2;
              z_large = atomic_number_1;
            }
            int z12 = z_small * max_elem - (z_small * (z_small - 1)) / 2 + (z_large - z_small);
            float d12_4 = d12_2 * d12_2;
            float d12_6 = d12_4 * d12_2;
            float d12_8 = d12_6 * d12_2;
            float c6 = 0.0f;
            float dc6 = 0.0f;
            int num_cn_2 = num_cn[atomic_number_2];
            if (num_cn_1 == 1 && num_cn_2 == 1) {
              c6 = g_c6_ref[z12 * max_cn2];
            } else {
              float W = 0.0f;
              float dW = 0.0f;
              float Z = 0.0f;
              float dZ = 0.0f;
              for (int i = 0; i < num_cn_1; ++i) {
                for (int j = 0; j < num_cn_2; ++j) {
                  float diff_i = g_cn[n1] - cn_ref[atomic_number_1 * max_cn + i];
                  float diff_j = g_cn[n2] - cn_ref[atomic_number_2 * max_cn + j];
                  float L_ij = exp(-4.0f * (diff_i * diff_i + diff_j * diff_j));
                  W += L_ij;
                  dW += L_ij * (-8.0f * diff_i);
                  float c6_ref_ij = (atomic_number_1 < atomic_number_2)
                                      ? g_c6_ref[z12 * max_cn2 + i * max_cn + j]
                                      : g_c6_ref[z12 * max_cn2 + j * max_cn + i];
                  Z += c6_ref_ij * L_ij;
                  dZ += c6_ref_ij * L_ij * (-8.0f * diff_i);
                }
              }
              if (W < 1.0e-30f) {
                int i = num_cn_1 - 1;
                int j = num_cn_2 - 1;
                c6 = (atomic_number_1 < atomic_number_2) ? g_c6_ref[z12 * max_cn2 + i * max_cn + j]
                                                         : g_c6_ref[z12 * max_cn2 + j * max_cn + i];
              } else {
                W = 1.0f / W;
                c6 = Z * W;
                dc6 = dZ * W - c6 * dW * W;
              }
            }
            c6 *= HartreeBohr6;
            dc6 *= HartreeBohr6;

            float c8_over_c6 = 3.0f * r2r4[atomic_number_1] * r2r4[atomic_number_2] * Bohr2;
            float c8 = c6 * c8_over_c6;
            float damp = dftd3_para.a1 * sqrt(c8_over_c6) + dftd3_para.a2;
            float damp_2 = damp * damp;
            float damp_4 = damp_2 * damp_2;
            float damp_6 = 1.0f / (d12_6 + damp_4 * damp_2);
            float damp_8 = 1.0f / (d12_8 + damp_4 * damp_4);
            s_pe -= (dftd3_para.s6 * c6 * damp_6 + dftd3_para.s8 * c8 * damp_8) * 0.5f;
            float f2 = dftd3_para.s6 * c6 * 3.0f * d12_4 * (damp_6 * damp_6) +
                       dftd3_para.s8 * c8 * 4.0f * d12_6 * (damp_8 * damp_8);
            float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
            s_fx += 2.0f * f12[0];
            s_fy += 2.0f * f12[1];
            s_fz += 2.0f * f12[2];
            s_sxx -= r12[0] * f12[0];
            s_sxy -= r12[0] * f12[1];
            s_sxz -= r12[0] * f12[2];
            s_syx -= r12[1] * f12[0];
            s_syy -= r12[1] * f12[1];
            s_syz -= r12[1] * f12[2];
            s_szx -= r12[2] * f12[0];
            s_szy -= r12[2] * f12[1];
            s_szz -= r12[2] * f12[2];
            dc6_sum += dc6 * dftd3_para.s6 * damp_6;
            dc8_sum += dc6 * c8_over_c6 * dftd3_para.s8 * damp_8;
          }
        }
      }
    }
  }
  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;
  g_virial[n1 + 0 * N] += s_sxx;
  g_virial[n1 + 1 * N] += s_syy;
  g_virial[n1 + 2 * N] += s_szz;
  g_virial[n1 + 3 * N] += s_sxy;
  g_virial[n1 + 4 * N] += s_sxz;
  g_virial[n1 + 5 * N] += s_syz;
  g_virial[n1 + 6 * N] += s_syx;
  g_virial[n1 + 7 * N] += s_szx;
  g_virial[n1 + 8 * N] += s_szy;
  g_potential[n1] += s_pe;
  g_dc6_sum[n1] = dc6_sum;
  g_dc8_sum[n1] = dc8_sum;
}

__global__ void find_dftd3_force_extra_large_box(
  const DFTD3::DFTD3_Para dftd3_para,
  const float rc,
  const int N,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* g_cell_count,
  const int* g_cell_count_sum,
  const int* g_cell_contents,
  const float* g_dc6_sum,
  const float* g_dc8_sum,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  int atomic_number_1 = dftd3_para.atomic_number[g_type[n1]];
  float R_cov_1 = Bohr * covalent_radius[atomic_number_1];
  float dc6_sum_plus_dc8_sum_1 = g_dc6_sum[n1] + g_dc8_sum[n1];
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(box, x1, y1, z1, 2.0f / rc, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;
  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell =
          find_neighbor_cell(cell_id, cell_id_x, cell_id_y, cell_id_z, nx, ny, nz, xx, yy, zz);
        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];
          if (n1 == n2) {
            continue;
          }
          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float r12[3] = {float(x12double), float(y12double), float(z12double)};
          float d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          if (d12_2 < rc * rc) {
            int atomic_number_2 = dftd3_para.atomic_number[g_type[n2]];
            float R_cov_2 = Bohr * covalent_radius[atomic_number_2];
            float d12 = sqrt(d12_2);
            float cn_exp_factor = exp(-16.0f * ((R_cov_1 + R_cov_2) / d12 - 1.0f));
            float f12_factor = cn_exp_factor * 16.0f * (R_cov_1 + R_cov_2);
            f12_factor /= (cn_exp_factor + 1.0f) * (cn_exp_factor + 1.0f) * d12 * d12_2;
            float f21_factor = f12_factor;
            f12_factor *= dc6_sum_plus_dc8_sum_1;
            f21_factor *= -(g_dc6_sum[n2] + g_dc8_sum[n2]);
            float f12[3] = {r12[0] * f12_factor, r12[1] * f12_factor, r12[2] * f12_factor};
            float f21[3] = {r12[0] * f21_factor, r12[1] * f21_factor, r12[2] * f21_factor};
            s_fx += f12[0] - f21[0];
            s_fy += f12[1] - f21[1];
            s_fz += f12[2] - f21[2];
            s_sxx += r12[0] * f21[0];
            s_sxy += r12[0] * f21[1];
            s_sxz += r12[0] * f21[2];
            s_syx += r12[1] * f21[0];
            s_syy += r12[1] * f21[1];
            s_syz += r12[1] * f21[2];
            s_szx += r12[2] * f21[0];
            s_szy += r12[2] * f21[1];
            s_szz += r12[2] * f21[2];
          }
        }
      }
    }
  }
  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;
  g_virial[n1 + 0 * N] += s_sxx;
  g_virial[n1 + 1 * N] += s_syy;
  g_virial[n1 + 2 * N] += s_szz;
  g_virial[n1 + 3 * N] += s_sxy;
  g_virial[n1 + 4 * N] += s_sxz;
  g_virial[n1 + 5 * N] += s_syz;
  g_virial[n1 + 6 * N] += s_syx;
  g_virial[n1 + 7 * N] += s_szx;
  g_virial[n1 + 8 * N] += s_szy;
}

bool set_para(
  const std::string& functional_input,
  const std::string& functional_library,
  const float s6,
  const float a1,
  const float s8,
  const float a2,
  DFTD3::DFTD3_Para& dftd3_para)
{
  if (functional_input == functional_library) {
    dftd3_para.s6 = s6;
    dftd3_para.a1 = a1;
    dftd3_para.s8 = s8;
    dftd3_para.a2 = a2 * Bohr;
    return true;
  }
  return false;
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

void DFTD3::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int N = type.size();
  if (cn.size() == 0) {
    cn.resize(N);
    dc6_sum.resize(N);
    dc8_sum.resize(N);
    cell_count_radial.resize(N);
    cell_count_sum_radial.resize(N);
    cell_contents_radial.resize(N);
    cell_count_angular.resize(N);
    cell_count_sum_angular.resize(N);
    cell_contents_angular.resize(N);
  }

  int num_bins_radial[3];
  int num_bins_angular[3];
  box.get_num_bins(0.5 * rc_radial, num_bins_radial);
  box.get_num_bins(0.5 * rc_angular, num_bins_angular);

  find_cell_list(
    0.5 * rc_radial,
    num_bins_radial,
    box,
    position_per_atom,
    cell_count_radial,
    cell_count_sum_radial,
    cell_contents_radial);
  find_cell_list(
    0.5 * rc_angular,
    num_bins_angular,
    box,
    position_per_atom,
    cell_count_angular,
    cell_count_sum_angular,
    cell_contents_angular);

  find_dftd3_coordination_number_large_box<<<(N - 1) / 64 + 1, 64>>>(
    dftd3_para,
    rc_angular,
    N,
    num_bins_angular[0],
    num_bins_angular[1],
    num_bins_angular[2],
    box,
    type.data(),
    cell_count_angular.data(),
    cell_count_sum_angular.data(),
    cell_contents_angular.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    cn.data());
  CUDA_CHECK_KERNEL

  find_dftd3_force_large_box<<<(N - 1) / 64 + 1, 64>>>(
    dftd3_para,
    rc_radial,
    N,
    num_bins_radial[0],
    num_bins_radial[1],
    num_bins_radial[2],
    box,
    type.data(),
    cell_count_radial.data(),
    cell_count_sum_radial.data(),
    cell_contents_radial.data(),
    c6_ref.data(),
    cn.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data(),
    dc6_sum.data(),
    dc8_sum.data());
  CUDA_CHECK_KERNEL

  find_dftd3_force_extra_large_box<<<(N - 1) / 64 + 1, 64>>>(
    dftd3_para,
    rc_angular,
    N,
    num_bins_angular[0],
    num_bins_angular[1],
    num_bins_angular[2],
    box,
    type.data(),
    cell_count_angular.data(),
    cell_count_sum_angular.data(),
    cell_contents_angular.data(),
    dc6_sum.data(),
    dc8_sum.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
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
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

void DFTD3::initialize(
  std::string& functional, const float rc_potential, const float rc_coordination_number)
{
  rc_radial = rc_potential;
  rc_angular = rc_coordination_number;

  std::transform(functional.begin(), functional.end(), functional.begin(), [](unsigned char c) {
    return std::tolower(c);
  });

  bool valid = false;
  valid = valid || set_para(functional, "b1b95", 1.000, 0.2092, 1.4507, 5.5545, dftd3_para);
  valid = valid || set_para(functional, "b2gpplyp", 0.560, 0.0000, 0.2597, 6.3332, dftd3_para);
  valid = valid || set_para(functional, "b2plyp", 0.640, 0.3065, 0.9147, 5.0570, dftd3_para);
  valid = valid || set_para(functional, "b3lyp", 1.000, 0.3981, 1.9889, 4.4211, dftd3_para);
  valid = valid || set_para(functional, "b3pw91", 1.000, 0.4312, 2.8524, 4.4693, dftd3_para);
  valid = valid || set_para(functional, "b97d", 1.000, 0.5545, 2.2609, 3.2297, dftd3_para);
  valid = valid || set_para(functional, "bhlyp", 1.000, 0.2793, 1.0354, 4.9615, dftd3_para);
  valid = valid || set_para(functional, "blyp", 1.000, 0.4298, 2.6996, 4.2359, dftd3_para);
  valid = valid || set_para(functional, "bmk", 1.000, 0.1940, 2.0860, 5.9197, dftd3_para);
  valid = valid || set_para(functional, "bop", 1.000, 0.4870, 3.295, 3.5043, dftd3_para);
  valid = valid || set_para(functional, "bp86", 1.000, 0.3946, 3.2822, 4.8516, dftd3_para);
  valid = valid || set_para(functional, "bpbe", 1.000, 0.4567, 4.0728, 4.3908, dftd3_para);
  valid = valid || set_para(functional, "camb3lyp", 1.000, 0.3708, 2.0674, 5.4743, dftd3_para);
  valid = valid || set_para(functional, "dsdblyp", 0.500, 0.0000, 0.2130, 6.0519, dftd3_para);
  valid = valid || set_para(functional, "hcth120", 1.000, 0.3563, 1.0821, 4.3359, dftd3_para);
  valid = valid || set_para(functional, "hf", 1.000, 0.3385, 0.9171, 2.883, dftd3_para);
  valid = valid || set_para(functional, "hse-hjs", 1.000, 0.3830, 2.3100, 5.685, dftd3_para);
  valid = valid || set_para(functional, "lc-wpbe08", 1.000, 0.3919, 1.8541, 5.0897, dftd3_para);
  valid = valid || set_para(functional, "lcwpbe", 1.000, 0.3919, 1.8541, 5.0897, dftd3_para);
  valid = valid || set_para(functional, "m11", 1.000, 0.0000, 2.8112, 10.1389, dftd3_para);
  valid = valid || set_para(functional, "mn12l", 1.000, 0.0000, 2.2674, 9.1494, dftd3_para);
  valid = valid || set_para(functional, "mn12sx", 1.000, 0.0983, 1.1674, 8.0259, dftd3_para);
  valid = valid || set_para(functional, "mpw1b95", 1.000, 0.1955, 1.0508, 6.4177, dftd3_para);
  valid = valid || set_para(functional, "mpwb1k", 1.000, 0.1474, 0.9499, 6.6223, dftd3_para);
  valid = valid || set_para(functional, "mpwlyp", 1.000, 0.4831, 2.0077, 4.5323, dftd3_para);
  valid = valid || set_para(functional, "n12sx", 1.000, 0.3283, 2.4900, 5.7898, dftd3_para);
  valid = valid || set_para(functional, "olyp", 1.000, 0.5299, 2.6205, 2.8065, dftd3_para);
  valid = valid || set_para(functional, "opbe", 1.000, 0.5512, 3.3816, 2.9444, dftd3_para);
  valid = valid || set_para(functional, "otpss", 1.000, 0.4634, 2.7495, 4.3153, dftd3_para);
  valid = valid || set_para(functional, "pbe", 1.000, 0.4289, 0.7875, 4.4407, dftd3_para);
  valid = valid || set_para(functional, "pbe0", 1.000, 0.4145, 1.2177, 4.8593, dftd3_para);
  valid = valid || set_para(functional, "pbe38", 1.000, 0.3995, 1.4623, 5.1405, dftd3_para);
  valid = valid || set_para(functional, "pbesol", 1.000, 0.4466, 2.9491, 6.1742, dftd3_para);
  valid = valid || set_para(functional, "ptpss", 0.750, 0.000, 0.2804, 6.5745, dftd3_para);
  valid = valid || set_para(functional, "pw6b95", 1.000, 0.2076, 0.7257, 6.375, dftd3_para);
  valid = valid || set_para(functional, "pwb6k", 1.000, 0.1805, 0.9383, 7.7627, dftd3_para);
  valid = valid || set_para(functional, "pwpb95", 0.820, 0.0000, 0.2904, 7.3141, dftd3_para);
  valid = valid || set_para(functional, "revpbe", 1.000, 0.5238, 2.3550, 3.5016, dftd3_para);
  valid = valid || set_para(functional, "revpbe0", 1.000, 0.4679, 1.7588, 3.7619, dftd3_para);
  valid = valid || set_para(functional, "revpbe38", 1.000, 0.4309, 1.4760, 3.9446, dftd3_para);
  valid = valid || set_para(functional, "revssb", 1.000, 0.4720, 0.4389, 4.0986, dftd3_para);
  valid = valid || set_para(functional, "rpbe", 1.000, 0.1820, 0.8318, 4.0094, dftd3_para);
  valid = valid || set_para(functional, "rpw86pbe", 1.000, 0.4613, 1.3845, 4.5062, dftd3_para);
  valid = valid || set_para(functional, "scan", 1.000, 0.5380, 0.0000, 5.42, dftd3_para);
  valid = valid || set_para(functional, "sogga11x", 1.000, 0.1330, 1.1426, 5.7381, dftd3_para);
  valid = valid || set_para(functional, "ssb", 1.000, -0.0952, -0.1744, 5.2170, dftd3_para);
  valid = valid || set_para(functional, "tpss", 1.000, 0.4535, 1.9435, 4.4752, dftd3_para);
  valid = valid || set_para(functional, "tpss0", 1.000, 0.3768, 1.2576, 4.5865, dftd3_para);
  valid = valid || set_para(functional, "tpssh", 1.000, 0.4529, 2.2382, 4.6550, dftd3_para);
  valid = valid || set_para(functional, "b2kplyp", 0.64, 0.0000, 0.1521, 7.1916, dftd3_para);
  valid = valid || set_para(functional, "dsd-pbep86", 0.418, 0.0000, 0.0000, 5.6500, dftd3_para);

  if (!valid) {
    std::cout << "The " << functional
              << " functional is not supported for DFT-D3 with BJ damping.\n"
              << std::endl;
    exit(1);
  }

  std::cout << "    Add DFT-D3:" << std::endl;
  std::cout << "        potential cutoff = " << rc_potential << " Angstrom" << std::endl;
  std::cout << "        coordination number cutoff = " << rc_coordination_number << " Angstrom"
            << std::endl;
  std::cout << "        functional = " << functional << std::endl;
  std::cout << "            s6 = " << dftd3_para.s6 << std::endl;
  std::cout << "            s8 = " << dftd3_para.s8 << std::endl;
  std::cout << "            a1 = " << dftd3_para.a1 << std::endl;
  std::cout << "            a2 = " << dftd3_para.a2 << " Angstrom" << std::endl;

  c6_ref.resize(111625);
  c6_ref.copy_from_host(c6_ref_cpu);

  std::string potential_file_name = get_potential_file_name();
  find_atomic_number(potential_file_name, dftd3_para.atomic_number);
}