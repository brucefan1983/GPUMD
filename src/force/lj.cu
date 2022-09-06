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
The class dealing with the Lennard-Jones (LJ) pairwise potentials.
------------------------------------------------------------------------------*/

#include "lj.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"

// best block size here: 128
#define BLOCK_SIZE_FORCE 128

LJ::LJ(FILE* fid, int num_types, int num_atoms)
{
  printf("Use %d-element LJ potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE)) {
    PRINT_INPUT_ERROR("Incorrect number of LJ parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for LJ potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  double epsilon, sigma, cutoff;
  rc = 0.0;
  for (int n = 0; n < num_types; n++) {
    for (int m = 0; m < num_types; m++) {
      int count = fscanf(fid, "%lf%lf%lf", &epsilon, &sigma, &cutoff);
      PRINT_SCANF_ERROR(count, 3, "Reading error for LJ potential.");

      lj_para.s6e4[n][m] = pow(sigma, 6.0) * epsilon * 4.0;
      lj_para.s12e4[n][m] = pow(sigma, 12.0) * epsilon * 4.0;
      lj_para.cutoff_square[n][m] = cutoff * cutoff;
      if (rc < cutoff)
        rc = cutoff;
    }
  }

  lj_data.NN.resize(num_atoms);
  lj_data.NL.resize(num_atoms * 1024); // the largest supported by CUDA
  lj_data.cell_count.resize(num_atoms);
  lj_data.cell_count_sum.resize(num_atoms);
  lj_data.cell_contents.resize(num_atoms);
}

LJ::~LJ(void)
{
  // nothing
}

// get U_ij and (d U_ij / d r_ij) / r_ij (the LJ potential)
static __device__ void find_p2_and_f2(float s6e4, float s12e4, float d12sq, float& p2, float& f2)
{
  float d12inv2 = 1.0f / d12sq;
  float d12inv6 = d12inv2 * d12inv2 * d12inv2;
  f2 = 6.0f * (s6e4 * d12inv6 - s12e4 * 2.0f * d12inv6 * d12inv6) * d12inv2;
  p2 = s12e4 * d12inv6 * d12inv6 - s6e4 * d12inv6;
}

// force evaluation kernel
static __global__ void gpu_find_force(
  LJ_Para lj,
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_potential)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  float s_fx = 0.0f;                                   // force_x
  float s_fy = 0.0f;                                   // force_y
  float s_fz = 0.0f;                                   // force_z
  float s_pe = 0.0f;                                   // potential energy
  float s_sxx = 0.0f;                                  // virial_stress_xx
  float s_sxy = 0.0f;                                  // virial_stress_xy
  float s_sxz = 0.0f;                                  // virial_stress_xz
  float s_syx = 0.0f;                                  // virial_stress_yx
  float s_syy = 0.0f;                                  // virial_stress_yy
  float s_syz = 0.0f;                                  // virial_stress_yz
  float s_szx = 0.0f;                                  // virial_stress_zx
  float s_szy = 0.0f;                                  // virial_stress_zy
  float s_szz = 0.0f;                                  // virial_stress_zz

  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      int type2 = g_type[n2];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12sq = x12 * x12 + y12 * y12 + z12 * z12;

      float p2, f2;
      if (d12sq >= lj.cutoff_square[type1][type2]) {
        continue;
      }
      find_p2_and_f2(lj.s6e4[type1][type2], lj.s12e4[type1][type2], d12sq, p2, f2);

      // treat two-body potential in the same way as many-body potential
      float f12x = f2 * x12 * 0.5f;
      float f12y = f2 * y12 * 0.5f;
      float f12z = f2 * z12 * 0.5f;
      float f21x = -f12x;
      float f21y = -f12y;
      float f21z = -f12z;

      // accumulate force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // accumulate potential energy and virial
      s_pe += p2 * 0.5f; // two-body potential
      s_sxx += x12 * f21x;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syy += y12 * f21y;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
      s_szz += z12 * f21z;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;

    // save potential
    g_potential[n1] += s_pe;
  }
}

// Find force and related quantities for pair potentials (A wrapper)
void LJ::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor(
      N1, N2, rc, box, type, position_per_atom, lj_data.cell_count, lj_data.cell_count_sum,
      lj_data.cell_contents, lj_data.NN,
      lj_data.NL); // TODO: generalize
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  gpu_find_force<<<grid_size, BLOCK_SIZE_FORCE>>>(
    lj_para, number_of_atoms, N1, N2, box, lj_data.NN.data(), lj_data.NL.data(), type.data(),
    position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, force_per_atom.data(),
    force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data(), potential_per_atom.data());
  CUDA_CHECK_KERNEL
}
