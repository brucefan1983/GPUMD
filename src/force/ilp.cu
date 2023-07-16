
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

#include "ilp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"

// TODO: best size here: 128
#define BLOCK_SIZE_FORCE 128

ILP::ILP(FILE* fid, int num_types, int num_atoms)
{
  printf("Use %d-element ILP potential with elements:\n", num_types);
  if (!(num_types >= 1 && num_types <= MAX_TYPE_ILP)) {
    PRINT_INPUT_ERROR("Incorrect number of ILP parameters.\n");
  }
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for ILP potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  // read parameters
  double beta, alpha, delta, epsilon, C, d, sR, reff, C6, S, rcut;
  rc = 0.0;
  for (int n = 0; n < num_types; ++n) {
    for (int m = 0; m < num_types; ++m) {
      int count = fscanf(fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", \
      &beta, &alpha, &delta, &epsilon, &C, &d, &sR, &reff, &C6, &S, &rcut);
      PRINT_SCANF_ERROR(count, 10, "Reading error for ILP potential.");

      ilp_para.C[n][m] = C;
      ilp_para.C_6[n][m] = C6;
      ilp_para.d[n][m] = d;
      ilp_para.d_Seff[n][m] = d / sR / reff;
      ilp_para.epsilon[n][m] = epsilon;
      ilp_para.z0[n][m] = beta;
      ilp_para.lambda[n][m] = alpha / beta;
      ilp_para.delta2inv[n][m] = 1.0 / (delta * delta); //TODO: how faster?
      ilp_para.S[n][m] = S;
      ilp_para.r_cut[n][m] = rcut;

      // TODO: ILP has taper function, check if necessary
      if (rc < rcut)
        rc = rcut;
    }
  }

  ilp_data.NN.resize(num_atoms);
  ilp_data.NL.resize(num_atoms * CUDA_MAX_NL);
  ilp_data.cell_count.resize(num_atoms);
  ilp_data.cell_count_sum.resize(num_atoms);
  ilp_data.cell_contents.resize(num_atoms);
}

ILP::~ILP(void)
{
  // TODO
}

// calculate the normals and its derivatives
static __device__ void calc_normal(void)
{
  // TODO
}

// calculate the van der Waals force and energy
static __device__ void calc_vdW(void)
{
  // TODO
}

// calculate the repulsive force and energy
static __device__ void calc_rep(void)
{
  // TODO
}

// force evaluation kernel
static __global__ void gpu_find_force(
  ILP_Para ilp)
{
  // TODO
}

// find force and related quantities
void ILP::compute(
  Box &box,
  const GPU_Vector<int> &type,
  const GPU_Vector<double> &position_per_atom,
  GPU_Vector<double> &potential_per_atom,
  GPU_Vector<double> &force_per_atom,
  GPU_Vector<double> &virial_per_atom)
{
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

// what's this??
#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor(
      N1,
      N2,
      rc,
      box,
      type,
      position_per_atom,
      ilp_data.cell_count,
      ilp_data.cell_count_sum,
      ilp_data.cell_contents,
      ilp_data.NN,
      ilp_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  gpu_find_force<<<grid_size, BLOCK_SIZE_FORCE>>>(ilp_para);
  // TODO
  CUDA_CHECK_KERNEL
}
