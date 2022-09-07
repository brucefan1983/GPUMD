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
The double-element version of the Tersoff potential as described in
    [1] J. Tersoff, Modeling solid-state chemistry: Interatomic potentials
        for multicomponent systems, PRB 39, 5566 (1989).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "tersoff1989.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

#define BLOCK_SIZE_FORCE 64 // 128 is also good

Tersoff1989::Tersoff1989(FILE* fid, int num_of_types, const int num_atoms)
{
  printf("Use Tersoff-1989 (%d-element) potential with element(s):", num_of_types);

  for (int n = 0; n < num_of_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for Tersoff-1989 potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  // first line
  int count;
  double a, b, lambda, mu, beta, n, c, d, h, r1, r2;
  count = fscanf(
    fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &a, &b, &lambda, &mu, &beta, &n, &c, &d, &h, &r1,
    &r2);
  PRINT_SCANF_ERROR(count, 11, "Reading error for Tersoff-1989 potential.");

  ters0.a = a;
  ters0.b = b;
  ters0.lambda = lambda;
  ters0.mu = mu;
  ters0.beta = beta;
  ters0.n = n;
  ters0.c = c;
  ters0.d = d;
  ters0.h = h;
  ters0.r1 = r1;
  ters0.r2 = r2;
  ters0.c2 = c * c;
  ters0.d2 = d * d;
  ters0.one_plus_c2overd2 = 1.0 + ters0.c2 / ters0.d2;
  ters0.pi_factor = PI / (r2 - r1);
  ters0.minus_half_over_n = -0.5 / n;
  rc = ters0.r2;

  if (num_of_types == 2) {
    // second line
    count = fscanf(
      fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &a, &b, &lambda, &mu, &beta, &n, &c, &d, &h, &r1,
      &r2);
    PRINT_SCANF_ERROR(count, 11, "Reading error for Tersoff-1989 potential.");

    ters1.a = a;
    ters1.b = b;
    ters1.lambda = lambda;
    ters1.mu = mu;
    ters1.beta = beta;
    ters1.n = n;
    ters1.c = c;
    ters1.d = d;
    ters1.h = h;
    ters1.r1 = r1;
    ters1.r2 = r2;
    ters1.c2 = c * c;
    ters1.d2 = d * d;
    ters1.one_plus_c2overd2 = 1.0 + ters1.c2 / ters1.d2;
    ters1.pi_factor = PI / (r2 - r1);
    ters1.minus_half_over_n = -0.5 / n;

    // third line
    double chi;
    count = fscanf(fid, "%lf", &chi);
    PRINT_SCANF_ERROR(count, 1, "Reading error for Tersoff-1989 potential.");

    // mixing type 0 and type 1
    ters2.a = sqrt(ters0.a * ters1.a);
    ters2.b = sqrt(ters0.b * ters1.b);
    ters2.b *= chi;
    ters2.lambda = 0.5 * (ters0.lambda + ters1.lambda);
    ters2.mu = 0.5 * (ters0.mu + ters1.mu);
    ters2.beta = 0.0; // not used
    ters2.n = 0.0;    // not used
    ters2.c2 = 0.0;   // not used
    ters2.d2 = 0.0;   // not used
    ters2.h = 0.0;    // not used
    ters2.r1 = sqrt(ters0.r1 * ters1.r1);
    ters2.r2 = sqrt(ters0.r2 * ters1.r2);
    ters2.one_plus_c2overd2 = 0.0; // not used
    ters2.pi_factor = PI / (ters2.r2 - ters2.r1);
    ters2.minus_half_over_n = 0.0; // not used

    // force cutoff
    rc = (ters0.r2 > ters1.r2) ? ters0.r2 : ters1.r2;
  }

  const int num_of_neighbors = 50 * num_atoms;
  tersoff_data.b.resize(num_of_neighbors);
  tersoff_data.bp.resize(num_of_neighbors);
  tersoff_data.f12x.resize(num_of_neighbors);
  tersoff_data.f12y.resize(num_of_neighbors);
  tersoff_data.f12z.resize(num_of_neighbors);
  tersoff_data.NN.resize(num_atoms);
  tersoff_data.NL.resize(num_of_neighbors);
  tersoff_data.cell_count.resize(num_atoms);
  tersoff_data.cell_count_sum.resize(num_atoms);
  tersoff_data.cell_contents.resize(num_atoms);
}

Tersoff1989::~Tersoff1989(void)
{
  // nothing
}

static __device__ void find_fr_and_frp(
  int type1,
  int type2,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  const Tersoff1989_Parameters& ters2,
  double d12,
  double& fr,
  double& frp)
{
  if (type1 == 0 && type2 == 0) {
    fr = ters0.a * exp(-ters0.lambda * d12);
    frp = -ters0.lambda * fr;
  } else if (type1 == 1 && type2 == 1) {
    fr = ters1.a * exp(-ters1.lambda * d12);
    frp = -ters1.lambda * fr;
  } else {
    fr = ters2.a * exp(-ters2.lambda * d12);
    frp = -ters2.lambda * fr;
  }
}

static __device__ void find_fa_and_fap(
  int type1,
  int type2,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  const Tersoff1989_Parameters& ters2,
  double d12,
  double& fa,
  double& fap)
{
  if (type1 == 0 && type2 == 0) {
    fa = ters0.b * exp(-ters0.mu * d12);
    fap = -ters0.mu * fa;
  } else if (type1 == 1 && type2 == 1) {
    fa = ters1.b * exp(-ters1.mu * d12);
    fap = -ters1.mu * fa;
  } else {
    fa = ters2.b * exp(-ters2.mu * d12);
    fap = -ters2.mu * fa;
  }
}

static __device__ void find_fa(
  int type1,
  int type2,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  const Tersoff1989_Parameters& ters2,
  double d12,
  double& fa)
{
  if (type1 == 0 && type2 == 0) {
    fa = ters0.b * exp(-ters0.mu * d12);
  } else if (type1 == 1 && type2 == 1) {
    fa = ters1.b * exp(-ters1.mu * d12);
  } else {
    fa = ters2.b * exp(-ters2.mu * d12);
  }
}

static __device__ void find_fc_and_fcp(
  int type1,
  int type2,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  const Tersoff1989_Parameters& ters2,
  double d12,
  double& fc,
  double& fcp)
{
  if (type1 == 0 && type2 == 0) {
    if (d12 < ters0.r1) {
      fc = 1.0;
      fcp = 0.0;
    } else if (d12 < ters0.r2) {
      fc = cos(ters0.pi_factor * (d12 - ters0.r1)) * 0.5 + 0.5;
      fcp = -sin(ters0.pi_factor * (d12 - ters0.r1)) * ters0.pi_factor * 0.5;
    } else {
      fc = 0.0;
      fcp = 0.0;
    }
  } else if (type1 == 1 && type2 == 1) {
    if (d12 < ters1.r1) {
      fc = 1.0;
      fcp = 0.0;
    } else if (d12 < ters1.r2) {
      fc = cos(ters1.pi_factor * (d12 - ters1.r1)) * 0.5 + 0.5;
      fcp = -sin(ters1.pi_factor * (d12 - ters1.r1)) * ters1.pi_factor * 0.5;
    } else {
      fc = 0.0;
      fcp = 0.0;
    }
  } else {
    if (d12 < ters2.r1) {
      fc = 1.0;
      fcp = 0.0;
    } else if (d12 < ters2.r2) {
      fc = cos(ters2.pi_factor * (d12 - ters2.r1)) * 0.5 + 0.5;
      fcp = -sin(ters2.pi_factor * (d12 - ters2.r1)) * ters2.pi_factor * 0.5;
    } else {
      fc = 0.0;
      fcp = 0.0;
    }
  }
}

static __device__ void find_fc(
  int type1,
  int type2,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  const Tersoff1989_Parameters& ters2,
  double d12,
  double& fc)
{
  if (type1 == 0 && type2 == 0) {
    if (d12 < ters0.r1) {
      fc = 1.0;
    } else if (d12 < ters0.r2) {
      fc = cos(ters0.pi_factor * (d12 - ters0.r1)) * 0.5 + 0.5;
    } else {
      fc = 0.0;
    }
  } else if (type1 == 1 && type2 == 1) {
    if (d12 < ters1.r1) {
      fc = 1.0;
    } else if (d12 < ters1.r2) {
      fc = cos(ters1.pi_factor * (d12 - ters1.r1)) * 0.5 + 0.5;
    } else {
      fc = 0.0;
    }
  } else {
    if (d12 < ters2.r1) {
      fc = 1.0;
    } else if (d12 < ters2.r2) {
      fc = cos(ters2.pi_factor * (d12 - ters2.r1)) * 0.5 + 0.5;
    } else {
      fc = 0.0;
    }
  }
}

static __device__ void find_g_and_gp(
  int type1,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  double cos,
  double& g,
  double& gp)
{
  if (type1 == 0) {
    double temp = ters0.d2 + (cos - ters0.h) * (cos - ters0.h);
    g = ters0.one_plus_c2overd2 - ters0.c2 / temp;
    gp = 2.0 * ters0.c2 * (cos - ters0.h) / (temp * temp);
  } else {
    double temp = ters1.d2 + (cos - ters1.h) * (cos - ters1.h);
    g = ters1.one_plus_c2overd2 - ters1.c2 / temp;
    gp = 2.0 * ters1.c2 * (cos - ters1.h) / (temp * temp);
  }
}

static __device__ void find_g(
  int type1,
  const Tersoff1989_Parameters& ters0,
  const Tersoff1989_Parameters& ters1,
  double cos,
  double& g)
{
  if (type1 == 0) {
    double temp = ters0.d2 + (cos - ters0.h) * (cos - ters0.h);
    g = ters0.one_plus_c2overd2 - ters0.c2 / temp;
  } else {
    double temp = ters1.d2 + (cos - ters1.h) * (cos - ters1.h);
    g = ters1.one_plus_c2overd2 - ters1.c2 / temp;
  }
}

// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_tersoff_step1(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const Tersoff1989_Parameters ters0,
  const Tersoff1989_Parameters ters1,
  const Tersoff1989_Parameters ters2,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_b,
  double* g_bp)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double zeta = 0.0;
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + number_of_particles * i2];
        if (n3 == n2) {
          continue;
        } // ensure that n3 != n2
        int type3 = g_type[n3];
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
        double fc13, g123;
        find_fc(type1, type3, ters0, ters1, ters2, d13, fc13);
        find_g(type1, ters0, ters1, cos123, g123);
        zeta += fc13 * g123;
      }
      double bzn, b12;
      if (type1 == 0) {
        bzn = pow(ters0.beta * zeta, ters0.n);
        b12 = pow(1.0 + bzn, ters0.minus_half_over_n);
      } else {
        bzn = pow(ters1.beta * zeta, ters1.n);
        b12 = pow(1.0 + bzn, ters1.minus_half_over_n);
      }
      if (zeta < 1.0e-16f) // avoid division by 0
      {
        g_b[i1 * number_of_particles + n1] = 1.0;
        g_bp[i1 * number_of_particles + n1] = 0.0;
      } else {
        g_b[i1 * number_of_particles + n1] = b12;
        g_bp[i1 * number_of_particles + n1] = -b12 * bzn * 0.5 / ((1.0 + bzn) * zeta);
      }
    }
  }
}

// step 2: calculate all the partial forces dU_i/dr_ij
static __global__ void __launch_bounds__(BLOCK_SIZE_FORCE, 10) find_force_tersoff_step2(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const Tersoff1989_Parameters ters0,
  const Tersoff1989_Parameters ters1,
  const Tersoff1989_Parameters ters2,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const double* __restrict__ g_b,
  const double* __restrict__ g_bp,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_potential,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    double potential_energy = 0.0;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int type2 = g_type[n2];

      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double d12inv = 1.0 / d12;
      double fc12, fcp12, fa12, fap12, fr12, frp12;
      find_fc_and_fcp(type1, type2, ters0, ters1, ters2, d12, fc12, fcp12);
      find_fa_and_fap(type1, type2, ters0, ters1, ters2, d12, fa12, fap12);
      find_fr_and_frp(type1, type2, ters0, ters1, ters2, d12, fr12, frp12);

      // (i,j) part
      double b12 = g_b[index];
      double factor3 = (fcp12 * (fr12 - b12 * fa12) + fc12 * (frp12 - b12 * fap12)) * d12inv;
      double f12x = x12 * factor3 * 0.5;
      double f12y = y12 * factor3 * 0.5;
      double f12z = z12 * factor3 * 0.5;

      // accumulate potential energy
      potential_energy += fc12 * (fr12 - b12 * fa12) * 0.5;

      // (i,j,k) part
      double bp12 = g_bp[index];
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int index_2 = n1 + number_of_particles * i2;
        int n3 = g_neighbor_list[index_2];
        if (n3 == n2) {
          continue;
        }
        int type3 = g_type[n3];
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        double fc13, fa13;
        find_fc(type1, type3, ters0, ters1, ters2, d13, fc13);
        find_fa(type1, type3, ters0, ters1, ters2, d13, fa13);

        double bp13 = g_bp[index_2];
        double one_over_d12d13 = 1.0 / (d12 * d13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) * one_over_d12d13;
        double cos123_over_d12d12 = cos123 * d12inv * d12inv;
        double g123, gp123;
        find_g_and_gp(type1, ters0, ters1, cos123, g123, gp123);

        double temp123a = (-bp12 * fc12 * fa12 * fc13 - bp13 * fc13 * fa13 * fc12) * gp123;
        double temp123b = -bp13 * fc13 * fa13 * fcp12 * g123 * d12inv;
        double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
        f12x += (x12 * temp123b + temp123a * cos_d) * 0.5;
        cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
        f12y += (y12 * temp123b + temp123a * cos_d) * 0.5;
        cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
        f12z += (z12 * temp123b + temp123a * cos_d) * 0.5;
      }
      g_f12x[index] = f12x;
      g_f12y[index] = f12y;
      g_f12z[index] = f12z;
    }
    // save potential
    g_potential[n1] += potential_energy;
  }
}

// Wrapper of force evaluation for the Tersoff potential
void Tersoff1989::compute(
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
      N1, N2, rc, box, type, position_per_atom, tersoff_data.cell_count,
      tersoff_data.cell_count_sum, tersoff_data.cell_contents, tersoff_data.NN, tersoff_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  // pre-compute the bond order functions and their derivatives
  find_force_tersoff_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, ters0, ters1, ters2, tersoff_data.NN.data(),
    tersoff_data.NL.data(), type.data(), position_per_atom.data(),
    position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
    tersoff_data.b.data(), tersoff_data.bp.data());
  CUDA_CHECK_KERNEL

  // pre-compute the partial forces
  find_force_tersoff_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, ters0, ters1, ters2, tersoff_data.NN.data(),
    tersoff_data.NL.data(), type.data(), tersoff_data.b.data(), tersoff_data.bp.data(),
    position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(),
    tersoff_data.f12x.data(), tersoff_data.f12y.data(), tersoff_data.f12z.data());
  CUDA_CHECK_KERNEL

  // the final step: calculate force and related quantities
  find_properties_many_body(
    box, tersoff_data.NN.data(), tersoff_data.NL.data(), tersoff_data.f12x.data(),
    tersoff_data.f12y.data(), tersoff_data.f12z.data(), position_per_atom, force_per_atom,
    virial_per_atom);
}
