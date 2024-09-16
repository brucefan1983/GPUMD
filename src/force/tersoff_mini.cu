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
The minimal Tersoff potential, as developed in the following paper:

Z. Fan, Y. Wang, X. Gu, P. Qian, Y. Su, and T. Ala-Nissila,
A minimal Tersoff potential for diamond silicon with improved
descriptions of elastic and phonon transport properties,
J. Phys.: Condens. Matter 32, 135901 (2020).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "tersoff_mini.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

#define BLOCK_SIZE_FORCE 64

Tersoff_mini::Tersoff_mini(FILE* fid, int num_of_types, const int num_atoms)
{
  num_types = num_of_types;
  printf("Use Tersoff-mini (%d-element) potential with element(s):\n", num_types);
  for (int n = 0; n < num_types; ++n) {
    char atom_symbol[10];
    int count = fscanf(fid, "%s", atom_symbol);
    PRINT_SCANF_ERROR(count, 1, "Reading error for Tersoff-mini potential.");
    printf(" %s", atom_symbol);
  }
  printf("\n");

  int n_entries = 2 * num_types - 1; // 1 or 3 entries

  const char err[] = "Reading error for Tersoff-mini potential.\n";
  rc = 0.0;
  int count;
  double d0, a, r0, s, beta, n, h, r1, r2;
  for (int i = 0; i < n_entries; i++) {
    count = fscanf(fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &d0, &a, &r0, &s, &beta, &n, &h, &r1, &r2);
    PRINT_SCANF_ERROR(count, 9, err);
    if (d0 <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (a <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (r0 <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (beta < 0.0)
      PRINT_INPUT_ERROR(err);
    if (n < 0.0)
      PRINT_INPUT_ERROR(err);
    if (h < -1.0 || h > 1.0)
      PRINT_INPUT_ERROR(err);
    if (r1 < 0.0)
      PRINT_INPUT_ERROR(err);
    if (r2 <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (r2 <= r1)
      PRINT_INPUT_ERROR(err);

    para.a[i] = d0 / (s - 1.0) * exp(sqrt(2.0 * s) * a * r0);
    para.b[i] = s * d0 / (s - 1.0) * exp(sqrt(2.0 / s) * a * r0);
    para.lambda[i] = sqrt(2.0 * s) * a;
    para.mu[i] = sqrt(2.0 / s) * a;
    para.beta[i] = beta;
    para.n[i] = n;
    para.h[i] = h;
    para.r1[i] = r1;
    para.r2[i] = r2;
    para.pi_factor[i] = PI / (r2 - r1);
    para.minus_half_over_n[i] = -0.5 / n;
    rc = r2 > rc ? r2 : rc;
  }

  const int num_of_neighbors = 50 * num_atoms;
  tersoff_mini_data.b.resize(num_of_neighbors);
  tersoff_mini_data.bp.resize(num_of_neighbors);
  tersoff_mini_data.f12x.resize(num_of_neighbors);
  tersoff_mini_data.f12y.resize(num_of_neighbors);
  tersoff_mini_data.f12z.resize(num_of_neighbors);
  tersoff_mini_data.NN.resize(num_atoms);
  tersoff_mini_data.NL.resize(num_of_neighbors);
  tersoff_mini_data.cell_count.resize(num_atoms);
  tersoff_mini_data.cell_count_sum.resize(num_atoms);
  tersoff_mini_data.cell_contents.resize(num_atoms);
}

Tersoff_mini::~Tersoff_mini(void)
{
  // nothing
}

static __device__ void find_fr_and_frp(double a, double lambda, double d12, double& fr, double& frp)
{
  fr = a * exp(-lambda * d12);
  frp = -lambda * fr;
}

static __device__ void find_fa_and_fap(double b, double mu, double d12, double& fa, double& fap)
{
  fa = b * exp(-mu * d12);
  fap = -mu * fa;
}

static __device__ void find_fa(double b, double mu, double d12, double& fa)
{
  fa = b * exp(-mu * d12);
}

static __device__ void
find_fc_and_fcp(double r1, double r2, double pi_factor, double d12, double& fc, double& fcp)
{
  if (d12 < r1) {
    fc = 1.0;
    fcp = 0.0;
  } else if (d12 < r2) {
    fc = 0.5 * cos(pi_factor * (d12 - r1)) + 0.5;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

static __device__ void find_fc(double r1, double r2, double pi_factor, double d12, double& fc)
{
  if (d12 < r1) {
    fc = 1.0;
  } else if (d12 < r2) {
    fc = 0.5 * cos(pi_factor * (d12 - r1)) + 0.5;
  } else {
    fc = 0.0;
  }
}

static __device__ void find_g_and_gp(double h, double cos, double& g, double& gp)
{
  double tmp = cos - h;
  g = tmp * tmp;
  gp = 2.0 * tmp;
}

static __device__ void find_g(double h, double cos, double& g)
{
  double tmp = cos - h;
  g = tmp * tmp;
}

// step 1: pre-compute all the bond-order functions and their derivatives
static __global__ void find_force_step1(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int num_types,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const Tersoff_mini_Para para,
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
      int type12 = type1 + g_type[n2];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double zeta = 0.0;
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + number_of_particles * i2];
        int type13 = type1 + g_type[n3];
        if (n3 == n2) {
          continue;
        } // ensure that n3 != n2
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
        double fc13, g123;
        find_fc(para.r1[type13], para.r2[type13], para.pi_factor[type13], d13, fc13);
        find_g(para.h[type12], cos123, g123);
        zeta += fc13 * g123;
      }

      double bzn, b12;
      bzn = pow(para.beta[type12] * zeta, para.n[type12]);
      b12 = pow(1.0 + bzn, para.minus_half_over_n[type12]);
      if (zeta < 1.0e-16) // avoid division by 0
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
static __global__ void __launch_bounds__(BLOCK_SIZE_FORCE, 10) find_force_step2(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int num_types,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const int* g_type,
  const Tersoff_mini_Para para,
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
    double pot_energy = 0.0;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int type12 = type1 + g_type[n2];

      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double d12inv = 1.0 / d12;
      double fc12, fcp12, fa12, fap12, fr12, frp12;
      find_fc_and_fcp(para.r1[type12], para.r2[type12], para.pi_factor[type12], d12, fc12, fcp12);
      find_fa_and_fap(para.b[type12], para.mu[type12], d12, fa12, fap12);
      find_fr_and_frp(para.a[type12], para.lambda[type12], d12, fr12, frp12);

      // (i,j) part
      double b12 = g_b[index];
      double factor3 = (fcp12 * (fr12 - b12 * fa12) + fc12 * (frp12 - b12 * fap12)) * d12inv;
      double f12x = x12 * factor3 * 0.5;
      double f12y = y12 * factor3 * 0.5;
      double f12z = z12 * factor3 * 0.5;

      // accumulate potential energy
      pot_energy += fc12 * (fr12 - b12 * fa12) * 0.5;

      // (i,j,k) part
      double bp12 = g_bp[index];
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int index_2 = n1 + number_of_particles * i2;
        int n3 = g_neighbor_list[index_2];
        if (n3 == n2) {
          continue;
        }
        int type13 = type1 + g_type[n3];
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        double fc13, fa13;
        find_fc(para.r1[type13], para.r2[type13], para.pi_factor[type13], d13, fc13);
        find_fa(para.b[type13], para.mu[type13], d13, fa13);
        double bp13 = g_bp[index_2];
        double one_over_d12d13 = 1.0 / (d12 * d13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) * one_over_d12d13;
        double cos123_over_d12d12 = cos123 * d12inv * d12inv;
        double g123, gp123;
        find_g_and_gp(para.h[type12], cos123, g123, gp123);
        // derivatives with cosine
        double dc = -fc12 * bp12 * fa12 * fc13 * gp123 - fc12 * bp13 * fa13 * fc13 * gp123;
        // derivatives with rij
        double dr = -fcp12 * bp13 * fa13 * g123 * fc13 * d12inv;
        double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
        f12x += (x12 * dr + dc * cos_d) * 0.5;
        cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
        f12y += (y12 * dr + dc * cos_d) * 0.5;
        cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
        f12z += (z12 * dr + dc * cos_d) * 0.5;
      }
      g_f12x[index] = f12x;
      g_f12y[index] = f12y;
      g_f12z[index] = f12z;
    }
    // save potential
    g_potential[n1] += pot_energy;
  }
}

// Wrapper of force evaluation for the SBOP potential
void Tersoff_mini::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

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
      tersoff_mini_data.cell_count,
      tersoff_mini_data.cell_count_sum,
      tersoff_mini_data.cell_contents,
      tersoff_mini_data.NN,
      tersoff_mini_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  // pre-compute the bond order functions and their derivatives
  find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    num_types,
    tersoff_mini_data.NN.data(),
    tersoff_mini_data.NL.data(),
    type.data(),
    para,
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    tersoff_mini_data.b.data(),
    tersoff_mini_data.bp.data());
  CUDA_CHECK_KERNEL

  // pre-compute the partial forces
  find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    N1,
    N2,
    box,
    num_types,
    tersoff_mini_data.NN.data(),
    tersoff_mini_data.NL.data(),
    type.data(),
    para,
    tersoff_mini_data.b.data(),
    tersoff_mini_data.bp.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    tersoff_mini_data.f12x.data(),
    tersoff_mini_data.f12y.data(),
    tersoff_mini_data.f12z.data());
  CUDA_CHECK_KERNEL

  // the final step: calculate force and related quantities
  find_properties_many_body(
    box,
    tersoff_mini_data.NN.data(),
    tersoff_mini_data.NL.data(),
    tersoff_mini_data.f12x.data(),
    tersoff_mini_data.f12y.data(),
    tersoff_mini_data.f12z.data(),
    position_per_atom,
    force_per_atom,
    virial_per_atom);
}
