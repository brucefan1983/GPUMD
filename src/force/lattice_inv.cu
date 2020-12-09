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

------------------------------------------------------------------------------*/

#include "lattice_inv.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

#define BLOCK_SIZE_FORCE 64

Lattice_Inv::Lattice_Inv(FILE* fid, int num_of_types, const Neighbor& neighbor)
{
  num_types = num_of_types;
  printf("Use Lattice_Inv (%d-element) potential.\n", num_types);
  int n_entries = 2 * num_types - 1; // 1 or 3 entries

  const char err[] = "Reading error for Lattice_Inv potential.\n";
  rc = 0.0;
  int count;
  double d0, a, r0, s, beta, gama, n, h, r1, r2;
  for (int i = 0; i < n_entries; i++) {
    count = fscanf(fid, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &d0, &a, &r0, &s, &beta, &gama, &n, &h, &r1, &r2);
    PRINT_SCANF_ERROR(count, 10, err);
    if (d0 <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (a <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (r0 <= 0.0)
      PRINT_INPUT_ERROR(err);
    if (gama < 0.0)
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
  para.minus_half_over_n[i] = -0.5 / n;
  para.gama[i]= gama;
  para.r1[i] = r1;
  para.r2[i] = r2;
  rc = r2;
  }

  const int num_of_neighbors = min(neighbor.MN, 50) * neighbor.NN.size();
  lattice_inv_data.b.resize(num_of_neighbors);
  lattice_inv_data.bp.resize(num_of_neighbors);
  lattice_inv_data.f12x.resize(num_of_neighbors);
  lattice_inv_data.f12y.resize(num_of_neighbors);
  lattice_inv_data.f12z.resize(num_of_neighbors);
  CHECK(cudaMallocManaged((void**)&lattice_inv_data.NN_short, num_of_neighbors));
  CHECK(cudaMallocManaged((void**)&lattice_inv_data.NL_short, num_of_neighbors* 20));
}

Lattice_Inv::~Lattice_Inv(void)
{
  CHECK(cudaFree(lattice_inv_data.NN_short));
  CHECK(cudaFree(lattice_inv_data.NL_short));
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

// two-body part(kernel)
static __global__ void find_force_step0
(
  const int number_of_particles, 
  const int N1, 
  const int N2, 
  const Box box,
  const int num_types,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  int* g_short_number,
  int* g_short_list,
  const Lattice_Inv_Para para,
  const int* g_type,
  const int shift,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx, 
  double* g_fy, 
  double* g_fz,
  double* g_virial, 
  double* g_potential
)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index
  double s_fx = 0; // force_x
  double s_fy = 0; // force_y
  double s_fz = 0; // force_z
  double s_pe = 0; // potential energy
  double s_sxx = 0; // virial_stress_xx
  double s_sxy = 0; // virial_stress_xy
  double s_sxz = 0; // virial_stress_xz
  double s_syx = 0; // virial_stress_yx
  double s_syy = 0; // virial_stress_yy
  double s_syz = 0; // virial_stress_yz
  double s_szx = 0; // virial_stress_zx
  double s_szy = 0; // virial_stress_zy
  double s_szz = 0; // virial_stress_zz
  if (n1 >= N1 && n1 < N2){ 
     int neighbor_number = g_neighbor_number[n1];
     int type1 = g_type[n1] - shift;
     double x1 = g_x[n1]; 
     double y1 = g_y[n1]; 
     double z1 = g_z[n1]; 
     int count = 0; // initialize g_NN_local[n1] to 0
     for (int i1 = 0; i1 < neighbor_number; ++i1)
     {
        int n2 = g_neighbor_list[n1 + number_of_particles * i1];
        int type12 = type1 + g_type[n2] - shift;
        double x12  =  g_x[n2] - x1;
        double y12  =  g_y[n2] - y1;
        double z12  =  g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

        if (d12 < para.r1[type12])
        {        
          // build the 3-body neighbor list            
          g_short_list[n1 + number_of_particles * (count++)] = n2;
        }
        else
        {
          double fa12, fap12, fr12, frp12;
          find_fa_and_fap(para.b[type12], para.mu[type12], d12, fa12, fap12);
          find_fr_and_frp(para.a[type12], para.lambda[type12], d12, fr12, frp12);
          double p2 = fr12 - fa12;
          double f2 = (frp12 - fap12) / d12;
            
          // treat two-body potential in the same way as many-body potential
          double f12x = f2 * x12 * 0.5; 
          double f12y = f2 * y12 * 0.5; 
          double f12z = f2 * z12 * 0.5; 
          double f21x = -f12x; 
          double f21y = -f12y; 
          double f21z = -f12z; 
       
          // accumulate force
          s_fx += (f12x - f21x); 
          s_fy += (f12y - f21y); 
          s_fz += (f12z - f21z); 

          // accumulate potential energy and virial 
          s_pe += p2 * 0.5; // two-body potential
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
      }

      g_short_number[n1] = count; // now the local neighbor list has been built
  
      g_fx[n1] += s_fx; // save force
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
  const int shift,
  const Lattice_Inv_Para para,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_b,
  double* g_bp)
{
  // start from the N1-th atom
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  // to the (N2-1)-th atom
  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1] - shift;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];
      int type12 = type1 + g_type[n2] - shift;
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
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
        double g123;
        find_g(para.h[type12], cos123, g123);
        zeta += g123;
      }
      double bzn, b12;
      bzn = pow(para.beta[type12] * zeta, para.n[type12]);
      b12 = pow(1.0 + bzn, para.minus_half_over_n[type12]);  
      if (zeta < 1.0e-16) // avoid division by 0
      {
        g_b[i1 * number_of_particles + n1] = 1.0;
        g_bp[i1 * number_of_particles + n1] = 0.0;
      } else {
        g_b[i1 * number_of_particles + n1]  =  b12;
        g_bp[i1 * number_of_particles + n1] =  -b12 * bzn * 0.5 / ((1.0 + bzn) * zeta);
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
  const int shift,
  const Lattice_Inv_Para para,
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
  // start from the N1-th atom
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  // to the (N2-1)-th atom
  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    int type1 = g_type[n1] - shift;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    double pot_energy = 0.0;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int type12 = type1 + g_type[n2] - shift;
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double d12inv = 1.0 / d12;
      double fa12, fap12, fr12, frp12;
      find_fa_and_fap(para.b[type12], para.mu[type12], d12, fa12, fap12);
      find_fr_and_frp(para.a[type12], para.lambda[type12], d12, fr12, frp12);

      // (i,j) part
      double b12 = g_b[index];
      double factor3 =  (frp12 - b12 * fap12) * d12inv;
      double f12x = x12 * factor3 * 0.5;
      double f12y = y12 * factor3 * 0.5;
      double f12z = z12 * factor3 * 0.5;

      // accumulate potential energy
      pot_energy += (fr12 - b12 * fa12) * 0.5;

      // (i,j,k) part
      double bp12 = g_bp[index];
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int index_2 = n1 + number_of_particles * i2;
        int n3 = g_neighbor_list[index_2];
        if (n3 == n2) {
          continue;
        }
        int type13 = type1 + g_type[n3] - shift;
        double x13 = g_x[n3] - x1;
        double y13 = g_y[n3] - y1;
        double z13 = g_z[n3] - z1;
        apply_mic(box, x13, y13, z13);
        double d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        double fa13;
        find_fa(para.b[type13], para.mu[type13], d13, fa13);
        double bp13 = g_bp[index_2];
        double one_over_d12d13 = 1.0 / (d12 * d13);
        double cos123 = (x12 * x13 + y12 * y13 + z12 * z13) * one_over_d12d13;
        double cos123_over_d12d12 = cos123 * d12inv * d12inv;
        double g123, gp123;
        find_g_and_gp(para.h[type12], cos123, g123, gp123);
        // derivatives with cosine
        double dc = - bp12 * fa12 * gp123 - bp13 * fa13 * gp123;
        // derivatives with rij
        double cos_d = x13 * one_over_d12d13 - x12 * cos123_over_d12d12;
        f12x += (dc * cos_d) * 0.5;
        cos_d = y13 * one_over_d12d13 - y12 * cos123_over_d12d12;
        f12y += (dc * cos_d) * 0.5;
        cos_d = z13 * one_over_d12d13 - z12 * cos123_over_d12d12;
        f12z += (dc * cos_d) * 0.5;
      }
      g_f12x[index] = f12x;
      g_f12y[index] = f12y;
      g_f12z[index] = f12z;
    }
    // save potential
    g_potential[n1] += pot_energy;
  }
}
static __global__ void __launch_bounds__(BLOCK_SIZE_FORCE, 10) find_force_step3( 
  const int N1,
  const int N2,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  const double gama
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N1 && n1 < N2){
    g_fx[n1] = g_fx[n1] * gama; // save force
    g_fy[n1] = g_fy[n1] * gama;
    g_fz[n1] = g_fz[n1] * gama;
   }
  
}

// Wrapper of force evaluation for the SBOP potential
void Lattice_Inv::compute(
  const int type_shift,
  const Box& box,
  const Neighbor& neighbor,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
  const double gama = para.gama[0];
  int* NN_short = lattice_inv_data.NN_short;
  int* NL_short = lattice_inv_data.NL_short;
  // pre-compute the bond order functions and their derivatives
  find_force_step0<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, num_types, neighbor.NN_local.data(), neighbor.NL_local.data(),
 NN_short, NL_short, para, type.data(), type_shift, position_per_atom.data(), position_per_atom.data() + number_of_atoms, 
 position_per_atom.data() + number_of_atoms * 2, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms, virial_per_atom.data(), potential_per_atom.data());
  CUDA_CHECK_KERNEL
    
  find_force_step1<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, num_types, NN_short, NL_short,type.data(), 
 type_shift, para, position_per_atom.data(), position_per_atom.data() + number_of_atoms, 
 position_per_atom.data() + number_of_atoms * 2, lattice_inv_data.b.data(), lattice_inv_data.bp.data());
  CUDA_CHECK_KERNEL

  // pre-compute the partial forces
  find_force_step2<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, num_types, NN_short, NL_short, 
    type.data(), type_shift, para, lattice_inv_data.b.data(), lattice_inv_data.bp.data(),
    position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(),
    lattice_inv_data.f12x.data(), lattice_inv_data.f12y.data(), lattice_inv_data.f12z.data());
  CUDA_CHECK_KERNEL
  
  // calculate force and related quantities
  find_properties_many_body(
    box, NN_short, NL_short, lattice_inv_data.f12x.data(),
    lattice_inv_data.f12y.data(), lattice_inv_data.f12z.data(), position_per_atom, force_per_atom,
    virial_per_atom);
 
  // calculate real force
  find_force_step3<<<grid_size, BLOCK_SIZE_FORCE>>>(N1, N2, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms, gama);
  CUDA_CHECK_KERNEL



}
