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
The EAM potential. Currently two analytical versions:
[1] X. W. Zhou et al. Phys. Rev. B 69, 144113 (2004).
[2] X. D. Dai et al. JPCM 18, 4527 (2006).
------------------------------------------------------------------------------*/

#include "eam.cuh"
#include "utilities/error.cuh"
#define BLOCK_SIZE_FORCE 64

EAM::EAM(FILE* fid, char* name, int num_types, const int number_of_atoms)
{
  if (strcmp(name, "eam_zhou_2004") == 0)
    initialize_eam2004zhou(fid, num_types);
  if (strcmp(name, "eam_dai_2006") == 0) {
    initialize_eam2006dai(fid);
    if (num_types > 1) {
      PRINT_INPUT_ERROR(
        "eam_dai_2006 has not been implemented for systems with two or more atom types.");
    }
  }

  eam_data.Fp.resize(number_of_atoms);
}

void EAM::initialize_eam2004zhou(FILE* fid, int num_types)
{
  printf("Use the EAM-type potential in the following reference:\n");
  printf("    X. W. Zhou et al., PRB 69, 144113 (2004).\n");
  potential_model = 0;

  rc = 0.0;
  for (int type = 0; type < num_types; ++type) {
    double x[21];
    for (int n = 0; n < 21; n++) {
      int count = fscanf(fid, "%lf", &x[n]);
      PRINT_SCANF_ERROR(count, 1, "Reading error for EAM potential.");
    }
    eam2004zhou.re[type] = x[0];
    eam2004zhou.fe[type] = x[1];
    eam2004zhou.rho_e[type] = x[2];
    eam2004zhou.rho_s[type] = x[3];
    eam2004zhou.alpha[type] = x[4];
    eam2004zhou.beta[type] = x[5];
    eam2004zhou.A[type] = x[6];
    eam2004zhou.B[type] = x[7];
    eam2004zhou.kappa[type] = x[8];
    eam2004zhou.lambda[type] = x[9];
    eam2004zhou.Fn0[type] = x[10];
    eam2004zhou.Fn1[type] = x[11];
    eam2004zhou.Fn2[type] = x[12];
    eam2004zhou.Fn3[type] = x[13];
    eam2004zhou.F0[type] = x[14];
    eam2004zhou.F1[type] = x[15];
    eam2004zhou.F2[type] = x[16];
    eam2004zhou.F3[type] = x[17];
    eam2004zhou.eta[type] = x[18];
    eam2004zhou.Fe[type] = x[19];
    eam2004zhou.rc[type] = x[20];
    eam2004zhou.rho_n[type] = eam2004zhou.rho_e[type] * 0.85;
    eam2004zhou.rho_0[type] = eam2004zhou.rho_e[type] * 1.15;
    if (rc < eam2004zhou.rc[type]) {
      rc = eam2004zhou.rc[type];
    }
  }
}

void EAM::initialize_eam2006dai(FILE* fid)
{
  printf("Use the EAM-type potential in the following reference:\n");
  printf("    X. D. Dai et al., JPCM 18, 4527 (2006).\n");
  potential_model = 1;

  double x[9];
  for (int n = 0; n < 9; n++) {
    int count = fscanf(fid, "%lf", &x[n]);
    PRINT_SCANF_ERROR(count, 1, "Reading error for EAM potential.");
  }
  eam2006dai.A = x[0];
  eam2006dai.d = x[1];
  eam2006dai.c = x[2];
  eam2006dai.c0 = x[3];
  eam2006dai.c1 = x[4];
  eam2006dai.c2 = x[5];
  eam2006dai.c3 = x[6];
  eam2006dai.c4 = x[7];
  eam2006dai.B = x[8];
  eam2006dai.rc = (eam2006dai.c > eam2006dai.d) ? eam2006dai.c : eam2006dai.d;
  rc = eam2006dai.rc;
}

EAM::~EAM(void)
{
  // nothing
}

// pair function (phi and phip have been intentionally halved here)
static __device__ void
find_phi(const EAM2004Zhou& eam, const int type, const double d12, double& phi, double& phip)
{
  double r_ratio = d12 / eam.re[type];
  double tmp1 = (r_ratio - eam.kappa[type]) * (r_ratio - eam.kappa[type]);   // 2
  tmp1 *= tmp1;                                                              // 4
  tmp1 *= tmp1 * tmp1 * tmp1 * tmp1;                                         // 20
  double tmp2 = (r_ratio - eam.lambda[type]) * (r_ratio - eam.lambda[type]); // 2
  tmp2 *= tmp2;                                                              // 4
  tmp2 *= tmp2 * tmp2 * tmp2 * tmp2;                                         // 20
  double phi1 = 0.5 * eam.A[type] * exp(-eam.alpha[type] * (r_ratio - 1.0)) / (1.0 + tmp1);
  double phi2 = 0.5 * eam.B[type] * exp(-eam.beta[type] * (r_ratio - 1.0)) / (1.0 + tmp2);
  phi = phi1 - phi2;
  phip = (phi2 / eam.re[type]) *
           (eam.beta[type] + 20.0 * tmp2 / (r_ratio - eam.lambda[type]) / (1.0 + tmp2)) -
         (phi1 / eam.re[type]) *
           (eam.alpha[type] + 20.0 * tmp1 / (r_ratio - eam.kappa[type]) / (1.0 + tmp1));
}

// density function f(r)
static __device__ void find_f(const EAM2004Zhou& eam, const int type, const double d12, double& f)
{
  double r_ratio = d12 / eam.re[type];
  double tmp = (r_ratio - eam.lambda[type]) * (r_ratio - eam.lambda[type]); // 2
  tmp *= tmp;                                                               // 4
  tmp *= tmp * tmp * tmp * tmp;                                             // 20
  f = eam.fe[type] * exp(-eam.beta[type] * (r_ratio - 1.0)) / (1.0 + tmp);
}

// derivative of the density function f'(r)
static __device__ void find_fp(const EAM2004Zhou& eam, const int type, const double d12, double& fp)
{
  double r_ratio = d12 / eam.re[type];
  double tmp = (r_ratio - eam.lambda[type]) * (r_ratio - eam.lambda[type]); // 2
  tmp *= tmp;                                                               // 4
  tmp *= tmp * tmp * tmp * tmp;                                             // 20
  double f = eam.fe[type] * exp(-eam.beta[type] * (r_ratio - 1.0)) / (1.0 + tmp);
  fp = -(f / eam.re[type]) *
       (eam.beta[type] + 20.0 * tmp / (r_ratio - eam.lambda[type]) / (1.0 + tmp));
}

static __device__ void
find_f_and_fp(const EAM2004Zhou& eam, const int type, const double d12, double& f, double& fp)
{
  double r_ratio = d12 / eam.re[type];
  double tmp = (r_ratio - eam.lambda[type]) * (r_ratio - eam.lambda[type]); // 2
  tmp *= tmp;                                                               // 4
  tmp *= tmp * tmp * tmp * tmp;                                             // 20
  f = eam.fe[type] * exp(-eam.beta[type] * (r_ratio - 1.0)) / (1.0 + tmp);
  fp = -(f / eam.re[type]) *
       (eam.beta[type] + 20.0 * tmp / (r_ratio - eam.lambda[type]) / (1.0 + tmp));
}

// pair function for EAM2004Zhou
static __device__ void find_phi(
  const EAM2004Zhou& eam,
  const int type1,
  const int type2,
  const double d12,
  double& phi,
  double& phip)
{
  if (type1 == type2) {
    find_phi(eam, type1, d12, phi, phip);
  } else {
    double phi1, phip1;
    find_phi(eam, type1, d12, phi1, phip1);
    double phi2, phip2;
    find_phi(eam, type2, d12, phi2, phip2);
    double f1, fp1;
    find_f_and_fp(eam, type1, d12, f1, fp1);
    double f2, fp2;
    find_f_and_fp(eam, type2, d12, f2, fp2);
    double f1inv = 1.0 / f1;
    double f2inv = 1.0 / f2;
    phi = 0.25 * (phi1 * f2 * f1inv + phi2 * f1 * f2inv);
    phip = (phip1 * f2 + phi1 * fp2) * f1inv - phi1 * f2 * fp1 * f1inv * f1inv;
    phip += (phip2 * f1 + phi2 * fp1) * f2inv - phi2 * f1 * fp2 * f2inv * f2inv;
    phip *= 0.5;
  }
}

// embedding function
static __device__ void
find_F(const EAM2004Zhou& eam, const int type, const double rho, double& F, double& Fp)
{
  if (rho < eam.rho_n[type]) {
    double x = rho / eam.rho_n[type] - 1.0;
    F = ((eam.Fn3[type] * x + eam.Fn2[type]) * x + eam.Fn1[type]) * x + eam.Fn0[type];
    Fp = ((3.0 * eam.Fn3[type] * x + 2.0 * eam.Fn2[type]) * x + eam.Fn1[type]) / eam.rho_n[type];
  } else if (rho < eam.rho_0[type]) {
    double x = rho / eam.rho_e[type] - 1.0;
    F = ((eam.F3[type] * x + eam.F2[type]) * x + eam.F1[type]) * x + eam.F0[type];
    Fp = ((3.0 * eam.F3[type] * x + 2.0 * eam.F2[type]) * x + eam.F1[type]) / eam.rho_e[type];
  } else {
    double x = rho / eam.rho_s[type];
    double x_eta = pow(x, eam.eta[type]);
    F = eam.Fe[type] * (1.0 - eam.eta[type] * log(x)) * x_eta;
    Fp = (eam.eta[type] / rho) * (F - eam.Fe[type] * x_eta);
  }
}

// pair function (phi and phip have been intentionally halved here)
static __device__ void find_phi(const EAM2006Dai& fs, const double d12, double& phi, double& phip)
{
  if (d12 > fs.c) {
    phi = 0.0;
    phip = 0.0;
  } else {
    double tmp = ((((fs.c4 * d12 + fs.c3) * d12 + fs.c2) * d12 + fs.c1) * d12 + fs.c0);

    phi = 0.5 * (d12 - fs.c) * (d12 - fs.c) * tmp;

    phip = 2.0 * (d12 - fs.c) * tmp;
    phip += (((4.0 * fs.c4 * d12 + 3.0 * fs.c3) * d12 + 2.0 * fs.c2) * d12 + fs.c1) * (d12 - fs.c) *
            (d12 - fs.c);
    phip *= 0.5;
  }
}

// density function f(r)
static __device__ void find_f(const EAM2006Dai& fs, const double d12, double& f)
{
  if (d12 > fs.d) {
    f = 0.0;
  } else {
    double tmp = (d12 - fs.d) * (d12 - fs.d);
    f = tmp + fs.B * fs.B * tmp * tmp;
  }
}

// derivative of the density function f'(r)
static __device__ void find_fp(const EAM2006Dai& fs, const double d12, double& fp)
{
  if (d12 > fs.d) {
    fp = 0.0;
  } else {
    double tmp = 2.0 * (d12 - fs.d);
    fp = tmp * (1.0 + fs.B * fs.B * tmp * (d12 - fs.d));
  }
}

// embedding function
static __device__ void find_F(const EAM2006Dai& fs, const double rho, double& F, double& Fp)
{
  double sqrt_rho = sqrt(rho);
  F = -fs.A * sqrt_rho;
  Fp = -fs.A * 0.5 / sqrt_rho;
}

// Calculate the embedding energy and its derivative
template <int potential_model>
static __global__ void find_force_eam_step1(
  const EAM2004Zhou eam2004zhou,
  const EAM2006Dai eam2006dai,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_Fp,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1; // particle index

  if (n1 < N2) {
    int NN = g_NN[n1];

    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    // Calculate the density
    double rho = 0.0;
    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double rho12 = 0.0;
      if (potential_model == 0) {
        find_f(eam2004zhou, g_type[n2], d12, rho12); // density is contributed by n2
      }
      if (potential_model == 1) {
        find_f(eam2006dai, d12, rho12);
      }
      rho += rho12;
    }

    // Calculate the embedding energy F and its derivative Fp
    double F, Fp;
    if (potential_model == 0)
      find_F(eam2004zhou, g_type[n1], rho, F, Fp); // embedding energy is for n1
    if (potential_model == 1)
      find_F(eam2006dai, rho, F, Fp);

    g_pe[n1] += F; // many-body potential energy
    g_Fp[n1] = Fp;
  }
}

// Force evaluation kernel
template <int potential_model>
static __global__ void find_force_eam_step2(
  const EAM2004Zhou eam2004zhou,
  const EAM2006Dai eam2006dai,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* __restrict__ g_Fp,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  double s_fx = 0.0;  // force_x
  double s_fy = 0.0;  // force_y
  double s_fz = 0.0;  // force_z
  double s_pe = 0.0;  // potential energy
  double s_sxx = 0.0; // virial_stress_xx
  double s_sxy = 0.0; // virial_stress_xy
  double s_sxz = 0.0; // virial_stress_xz
  double s_syx = 0.0; // virial_stress_yx
  double s_syy = 0.0; // virial_stress_yy
  double s_syz = 0.0; // virial_stress_yz
  double s_szx = 0.0; // virial_stress_zx
  double s_szy = 0.0; // virial_stress_zy
  double s_szz = 0.0; // virial_stress_zz

  if (n1 < N2) {
    int type1 = g_type[n1];
    int NN = g_NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    double Fp1 = g_Fp[n1];

    for (int i1 = 0; i1 < NN; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int type2 = g_type[n2];
      double Fp2 = g_Fp[n2];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

      double phi, phip, fp1, fp2;
      if (potential_model == 0) {
        find_phi(eam2004zhou, type1, type2, d12, phi, phip);
        if (type1 == type2) {
          find_fp(eam2004zhou, type1, d12, fp1);
          fp2 = fp1;
        } else {
          find_fp(eam2004zhou, type1, d12, fp1);
          find_fp(eam2004zhou, type2, d12, fp2);
        }
      }
      if (potential_model == 1) {
        find_phi(eam2006dai, d12, phi, phip);
        find_fp(eam2006dai, d12, fp1);
        fp2 = fp1;
      }
      // TODO: use d12inv
      phip /= d12;
      fp1 /= d12;
      fp2 /= d12;
      double f12x = x12 * (phip + Fp1 * fp2);
      double f12y = y12 * (phip + Fp1 * fp2);
      double f12z = z12 * (phip + Fp1 * fp2);
      double f21x = -x12 * (phip + Fp2 * fp1);
      double f21y = -y12 * (phip + Fp2 * fp1);
      double f21z = -z12 * (phip + Fp2 * fp1);

      // two-body potential energy
      s_pe += phi;

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
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
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;

    // save potential energy
    g_pe[n1] += s_pe;
  }
}

// Force evaluation wrapper
void EAM::compute(
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
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

  if (potential_model == 0) {
    find_force_eam_step1<0><<<grid_size, BLOCK_SIZE_FORCE>>>(
      eam2004zhou, eam2006dai, number_of_atoms, N1, N2, box, neighbor.NN_local.data(),
      neighbor.NL_local.data(), type.data(), position_per_atom.data(),
      position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
      eam_data.Fp.data(), potential_per_atom.data());
    CUDA_CHECK_KERNEL

    find_force_eam_step2<0><<<grid_size, BLOCK_SIZE_FORCE>>>(
      eam2004zhou, eam2006dai, number_of_atoms, N1, N2, box, neighbor.NN_local.data(),
      neighbor.NL_local.data(), type.data(), eam_data.Fp.data(), position_per_atom.data(),
      position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
      force_per_atom.data(), force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms, virial_per_atom.data(),
      potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }

  if (potential_model == 1) {
    find_force_eam_step1<1><<<grid_size, BLOCK_SIZE_FORCE>>>(
      eam2004zhou, eam2006dai, number_of_atoms, N1, N2, box, neighbor.NN_local.data(),
      neighbor.NL_local.data(), type.data(), position_per_atom.data(),
      position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
      eam_data.Fp.data(), potential_per_atom.data());
    CUDA_CHECK_KERNEL

    find_force_eam_step2<1><<<grid_size, BLOCK_SIZE_FORCE>>>(
      eam2004zhou, eam2006dai, number_of_atoms, N1, N2, box, neighbor.NN_local.data(),
      neighbor.NL_local.data(), type.data(), eam_data.Fp.data(), position_per_atom.data(),
      position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
      force_per_atom.data(), force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms, virial_per_atom.data(),
      potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }
}
