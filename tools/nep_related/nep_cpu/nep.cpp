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
A CPU implementation of the neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const int NUM_OF_ABC = 24; // 3 + 5 + 7 + 9 for L_max = 4
float C3B[NUM_OF_ABC] = {
  0.238732414637843f, 0.119366207318922f, 0.119366207318922f, 0.099471839432435f,
  0.596831036594608f, 0.596831036594608f, 0.149207759148652f, 0.149207759148652f,
  0.139260575205408f, 0.104445431404056f, 0.104445431404056f, 1.044454314040563f,
  1.044454314040563f, 0.174075719006761f, 0.174075719006761f, 0.011190581936149f,
  0.223811638722978f, 0.223811638722978f, 0.111905819361489f, 0.111905819361489f,
  1.566681471060845f, 1.566681471060845f, 0.195835183882606f, 0.195835183882606f};
float C4B[5] = {
  -0.007499480826664f, -0.134990654879954f, 0.067495327439977f, 0.404971964639861f,
  -0.809943929279723f};
float C5B[3] = {0.026596810706114f, 0.053193621412227f, 0.026596810706114f};
#define K_C_SP 14.399645f // 1/(4*PI*epsilon_0)

const int MAX_NUM_N = 20; // n_max+1 = 19+1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;

const int NUM_ELEMENTS = 103;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

NEP3::NEP3(int N)
{
  std::ifstream input_file("nep.txt");

  if (!input_file) {
    std::cout << "Cannot open nep.txt\n";
    exit(1);
  }

  std::string nep_name;
  input_file >> nep_name;
  if (nep_name == "nep") {
    paramb.version = 2;
    zbl.enabled = false;
  } else if (nep_name == "nep_zbl") {
    paramb.version = 2;
    zbl.enabled = true;
  } else if (nep_name == "nep3") {
    paramb.version = 3;
    zbl.enabled = false;
  } else if (nep_name == "nep3_zbl") {
    paramb.version = 3;
    zbl.enabled = true;
  }

  input_file >> paramb.num_types;

  if (paramb.version == 2) {
    std::cout << "Use the NEP2 potential with " << paramb.num_types << " atom type(s).\n";
  } else {
    std::cout << "Use the NEP3 potential with " << paramb.num_types << " atom type(s).\n";
  }

  char name[20];

  for (int n = 0; n < paramb.num_types; ++n) {
    std::string element;
    input_file >> element;
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (element == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    std::cout << "    type " << n << " (" << element << " with Z = " << zbl.atomic_numbers[n]
              << ").\n";
  }

  if (zbl.enabled) {
    input_file >> name >> zbl.rc_inner >> zbl.rc_outer;
    std::cout << "    has ZBL with inner cutoff " << zbl.rc_inner << " A and outer cutoff "
              << zbl.rc_outer << " A.\n";
  }

  input_file >> name >> paramb.rc_radial >> paramb.rc_angular;
  std::cout << "    radial cutoff = " << paramb.rc_radial << " A.\n";
  std::cout << "    angular cutoff = " << paramb.rc_angular << " A.\n";

  input_file >> name >> paramb.n_max_radial >> paramb.n_max_angular;
  std::cout << "    n_max_radial = " << paramb.n_max_radial << ".\n";
  std::cout << "    n_max_angular = " << paramb.n_max_angular << ".\n";

  input_file >> name >> paramb.basis_size_radial >> paramb.basis_size_angular;
  std::cout << "    basis_size_radial = " << paramb.basis_size_radial << ".\n";
  std::cout << "    basis_size_angular = " << paramb.basis_size_angular << ".\n";

  int L_max_4body = 0;
  int L_max_5body = 0;
  if (paramb.version == 2) {
    input_file >> name >> paramb.L_max;
    std::cout << "    l_max_3body = " << paramb.L_max << ".\n";
  } else {
    input_file >> name >> paramb.L_max >> L_max_4body >> L_max_5body;
    std::cout << "    l_max_3body = " << paramb.L_max << ".\n";
    std::cout << "    l_max_4body = " << L_max_4body << ".\n";
    std::cout << "    l_max_5body = " << L_max_5body << ".\n";
  }

  paramb.num_L = paramb.L_max;
  if (paramb.version == 3) {
    if (L_max_4body == 2) {
      paramb.num_L += 1;
    }
    if (L_max_5body == 1) {
      paramb.num_L += 1;
    }
  }
  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  int num_neurons2;
  input_file >> name >> annmb.num_neurons1 >> num_neurons2;

  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  annmb.dim = (paramb.n_max_radial + 1) + (paramb.n_max_angular + 1) * paramb.num_L;

  std::cout << "    ANN = " << annmb.dim << "-" << annmb.num_neurons1 << "-1.\n";

  annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  std::cout << "    number of neural network parameters = " << annmb.num_para << ".\n";
  int num_para_descriptor = paramb.num_types * paramb.num_types *
                            ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                             (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (paramb.version == 2) {
    num_para_descriptor =
      (paramb.num_types == 1)
        ? 0
        : paramb.num_types * paramb.num_types * (paramb.n_max_radial + paramb.n_max_angular + 2);
  }
  std::cout << "    number of descriptor parameters = " << num_para_descriptor << ".\n";
  annmb.num_para += num_para_descriptor;
  std::cout << "    total number of parameters = " << annmb.num_para << ".\n";

  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  Fp.resize(N * annmb.dim);
  sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  parameters.resize(annmb.num_para);

  for (int n = 0; n < annmb.num_para; ++n) {
    input_file >> parameters[n];
  }
  update_potential(parameters.data(), annmb);

  for (int d = 0; d < annmb.dim; ++d) {
    input_file >> paramb.q_scaler[d];
  }

  input_file.close();
}

void NEP3::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  ann.b1 = ann.w1 + ann.num_neurons1;
  ann.c = ann.b1 + 1;
}

static void apply_ann_one_layer(
  const int dim,
  const int num_neurons1,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  float* q,
  float& energy,
  float* energy_derivative)
{
  for (int n = 0; n < num_neurons1; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < dim; ++d) {
      w0_times_q += w0[n * dim + d] * q[d];
    }
    float x1 = tanh(w0_times_q - b0[n]);
    energy += w1[n] * x1;
    for (int d = 0; d < dim; ++d) {
      float y1 = (1.0f - x1 * x1) * w0[n * dim + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[0];
}

static void find_fc(float rc, float rcinv, float d12, float& fc)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(3.1415927f * x) + 0.5f;
  } else {
    fc = 0.0f;
  }
}

static void find_fc_and_fcp(float rc, float rcinv, float d12, float& fc, float& fcp)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(3.1415927f * x) + 0.5f;
    fcp = -1.5707963f * sin(3.1415927f * x);
    fcp *= rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static void find_fc_and_fcp_zbl(float r1, float r2, float d12, float& fc, float& fcp)
{
  if (d12 < r1) {
    fc = 1.0f;
    fcp = 0.0f;
  } else if (d12 < r2) {
    float pi_factor = 3.1415927f / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5f + 0.5f;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5f;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static void find_phi_and_phip_zbl(float a, float b, float x, float& phi, float& phip)
{
  float tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

static void find_f_and_fp_zbl(
  float zizj,
  float a_inv,
  float rc_inner,
  float rc_outer,
  float d12,
  float d12inv,
  float& f,
  float& fp)
{
  float x = d12 * a_inv;
  f = fp = 0.0f;
  float Zbl_para[8] = {0.18175f, 3.1998f, 0.50986f, 0.94229f,
                       0.28022f, 0.4029f, 0.02817f, 0.20162f};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  float fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static void find_fn(const int n, const float rcinv, const float d12, const float fc12, float& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5f * fc12;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0f) * 0.5f * fc12;
  }
}

static void find_fn_and_fnp(
  const int n,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float& fn,
  float& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5f;
    fnp = 2.0f * (d12 * rcinv - 1.0f) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    float u0 = 1.0f;
    float u1 = 2.0f * x;
    float u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0f * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0f) * 0.5f;
    fnp = n * u0 * 2.0f * (d12 * rcinv - 1.0f) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

static void
find_fn(const int n_max, const float rcinv, const float d12, const float fc12, float* fn)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f * fc12;
  }
}

static void find_fn_and_fnp(
  const int n_max,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float* fn,
  float* fnp)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fnp[0] = 0.0f;
  fn[1] = x;
  fnp[1] = 1.0f;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  float u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f;
    fnp[m] *= 2.0f * (d12 * rcinv - 1.0f) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

static void get_f12_1(
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float tmp = s[1] * r12[0];
  tmp += s[2] * r12[1];
  tmp *= 2.0f;
  tmp += s[0] * r12[2];
  tmp *= Fp * fnp * d12inv * 2.0f;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  tmp = Fp * fn * 2.0f;
  f12[0] += tmp * 2.0f * s[1];
  f12[1] += tmp * 2.0f * s[2];
  f12[2] += tmp * s[0];
}

static void get_f12_2(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float tmp = s[1] * r12[0] * r12[2];                // Re[Y21]
  tmp += s[2] * r12[1] * r12[2];                     // Im[Y21]
  tmp += s[3] * (r12[0] * r12[0] - r12[1] * r12[1]); // Re[Y22]
  tmp += s[4] * 2.0f * r12[0] * r12[1];              // Im[Y22]
  tmp *= 2.0f;
  tmp += s[0] * (3.0f * r12[2] * r12[2] - d12 * d12); // Y20
  tmp *= Fp * fnp * d12inv * 2.0f;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  tmp = Fp * fn * 4.0f;
  f12[0] += tmp * (-s[0] * r12[0] + s[1] * r12[2] + 2.0f * s[3] * r12[0] + 2.0f * s[4] * r12[1]);
  f12[1] += tmp * (-s[0] * r12[1] + s[2] * r12[2] - 2.0f * s[3] * r12[1] + 2.0f * s[4] * r12[0]);
  f12[2] += tmp * (2.0f * s[0] * r12[2] + s[1] * r12[0] + s[2] * r12[1]);
}

static void get_f12_4body(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float fn_factor = Fp * fn;
  float fnp_factor = Fp * fnp * d12inv;
  float y20 = (3.0f * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  float tmp0 = C4B[0] * 3.0f * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
               C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  float tmp1 = tmp0 * y20 * fnp_factor;
  float tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0f * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0f * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0f * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * 2.0f - C4B[3] * s[3] * s[1] * 2.0f + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * 2.0f + C4B[3] * s[3] * s[2] * 2.0f + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * 2.0f + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0f * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0f * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * 2.0f + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (2.0f * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0f * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0f * r12[0];
  f12[2] += tmp1 * r12[2];
}

static void get_f12_5body(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float fn_factor = Fp * fn;
  float fnp_factor = Fp * fnp * d12inv;
  float s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  float tmp0 = C5B[0] * 4.0f * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * 2.0f * s[0];
  float tmp1 = tmp0 * r12[2] * fnp_factor;
  float tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s[1]
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * 2.0f + C5B[2] * s1_sq_plus_s2_sq * s[1] * 4.0f;
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * 2.0f + C5B[2] * s1_sq_plus_s2_sq * s[2] * 4.0f;
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

static void get_f12_3(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float d12sq = d12 * d12;
  float x2 = r12[0] * r12[0];
  float y2 = r12[1] * r12[1];
  float z2 = r12[2] * r12[2];
  float xy = r12[0] * r12[1];
  float xz = r12[0] * r12[2];
  float yz = r12[1] * r12[2];

  float tmp = s[1] * (5.0f * z2 - d12sq) * r12[0];
  tmp += s[2] * (5.0f * z2 - d12sq) * r12[1];
  tmp += s[3] * (x2 - y2) * r12[2];
  tmp += s[4] * 2.0f * xy * r12[2];
  tmp += s[5] * r12[0] * (x2 - 3.0f * y2);
  tmp += s[6] * r12[1] * (3.0f * x2 - y2);
  tmp *= 2.0f;
  tmp += s[0] * (5.0f * z2 - 3.0f * d12sq) * r12[2];
  tmp *= Fp * fnp * d12inv * 2.0f;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }

  // x
  tmp = s[1] * (4.0f * z2 - 3.0f * x2 - y2);
  tmp += s[2] * (-2.0f * xy);
  tmp += s[3] * 2.0f * xz;
  tmp += s[4] * (2.0f * yz);
  tmp += s[5] * (3.0f * (x2 - y2));
  tmp += s[6] * (6.0f * xy);
  tmp *= 2.0f;
  tmp += s[0] * (-6.0f * xz);
  f12[0] += tmp * Fp * fn * 2.0f;
  // y
  tmp = s[1] * (-2.0f * xy);
  tmp += s[2] * (4.0f * z2 - 3.0f * y2 - x2);
  tmp += s[3] * (-2.0f * yz);
  tmp += s[4] * (2.0f * xz);
  tmp += s[5] * (-6.0f * xy);
  tmp += s[6] * (3.0f * (x2 - y2));
  tmp *= 2.0f;
  tmp += s[0] * (-6.0f * yz);
  f12[1] += tmp * Fp * fn * 2.0f;
  // z
  tmp = s[1] * (8.0f * xz);
  tmp += s[2] * (8.0f * yz);
  tmp += s[3] * (x2 - y2);
  tmp += s[4] * (2.0f * xy);
  tmp *= 2.0f;
  tmp += s[0] * (9.0f * z2 - 3.0f * d12sq);
  f12[2] += tmp * Fp * fn * 2.0f;
}

static void get_f12_4(
  const float x,
  const float y,
  const float z,
  const float r,
  const float rinv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  float* f12)
{
  const float r2 = r * r;
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float xyz = x * yz;
  const float x2my2 = x2 - y2;

  float tmp = s[1] * (7.0f * z2 - 3.0f * r2) * xz; // Y41_real
  tmp += s[2] * (7.0f * z2 - 3.0f * r2) * yz;      // Y41_imag
  tmp += s[3] * (7.0f * z2 - r2) * x2my2;          // Y42_real
  tmp += s[4] * (7.0f * z2 - r2) * 2.0f * xy;      // Y42_imag
  tmp += s[5] * (x2 - 3.0f * y2) * xz;             // Y43_real
  tmp += s[6] * (3.0f * x2 - y2) * yz;             // Y43_imag
  tmp += s[7] * (x2my2 * x2my2 - 4.0f * x2 * y2);  // Y44_real
  tmp += s[8] * (4.0f * xy * x2my2);               // Y44_imag
  tmp *= 2.0f;
  tmp += s[0] * ((35.0f * z2 - 30.0f * r2) * z2 + 3.0f * r2 * r2); // Y40
  tmp *= Fp * fnp * rinv * 2.0f;
  f12[0] += tmp * x;
  f12[1] += tmp * y;
  f12[2] += tmp * z;

  // x
  tmp = s[1] * z * (7.0f * z2 - 3.0f * r2 - 6.0f * x2);  // Y41_real
  tmp += s[2] * (-6.0f * xyz);                           // Y41_imag
  tmp += s[3] * 4.0f * x * (3.0f * z2 - x2);             // Y42_real
  tmp += s[4] * 2.0f * y * (7.0f * z2 - r2 - 2.0f * x2); // Y42_imag
  tmp += s[5] * 3.0f * z * x2my2;                        // Y43_real
  tmp += s[6] * 6.0f * xyz;                              // Y43_imag
  tmp += s[7] * 4.0f * x * (x2 - 3.0f * y2);             // Y44_real
  tmp += s[8] * 4.0f * y * (3.0f * x2 - y2);             // Y44_imag
  tmp *= 2.0f;
  tmp += s[0] * 12.0f * x * (r2 - 5.0f * z2); // Y40
  f12[0] += tmp * Fp * fn * 2.0f;
  // y
  tmp = s[1] * (-6.0f * xyz);                            // Y41_real
  tmp += s[2] * z * (7.0f * z2 - 3.0f * r2 - 6.0f * y2); // Y41_imag
  tmp += s[3] * 4.0f * y * (y2 - 3.0f * z2);             // Y42_real
  tmp += s[4] * 2.0f * x * (7.0f * z2 - r2 - 2.0f * y2); // Y42_imag
  tmp += s[5] * (-6.0f * xyz);                           // Y43_real
  tmp += s[6] * 3.0f * z * x2my2;                        // Y43_imag
  tmp += s[7] * 4.0f * y * (y2 - 3.0f * x2);             // Y44_real
  tmp += s[8] * 4.0f * x * (x2 - 3.0f * y2);             // Y44_imag
  tmp *= 2.0f;
  tmp += s[0] * 12.0f * y * (r2 - 5.0f * z2); // Y40
  f12[1] += tmp * Fp * fn * 2.0f;
  // z
  tmp = s[1] * 3.0f * x * (5.0f * z2 - r2);  // Y41_real
  tmp += s[2] * 3.0f * y * (5.0f * z2 - r2); // Y41_imag
  tmp += s[3] * 12.0f * z * x2my2;           // Y42_real
  tmp += s[4] * 24.0f * xyz;                 // Y42_imag
  tmp += s[5] * x * (x2 - 3.0f * y2);        // Y43_real
  tmp += s[6] * y * (3.0f * x2 - y2);        // Y43_imag
  tmp *= 2.0f;
  tmp += s[0] * 16.0f * z * (5.0f * z2 - 3.0f * r2); // Y40
  f12[2] += tmp * Fp * fn * 2.0f;
}

static void accumulate_f12(
  const int n,
  const int n_max_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* Fp,
  const float* sum_fxyz,
  float* f12)
{
  const float d12inv = 1.0f / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0], sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * C3B[3], sum_fxyz[n * NUM_OF_ABC + 4] * C3B[4],
    sum_fxyz[n * NUM_OF_ABC + 5] * C3B[5], sum_fxyz[n * NUM_OF_ABC + 6] * C3B[6],
    sum_fxyz[n * NUM_OF_ABC + 7] * C3B[7]};
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s3[7] = {sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],   sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
                 sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10], sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
                 sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12], sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
                 sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s4[9] = {sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15], sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
                 sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17], sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
                 sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19], sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
                 sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21], sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
                 sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

static void accumulate_f12_with_4body(
  const int n,
  const int n_max_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* Fp,
  const float* sum_fxyz,
  float* f12)
{
  const float d12inv = 1.0f / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0], sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};
  get_f12_4body(d12, d12inv, fn, fnp, Fp[4 * n_max_angular_plus_1 + n], s2, r12, f12);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s3[7] = {sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],   sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
                 sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10], sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
                 sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12], sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
                 sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s4[9] = {sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15], sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
                 sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17], sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
                 sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19], sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
                 sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21], sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
                 sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

static void accumulate_f12_with_5body(
  const int n,
  const int n_max_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* Fp,
  const float* sum_fxyz,
  float* f12)
{
  const float d12inv = 1.0f / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  get_f12_5body(d12, d12inv, fn, fnp, Fp[5 * n_max_angular_plus_1 + n], s1, r12, f12);
  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3], sum_fxyz[n * NUM_OF_ABC + 4], sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6], sum_fxyz[n * NUM_OF_ABC + 7]};
  get_f12_4body(d12, d12inv, fn, fnp, Fp[4 * n_max_angular_plus_1 + n], s2, r12, f12);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s3[7] = {sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],   sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
                 sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10], sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
                 sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12], sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
                 sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s4[9] = {sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15], sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
                 sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17], sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
                 sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19], sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
                 sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21], sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
                 sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

static void accumulate_s(const float d12, float x12, float y12, float z12, const float fn, float* s)
{
  float d12inv = 1.0f / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  float x12sq = x12 * x12;
  float y12sq = y12 * y12;
  float z12sq = z12 * z12;
  float x12sq_minus_y12sq = x12sq - y12sq;
  s[0] += z12 * fn;                                                             // Y10
  s[1] += x12 * fn;                                                             // Y11_real
  s[2] += y12 * fn;                                                             // Y11_imag
  s[3] += (3.0f * z12sq - 1.0f) * fn;                                           // Y20
  s[4] += x12 * z12 * fn;                                                       // Y21_real
  s[5] += y12 * z12 * fn;                                                       // Y21_imag
  s[6] += x12sq_minus_y12sq * fn;                                               // Y22_real
  s[7] += 2.0f * x12 * y12 * fn;                                                // Y22_imag
  s[8] += (5.0f * z12sq - 3.0f) * z12 * fn;                                     // Y30
  s[9] += (5.0f * z12sq - 1.0f) * x12 * fn;                                     // Y31_real
  s[10] += (5.0f * z12sq - 1.0f) * y12 * fn;                                    // Y31_imag
  s[11] += x12sq_minus_y12sq * z12 * fn;                                        // Y32_real
  s[12] += 2.0f * x12 * y12 * z12 * fn;                                         // Y32_imag
  s[13] += (x12 * x12 - 3.0f * y12 * y12) * x12 * fn;                           // Y33_real
  s[14] += (3.0f * x12 * x12 - y12 * y12) * y12 * fn;                           // Y33_imag
  s[15] += ((35.0f * z12sq - 30.0f) * z12sq + 3.0f) * fn;                       // Y40
  s[16] += (7.0f * z12sq - 3.0f) * x12 * z12 * fn;                              // Y41_real
  s[17] += (7.0f * z12sq - 3.0f) * y12 * z12 * fn;                              // Y41_iamg
  s[18] += (7.0f * z12sq - 1.0f) * x12sq_minus_y12sq * fn;                      // Y42_real
  s[19] += (7.0f * z12sq - 1.0f) * x12 * y12 * 2.0f * fn;                       // Y42_imag
  s[20] += (x12sq - 3.0f * y12sq) * x12 * z12 * fn;                             // Y43_real
  s[21] += (3.0f * x12sq - y12sq) * y12 * z12 * fn;                             // Y43_imag
  s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0f * x12sq * y12sq) * fn; // Y44_real
  s[23] += (4.0f * x12 * y12 * x12sq_minus_y12sq) * fn;                         // Y44_imag
}

static void find_q(const int n_max_angular_plus_1, const int n, const float* s, float* q)
{
  q[n] = C3B[0] * s[0] * s[0] + 2.0f * (C3B[1] * s[1] * s[1] + C3B[2] * s[2] * s[2]);
  q[n_max_angular_plus_1 + n] =
    C3B[3] * s[3] * s[3] + 2.0f * (C3B[4] * s[4] * s[4] + C3B[5] * s[5] * s[5] +
                                   C3B[6] * s[6] * s[6] + C3B[7] * s[7] * s[7]);
  q[2 * n_max_angular_plus_1 + n] =
    C3B[8] * s[8] * s[8] +
    2.0f * (C3B[9] * s[9] * s[9] + C3B[10] * s[10] * s[10] + C3B[11] * s[11] * s[11] +
            C3B[12] * s[12] * s[12] + C3B[13] * s[13] * s[13] + C3B[14] * s[14] * s[14]);
  q[3 * n_max_angular_plus_1 + n] =
    C3B[15] * s[15] * s[15] +
    2.0f * (C3B[16] * s[16] * s[16] + C3B[17] * s[17] * s[17] + C3B[18] * s[18] * s[18] +
            C3B[19] * s[19] * s[19] + C3B[20] * s[20] * s[20] + C3B[21] * s[21] * s[21] +
            C3B[22] * s[22] * s[22] + C3B[23] * s[23] * s[23]);
}

static void find_q_with_4body(const int n_max_angular_plus_1, const int n, const float* s, float* q)
{
  find_q(n_max_angular_plus_1, n, s, q);
  q[4 * n_max_angular_plus_1 + n] =
    C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
    C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
    C4B[4] * s[4] * s[5] * s[7];
}

static void find_q_with_5body(const int n_max_angular_plus_1, const int n, const float* s, float* q)
{
  find_q_with_4body(n_max_angular_plus_1, n, s, q);
  float s0_sq = s[0] * s[0];
  float s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
  q[5 * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                    C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
}

static void find_descriptor_small_box(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const float* g_x12_radial,
  const float* g_y12_radial,
  const float* g_z12_radial,
  const float* g_x12_angular,
  const float* g_y12_angular,
  const float* g_z12_angular,
  double* g_pe,
  float* g_Fp,
  float* g_sum_fxyz)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      int t2 = g_type[n2];
      float fn12[MAX_NUM_N];
      if (paramb.version == 2) {
        find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float c = (paramb.num_types == 1)
                      ? 1.0f
                      : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          q[n] += fn12[n] * c;
        }
      } else {
        find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          q[n] += gn12;
        }
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        int t2 = g_type[n2];
        if (paramb.version == 2) {
          float fn;
          find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
          fn *=
            (paramb.num_types == 1)
              ? 1.0f
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          accumulate_s(d12, r12[0], r12[1], r12[2], fn, s);
        } else {
          float fn12[MAX_NUM_N];
          find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
        }
      }
      if (paramb.num_L == paramb.L_max) {
        find_q(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else if (paramb.num_L == paramb.L_max + 1) {
        find_q_with_4body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else {
        find_q_with_5body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0, annmb.b0, annmb.w1, annmb.b1, q, F, Fp);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

static void find_force_radial_small_box(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];

      float f12[3] = {0.0f};
      if (paramb.version == 2) {
        find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float tmp12 = g_Fp[n1 + n * N] * fnp12[n] * d12inv;
          tmp12 *= (paramb.num_types == 1)
                     ? 1.0f
                     : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        }
      } else {
        find_fn_and_fnp(
          paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gnp12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gnp12 += fnp12[k] * annmb.c[c_index];
          }
          float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        }
      }

      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];
      g_virial[n1 + 0 * N] -= r12[0] * f12[0];
      g_virial[n1 + 1 * N] -= r12[0] * f12[1];
      g_virial[n1 + 2 * N] -= r12[0] * f12[2];
      g_virial[n1 + 3 * N] -= r12[1] * f12[0];
      g_virial[n1 + 4 * N] -= r12[1] * f12[1];
      g_virial[n1 + 5 * N] -= r12[1] * f12[2];
      g_virial[n1 + 6 * N] -= r12[2] * f12[0];
      g_virial[n1 + 7 * N] -= r12[2] * f12[1];
      g_virial[n1 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

static void find_force_angular_small_box(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_Fp,
  const float* g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  for (int n1 = 0; n1 < N; ++n1) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float f12[3] = {0.0f};

      if (paramb.version == 2) {
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          float fn;
          float fnp;
          find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);
          const float c =
            (paramb.num_types == 1)
              ? 1.0f
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          fn *= c;
          fnp *= c;
          accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, fn, fnp, Fp, sum_fxyz, f12);
        }
      } else {
        float fn12[MAX_NUM_N];
        float fnp12[MAX_NUM_N];
        find_fn_and_fnp(
          paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          float gn12 = 0.0f;
          float gnp12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
            gnp12 += fnp12[k] * annmb.c[c_index];
          }
          if (paramb.num_L == paramb.L_max) {
            accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else if (paramb.num_L == paramb.L_max + 1) {
            accumulate_f12_with_4body(
              n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          } else {
            accumulate_f12_with_5body(
              n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
          }
        }
      }
      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];
      g_virial[n1 + 0 * N] -= r12[0] * f12[0];
      g_virial[n1 + 1 * N] -= r12[0] * f12[1];
      g_virial[n1 + 2 * N] -= r12[0] * f12[2];
      g_virial[n1 + 3 * N] -= r12[1] * f12[0];
      g_virial[n1 + 4 * N] -= r12[1] * f12[1];
      g_virial[n1 + 5 * N] -= r12[1] * f12[2];
      g_virial[n1 + 6 * N] -= r12[2] * f12[0];
      g_virial[n1 + 7 * N] -= r12[2] * f12[1];
      g_virial[n1 + 8 * N] -= r12[2] * f12[2];
    }
  }
}

static void find_force_ZBL_small_box(
  const int N,
  const NEP3::ZBL zbl,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  for (int n1 = 0; n1 < N; ++n1) {
    float zi = zbl.atomic_numbers[g_type[n1]];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      float zj = zbl.atomic_numbers[g_type[n2]];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      g_fx[n1] += f12[0];
      g_fy[n1] += f12[1];
      g_fz[n1] += f12[2];
      g_fx[n2] -= f12[0];
      g_fy[n2] -= f12[1];
      g_fz[n2] -= f12[2];
      g_virial[n1 + 0 * N] -= r12[0] * f12[0];
      g_virial[n1 + 1 * N] -= r12[0] * f12[1];
      g_virial[n1 + 2 * N] -= r12[0] * f12[2];
      g_virial[n1 + 3 * N] -= r12[1] * f12[0];
      g_virial[n1 + 4 * N] -= r12[1] * f12[1];
      g_virial[n1 + 5 * N] -= r12[1] * f12[2];
      g_virial[n1 + 6 * N] -= r12[2] * f12[0];
      g_virial[n1 + 7 * N] -= r12[2] * f12[1];
      g_virial[n1 + 8 * N] -= r12[2] * f12[2];
      g_pe[n1] += f * 0.5f;
    }
  }
}

void NEP3::compute(
  const std::vector<int>& NN_radial,
  const std::vector<int>& NL_radial,
  const std::vector<int>& NN_angular,
  const std::vector<int>& NL_angular,
  const std::vector<int>& type,
  const std::vector<float>& r12,
  std::vector<double>& potential_per_atom,
  std::vector<double>& force_per_atom,
  std::vector<double>& virial_per_atom)
{
  const int N = NN_radial.size();
  const int size_x12 = NL_radial.size();

  for (int n = 0; n < potential_per_atom.size(); ++n) {
    potential_per_atom[n] = 0.0;
  }
  for (int n = 0; n < force_per_atom.size(); ++n) {
    force_per_atom[n] = 0.0;
  }
  for (int n = 0; n < virial_per_atom.size(); ++n) {
    virial_per_atom[n] = 0.0;
  }

  find_descriptor_small_box(
    paramb, annmb, N, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(),
    type.data(), r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    potential_per_atom.data(), Fp.data(), sum_fxyz.data());

  find_force_radial_small_box(
    paramb, annmb, N, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, Fp.data(), force_per_atom.data(),
    force_per_atom.data() + N, force_per_atom.data() + N * 2, virial_per_atom.data());

  find_force_angular_small_box(
    paramb, annmb, N, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
    r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, Fp.data(), sum_fxyz.data(),
    force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + N * 2,
    virial_per_atom.data());

  if (zbl.enabled) {
    find_force_ZBL_small_box(
      N, zbl, NN_angular.data(), NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12,
      r12.data() + size_x12 * 2, force_per_atom.data(), force_per_atom.data() + N,
      force_per_atom.data() + N * 2, virial_per_atom.data(), potential_per_atom.data());
  }
}
