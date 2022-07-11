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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep3.cuh"
#include "nep3_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include <string>
#include <vector>

const int NUM_ELEMENTS = 103;
const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};

NEP3::NEP3(
  FILE* fid, char* input_dir, int num_types, int version, bool enable_zbl, const int num_atoms)
{
  paramb.version = version;
  if (paramb.version == 2) {
    if (num_types == 1) {
      printf("Use the NEP2 potential with %d atom type.\n", num_types);
    } else {
      printf("Use the NEP2 potential with %d atom types.\n", num_types);
    }
  } else {
    if (num_types == 1) {
      printf("Use the NEP3 potential with %d atom type.\n", num_types);
    } else {
      printf("Use the NEP3 potential with %d atom types.\n", num_types);
    }
  }

  char name[20];

  for (int n = 0; n < num_types; ++n) {
    int count = fscanf(fid, "%s", name);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
    std::string element(name);
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (element == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %g).\n", n, name, zbl.atomic_numbers[n]);
  }

  paramb.num_types = num_types;

  if (enable_zbl) {
    int count = fscanf(fid, "%s%f%f", name, &zbl.rc_inner, &zbl.rc_outer);
    PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
    zbl.enabled = true;
    printf(
      "    has ZBL with inner cutoff %g A and outer cutoff %g A.\n", zbl.rc_inner, zbl.rc_outer);
  }

  int count = fscanf(fid, "%s%f%f", name, &paramb.rc_radial, &paramb.rc_angular);
  PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
  printf("    radial cutoff = %g A.\n", paramb.rc_radial);
  printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  count = fscanf(fid, "%s%d%d", name, &paramb.n_max_radial, &paramb.n_max_angular);
  PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  if (paramb.version == 3) {
    count = fscanf(fid, "%s%d%d", name, &paramb.basis_size_radial, &paramb.basis_size_angular);
    PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");
    printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
    printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);
  }

  int L_max_4body = 0;
  int L_max_5body = 0;
  if (paramb.version == 2) {
    count = fscanf(fid, "%s%d", name, &paramb.L_max);
    PRINT_SCANF_ERROR(count, 2, "reading error for NEP potential.");
    printf("    l_max_3body = %d.\n", paramb.L_max);
  } else {
    count = fscanf(fid, "%s%d%d%d", name, &paramb.L_max, &L_max_4body, &L_max_5body);
    PRINT_SCANF_ERROR(count, 4, "reading error for NEP potential.");
    printf("    l_max_3body = %d.\n", paramb.L_max);
    printf("    l_max_4body = %d.\n", L_max_4body);
    printf("    l_max_5body = %d.\n", L_max_5body);
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
  count = fscanf(fid, "%s%d%d", name, &annmb.num_neurons1, &num_neurons2);
  PRINT_SCANF_ERROR(count, 3, "reading error for NEP potential.");

  rc = paramb.rc_radial; // largest cutoff

  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  annmb.dim = (paramb.n_max_radial + 1) + (paramb.n_max_angular + 1) * paramb.num_L;

  printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  annmb.num_para = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  printf("    number of neural network parameters = %d.\n", annmb.num_para);
  int num_para_descriptor = paramb.num_types * paramb.num_types *
                            ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                             (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (paramb.version == 2) {
    num_para_descriptor =
      (paramb.num_types == 1)
        ? 0
        : paramb.num_types * paramb.num_types * (paramb.n_max_radial + paramb.n_max_angular + 2);
  }
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para += num_para_descriptor;
  printf("    total number of parameters = %d\n", annmb.num_para);

  paramb.num_types_sq = paramb.num_types * paramb.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  int radial_neighbor_size = 200; // TODO: check this in NEP training
  int angular_neighbor_size = 50; // TODO: check this in NEP training
  nep_data.f12x.resize(num_atoms * angular_neighbor_size);
  nep_data.f12y.resize(num_atoms * angular_neighbor_size);
  nep_data.f12z.resize(num_atoms * angular_neighbor_size);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * radial_neighbor_size);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * angular_neighbor_size);
  nep_data.Fp.resize(num_atoms * annmb.dim);
  nep_data.sum_fxyz.resize(num_atoms * (paramb.n_max_angular + 1) * NUM_OF_ABC);
  nep_data.parameters.resize(annmb.num_para);
  cell_count.resize(num_atoms);
  cell_count_sum.resize(num_atoms);
  cell_contents.resize(num_atoms);

  update_potential(fid);
}

NEP3::~NEP3(void)
{
  // nothing
}

void NEP3::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  ann.b1 = ann.w1 + ann.num_neurons1;
  ann.c = ann.b1 + 1;
}

void NEP3::update_potential(FILE* fid)
{
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    int count = fscanf(fid, "%f", &parameters[n]);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
  }
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);

  for (int d = 0; d < annmb.dim; ++d) {
    int count = fscanf(fid, "%f", &paramb.q_scaler[d]);
    PRINT_SCANF_ERROR(count, 1, "reading error for NEP potential.");
  }
}

static __global__ void find_neighbor_list_large_box(
  NEP3::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int count_radial = 0;
  int count_angular = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box, x1, y1, z1, 2.0f * paramb.rcinv_radial, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
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

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n2 < N1 || n2 >= N2 || n1 == n2) {
            continue;
          }

          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
          float d12_square = x12 * x12 + y12 * y12 + z12 * z12;

          if (d12_square >= paramb.rc_radial * paramb.rc_radial) {
            continue;
          }

          g_NL_radial[count_radial++ * N + n1] = n2;

          if (d12_square < paramb.rc_angular * paramb.rc_angular) {
            g_NL_angular[count_angular++ * N + n1] = n2;
          }
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
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

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        apply_mic(box, x12double, y12double, z12double);
        float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
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
          accumulate_s(d12, x12, y12, z12, fn, s);
        } else {
          float fn12[MAX_NUM_N];
          find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          accumulate_s(d12, x12, y12, z12, gn12, s);
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

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0, annmb.b0, annmb.w1, annmb.b1, q, F, Fp);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
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
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];

      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      if (paramb.version == 2) {
        find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float tmp12 = g_Fp[n1 + n * N] * fnp12[n] * d12inv;
          float tmp21 = g_Fp[n2 + n * N] * fnp12[n] * d12inv;
          tmp12 *= (paramb.num_types == 1)
                     ? 1.0f
                     : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          tmp21 *= (paramb.num_types == 1)
                     ? 1.0f
                     : annmb.c[(n * paramb.num_types + t2) * paramb.num_types + t1];
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
            f21[d] -= tmp21 * r12[d];
          }
        }
      } else {
        find_fn_and_fnp(
          paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gnp12 = 0.0f;
          float gnp21 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
            gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
          }
          float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
          float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
            f21[d] -= tmp21 * r12[d];
          }
        }
      }
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
  }
}

static __global__ void find_partial_force_angular(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  double* g_f12x,
  double* g_f12y,
  double* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
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
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP3::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
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
  if (n1 < N2) {
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
    int type1 = g_type[n1];
    float zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      float zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
#ifdef USE_JESPER_HEA
      find_f_and_fp_zbl(type1, type2, zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
#else
      find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
#endif
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
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
    g_pe[n1] += s_pe;
  }
}

// large box fo MD applications
void NEP3::compute_large_box(
  const int type_shift,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const double rc_cell_list = 0.5 * rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(rc_cell_list, num_bins, box, position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, N, N1, N2, num_bins[0], num_bins[1], num_bins[2], box, cell_count.data(),
    cell_count_sum.data(), cell_contents.data(), position_per_atom.data(),
    position_per_atom.data() + N, position_per_atom.data() + N * 2, nep_data.NN_radial.data(),
    nep_data.NL_radial.data(), nep_data.NN_angular.data(), nep_data.NL_angular.data());
  CUDA_CHECK_KERNEL

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    paramb, annmb, N, N1, N2, box, nep_data.NN_radial.data(), nep_data.NL_radial.data(),
    nep_data.NN_angular.data(), nep_data.NL_angular.data(), type.data(), position_per_atom.data(),
    position_per_atom.data() + N, position_per_atom.data() + N * 2, potential_per_atom.data(),
    nep_data.Fp.data(), nep_data.sum_fxyz.data());
  CUDA_CHECK_KERNEL

  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb, annmb, N, N1, N2, box, nep_data.NN_radial.data(), nep_data.NL_radial.data(),
    type.data(), position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, nep_data.Fp.data(), force_per_atom.data(),
    force_per_atom.data() + N, force_per_atom.data() + N * 2, virial_per_atom.data());
  CUDA_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE>>>(
    paramb, annmb, N, N1, N2, box, nep_data.NN_angular.data(), nep_data.NL_angular.data(),
    type.data(), position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, nep_data.Fp.data(), nep_data.sum_fxyz.data(),
    nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
  CUDA_CHECK_KERNEL
  find_properties_many_body(
    box, nep_data.NN_angular.data(), nep_data.NL_angular.data(), nep_data.f12x.data(),
    nep_data.f12y.data(), nep_data.f12z.data(), position_per_atom, force_per_atom, virial_per_atom);
  CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      N, zbl, N1, N2, box, nep_data.NN_angular.data(), nep_data.NL_angular.data(), type.data(),
      position_per_atom.data(), position_per_atom.data() + N, position_per_atom.data() + N * 2,
      force_per_atom.data(), force_per_atom.data() + N, force_per_atom.data() + N * 2,
      virial_per_atom.data(), potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP3::compute_small_box(
  const int type_shift,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;
  GPU_Vector<int> NN_radial(type.size());
  GPU_Vector<int> NL_radial(size_x12);
  GPU_Vector<int> NN_angular(type.size());
  GPU_Vector<int> NL_angular(size_x12);
  GPU_Vector<float> r12(size_x12 * 6);

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, N, N1, N2, box, ebox, position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, NN_radial.data(), NL_radial.data(), NN_angular.data(),
    NL_angular.data(), r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5);
  CUDA_CHECK_KERNEL

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, annmb, N, N1, N2, NN_radial.data(), NL_radial.data(), NN_angular.data(),
    NL_angular.data(), type.data(), r12.data(), r12.data() + size_x12, r12.data() + size_x12 * 2,
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    potential_per_atom.data(), nep_data.Fp.data(), nep_data.sum_fxyz.data());
  CUDA_CHECK_KERNEL

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, annmb, N, N1, N2, NN_radial.data(), NL_radial.data(), type.data(), r12.data(),
    r12.data() + size_x12, r12.data() + size_x12 * 2, nep_data.Fp.data(), force_per_atom.data(),
    force_per_atom.data() + N, force_per_atom.data() + N * 2, virial_per_atom.data());
  CUDA_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb, annmb, N, N1, N2, NN_angular.data(), NL_angular.data(), type.data(),
    r12.data() + size_x12 * 3, r12.data() + size_x12 * 4, r12.data() + size_x12 * 5,
    nep_data.Fp.data(), nep_data.sum_fxyz.data(), force_per_atom.data(), force_per_atom.data() + N,
    force_per_atom.data() + N * 2, virial_per_atom.data());
  CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      N, zbl, N1, N2, NN_angular.data(), NL_angular.data(), type.data(), r12.data() + size_x12 * 3,
      r12.data() + size_x12 * 4, r12.data() + size_x12 * 5, force_per_atom.data(),
      force_per_atom.data() + N, force_per_atom.data() + N * 2, virial_per_atom.data(),
      potential_per_atom.data());
    CUDA_CHECK_KERNEL
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP3::ExpandedBox& ebox)
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

  if (ebox.num_cells[0] * ebox.num_cells[1] * ebox.num_cells[2] > 1) {
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

void NEP3::compute(
  const int type_shift,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial, box, ebox);

  if (is_small_box) {
    compute_small_box(
      type_shift, box, type, position_per_atom, potential_per_atom, force_per_atom,
      virial_per_atom);
  } else {
    compute_large_box(
      type_shift, box, type, position_per_atom, potential_per_atom, force_per_atom,
      virial_per_atom);
  }
}
