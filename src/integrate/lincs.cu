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
The LINCS (linear constraint solver) bond-length constraint method:
Hess et al., J. Comput. Chem. 18, 1463 (1997).

For each constraint k between atoms (i, j) with target distance d_k,
sigma_k(r) = |r_i - r_j|^2 - d_k^2 = 0.
With B being the constraint gradient matrix (row k has +2*b_k at atom i and
-2*b_k at atom j, where b_k is the bond vector at the constrained positions)
and T = B * M^{-1} * B^T, the correction of the unconstrained positions r'
reads r'' = r' - M^{-1} * B^T * lambda, where T * lambda = sigma(r').
T is sparse: only constraints sharing an atom are coupled. The linear system
is solved by a truncated Neumann series (the LINCS expansion), and the whole
procedure is re-linearized at the corrected positions num_iterations times.
Velocities are corrected consistently: by delta_r / dt after the position
update, and by projecting out the components along the bonds after the
second velocity half step.
------------------------------------------------------------------------------*/

#include "lincs.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace
{

// element symbols up to Cm (Z = 96)
const char* ELEMENT_SYMBOLS[] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
  "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
  "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
  "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
  "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
  "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U",  "Np", "Pu", "Am", "Cm"};

// covalent radii in Angstrom, Cordero et al., Dalton Trans., 2832 (2008)
const double COVALENT_RADII[] = {
  0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
  1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
  1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22,
  1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
  1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
  1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
  1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
  1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
  1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
  2.00, 1.96, 1.90, 1.87, 1.80, 1.69};

int get_atomic_number(const std::string& symbol)
{
  for (int n = 0; n < 96; ++n) {
    if (symbol == ELEMENT_SYMBOLS[n]) {
      return n + 1;
    }
  }
  return 0;
}

} // namespace

void Lincs::parse(const char** param, int num_param)
{
  if (num_param < 2) {
    PRINT_INPUT_ERROR("lincs should have at least 1 parameter.");
  }

  int i = 2;
  if (strcmp(param[1], "H") == 0) {
    mode = 1;
    enabled = true;
    printf("Use LINCS bond-length constraints.\n");
    printf("    constrain bonds involving H atoms.\n");
  } else if (strcmp(param[1], "all") == 0) {
    mode = 2;
    enabled = true;
    printf("Use LINCS bond-length constraints.\n");
    printf("    constrain all detected bonds.\n");
  } else if (strcmp(param[1], "file") == 0) {
    mode = 3;
    enabled = true;
    if (num_param < 3) {
      PRINT_INPUT_ERROR("lincs file needs a file name.");
    }
    filename = param[2];
    i = 3;
    printf("Use LINCS bond-length constraints.\n");
    printf("    read constraints from file %s.\n", filename.c_str());
  } else if (strcmp(param[1], "off") == 0) {
    enabled = false;
    return;
  } else {
    PRINT_INPUT_ERROR("lincs mode should be H, all, file, or off.");
  }

  // optional key-value pairs
  while (i < num_param) {
    if (i + 1 >= num_param) {
      PRINT_INPUT_ERROR("Missing value for lincs option.");
    }
    if (strcmp(param[i], "order") == 0) {
      if (!is_valid_int(param[i + 1], &expansion_order)) {
        PRINT_INPUT_ERROR("lincs order should be an integer.");
      }
      if (expansion_order < 1) {
        PRINT_INPUT_ERROR("lincs order should >= 1.");
      }
    } else if (strcmp(param[i], "iter") == 0) {
      if (!is_valid_int(param[i + 1], &num_iterations)) {
        PRINT_INPUT_ERROR("lincs iter should be an integer.");
      }
      if (num_iterations < 0) {
        PRINT_INPUT_ERROR("lincs iter should >= 0.");
      }
    } else if (strcmp(param[i], "factor") == 0) {
      if (!is_valid_real(param[i + 1], &cutoff_factor)) {
        PRINT_INPUT_ERROR("lincs factor should be a number.");
      }
      if (cutoff_factor <= 1.0) {
        PRINT_INPUT_ERROR("lincs factor should > 1.");
      }
    } else {
      PRINT_INPUT_ERROR("Unknown lincs option.");
    }
    i += 2;
  }

  printf("    expansion order = %d, number of iterations = %d.\n", expansion_order, num_iterations);
  if (mode != 3) {
    printf("    bond detection factor = %g.\n", cutoff_factor);
  }
}

// bond vectors at the current positions (with minimum image convention)
static __global__ void gpu_compute_bond_vectors(
  const int number_of_constraints,
  const Box box,
  const int* __restrict__ atom_i,
  const int* __restrict__ atom_j,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  double* __restrict__ bond_vec)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int i = atom_i[k];
    const int j = atom_j[k];
    double dx = x[i] - x[j];
    double dy = y[i] - y[j];
    double dz = z[i] - z[j];
    apply_mic(box, dx, dy, dz);
    const int K = number_of_constraints;
    bond_vec[k] = dx;
    bond_vec[K + k] = dy;
    bond_vec[2 * K + k] = dz;
  }
}

// right-hand side for positions: w_k = |r_ij|^2 - d_k^2 at the current positions
static __global__ void gpu_compute_rhs(
  const int number_of_constraints,
  const Box box,
  const int* __restrict__ atom_i,
  const int* __restrict__ atom_j,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const double* __restrict__ target,
  double* __restrict__ rhs)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int i = atom_i[k];
    const int j = atom_j[k];
    double dx = x[i] - x[j];
    double dy = y[i] - y[j];
    double dz = z[i] - z[j];
    apply_mic(box, dx, dy, dz);
    rhs[k] = dx * dx + dy * dy + dz * dz - target[k] * target[k];
  }
}

// inverse of the diagonal of T = B * M^{-1} * B^T:
// T_kk = 4 * |b_k|^2 * (1/m_i + 1/m_j)
static __global__ void gpu_compute_diag_inv(
  const int number_of_constraints,
  const int* __restrict__ atom_i,
  const int* __restrict__ atom_j,
  const double* __restrict__ mass,
  const double* __restrict__ bond_vec,
  double* __restrict__ diag_inv)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int K = number_of_constraints;
    const double bx = bond_vec[k];
    const double by = bond_vec[K + k];
    const double bz = bond_vec[2 * K + k];
    const double mass_inv = 1.0 / mass[atom_i[k]] + 1.0 / mass[atom_j[k]];
    diag_inv[k] = 1.0 / (4.0 * (bx * bx + by * by + bz * bz) * mass_inv);
  }
}

// sparse matrix-vector product with N = D^{-1} * (D - T) (zero diagonal):
// (N * in)_k = -diag_inv[k] * sum_l T_kl * in_l,
// T_kl = sign * 4 * (b_k . b_l) / m_shared
static __global__ void gpu_neumann_matvec(
  const int number_of_constraints,
  const double* __restrict__ bond_vec,
  const int* __restrict__ coupl_offset,
  const int* __restrict__ coupl_index,
  const int* __restrict__ coupl_atom,
  const double* __restrict__ coupl_sign,
  const double* __restrict__ mass,
  const double* __restrict__ diag_inv,
  const double* __restrict__ in,
  double* __restrict__ out)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int K = number_of_constraints;
    const double bx = bond_vec[k];
    const double by = bond_vec[K + k];
    const double bz = bond_vec[2 * K + k];
    double sum = 0.0;
    for (int p = coupl_offset[k]; p < coupl_offset[k + 1]; ++p) {
      const int l = coupl_index[p];
      const double t_kl = coupl_sign[p] * 4.0 *
        (bx * bond_vec[l] + by * bond_vec[K + l] + bz * bond_vec[2 * K + l]) /
        mass[coupl_atom[p]];
      sum -= t_kl * in[l];
    }
    out[k] = diag_inv[k] * sum;
  }
}

// lambda = D^{-1} * rhs; series = lambda
static __global__ void gpu_init_lambda(
  const int number_of_constraints,
  const double* __restrict__ diag_inv,
  const double* __restrict__ rhs,
  double* __restrict__ lagrange,
  double* __restrict__ series)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const double value = diag_inv[k] * rhs[k];
    lagrange[k] = value;
    series[k] = value;
  }
}

static __global__ void gpu_accumulate(
  const int number_of_constraints,
  const double* __restrict__ in,
  double* __restrict__ lagrange)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    lagrange[k] += in[k];
  }
}

// r_i -= 2 * lambda_k * b_k / m_i; r_j += 2 * lambda_k * b_k / m_j
// velocities are corrected consistently by delta_r / time_step
static __global__ void gpu_apply_position_correction(
  const int number_of_constraints,
  const double inverse_time_step,
  const int* __restrict__ atom_i,
  const int* __restrict__ atom_j,
  const double* __restrict__ mass,
  const double* __restrict__ bond_vec,
  const double* __restrict__ lagrange,
  double* __restrict__ x,
  double* __restrict__ y,
  double* __restrict__ z,
  double* __restrict__ vx,
  double* __restrict__ vy,
  double* __restrict__ vz)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int K = number_of_constraints;
    const int i = atom_i[k];
    const int j = atom_j[k];
    const double cx = 2.0 * lagrange[k] * bond_vec[k];
    const double cy = 2.0 * lagrange[k] * bond_vec[K + k];
    const double cz = 2.0 * lagrange[k] * bond_vec[2 * K + k];
    const double mass_i_inv = 1.0 / mass[i];
    const double mass_j_inv = 1.0 / mass[j];
    atomicAdd(x + i, -cx * mass_i_inv);
    atomicAdd(y + i, -cy * mass_i_inv);
    atomicAdd(z + i, -cz * mass_i_inv);
    atomicAdd(x + j, cx * mass_j_inv);
    atomicAdd(y + j, cy * mass_j_inv);
    atomicAdd(z + j, cz * mass_j_inv);
    atomicAdd(vx + i, -cx * mass_i_inv * inverse_time_step);
    atomicAdd(vy + i, -cy * mass_i_inv * inverse_time_step);
    atomicAdd(vz + i, -cz * mass_i_inv * inverse_time_step);
    atomicAdd(vx + j, cx * mass_j_inv * inverse_time_step);
    atomicAdd(vy + j, cy * mass_j_inv * inverse_time_step);
    atomicAdd(vz + j, cz * mass_j_inv * inverse_time_step);
  }
}

// right-hand side for velocities: w_k = 2 * b_k . (v_i - v_j)
static __global__ void gpu_compute_velocity_rhs(
  const int number_of_constraints,
  const int* __restrict__ atom_i,
  const int* __restrict__ atom_j,
  const double* __restrict__ bond_vec,
  const double* __restrict__ vx,
  const double* __restrict__ vy,
  const double* __restrict__ vz,
  double* __restrict__ rhs)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int K = number_of_constraints;
    const int i = atom_i[k];
    const int j = atom_j[k];
    rhs[k] = 2.0 *
      (bond_vec[k] * (vx[i] - vx[j]) + bond_vec[K + k] * (vy[i] - vy[j]) +
       bond_vec[2 * K + k] * (vz[i] - vz[j]));
  }
}

// v_i -= 2 * mu_k * b_k / m_i; v_j += 2 * mu_k * b_k / m_j
static __global__ void gpu_apply_velocity_correction(
  const int number_of_constraints,
  const int* __restrict__ atom_i,
  const int* __restrict__ atom_j,
  const double* __restrict__ mass,
  const double* __restrict__ bond_vec,
  const double* __restrict__ lagrange,
  double* __restrict__ vx,
  double* __restrict__ vy,
  double* __restrict__ vz)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < number_of_constraints) {
    const int K = number_of_constraints;
    const int i = atom_i[k];
    const int j = atom_j[k];
    const double cx = 2.0 * lagrange[k] * bond_vec[k];
    const double cy = 2.0 * lagrange[k] * bond_vec[K + k];
    const double cz = 2.0 * lagrange[k] * bond_vec[2 * K + k];
    const double mass_i_inv = 1.0 / mass[i];
    const double mass_j_inv = 1.0 / mass[j];
    atomicAdd(vx + i, -cx * mass_i_inv);
    atomicAdd(vy + i, -cy * mass_i_inv);
    atomicAdd(vz + i, -cz * mass_i_inv);
    atomicAdd(vx + j, cx * mass_j_inv);
    atomicAdd(vy + j, cy * mass_j_inv);
    atomicAdd(vz + j, cz * mass_j_inv);
  }
}

void Lincs::compute_bond_vectors(const Box& box, Atom& atom)
{
  const int K = number_of_constraints;
  const int N = atom.number_of_atoms;
  gpu_compute_bond_vectors<<<(K - 1) / 128 + 1, 128>>>(
    K,
    box,
    atom_i.data(),
    atom_j.data(),
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + N,
    atom.position_per_atom.data() + 2 * N,
    bond_vec.data());
  GPU_CHECK_KERNEL
}

void Lincs::compute_diag_inv(Atom& atom)
{
  const int K = number_of_constraints;
  gpu_compute_diag_inv<<<(K - 1) / 128 + 1, 128>>>(
    K, atom_i.data(), atom_j.data(), atom.mass.data(), bond_vec.data(), diag_inv.data());
  GPU_CHECK_KERNEL
}

void Lincs::compute_rhs(const Box& box, Atom& atom)
{
  const int K = number_of_constraints;
  const int N = atom.number_of_atoms;
  gpu_compute_rhs<<<(K - 1) / 128 + 1, 128>>>(
    K,
    box,
    atom_i.data(),
    atom_j.data(),
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + N,
    atom.position_per_atom.data() + 2 * N,
    target.data(),
    rhs.data());
  GPU_CHECK_KERNEL
}

void Lincs::compute_velocity_rhs(Atom& atom)
{
  const int K = number_of_constraints;
  const int N = atom.number_of_atoms;
  gpu_compute_velocity_rhs<<<(K - 1) / 128 + 1, 128>>>(
    K,
    atom_i.data(),
    atom_j.data(),
    bond_vec.data(),
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + N,
    atom.velocity_per_atom.data() + 2 * N,
    rhs.data());
  GPU_CHECK_KERNEL
}

// solve T * lambda = rhs by a truncated Neumann series:
// lambda = [I + N + N^2 + ...] * D^{-1} * rhs, with N = D^{-1} * (D - T)
void Lincs::solve_linear_system(Atom& atom)
{
  const int K = number_of_constraints;
  const int grid_size = (K - 1) / 128 + 1;
  gpu_init_lambda<<<grid_size, 128>>>(
    K, diag_inv.data(), rhs.data(), lagrange.data(), matvec_a.data());
  GPU_CHECK_KERNEL
  const double* in = matvec_a.data();
  double* out = matvec_b.data();
  for (int m = 1; m < expansion_order; ++m) {
    gpu_neumann_matvec<<<grid_size, 128>>>(
      K,
      bond_vec.data(),
      coupl_offset.data(),
      coupl_index.data(),
      coupl_atom.data(),
      coupl_sign.data(),
      atom.mass.data(),
      diag_inv.data(),
      in,
      out);
    GPU_CHECK_KERNEL
    gpu_accumulate<<<grid_size, 128>>>(K, out, lagrange.data());
    GPU_CHECK_KERNEL
    const double* temp = in;
    in = out;
    out = const_cast<double*>(temp);
  }
}

void Lincs::apply_position_correction(const double time_step, Atom& atom)
{
  const int K = number_of_constraints;
  const int N = atom.number_of_atoms;
  gpu_apply_position_correction<<<(K - 1) / 128 + 1, 128>>>(
    K,
    1.0 / time_step,
    atom_i.data(),
    atom_j.data(),
    atom.mass.data(),
    bond_vec.data(),
    lagrange.data(),
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + N,
    atom.position_per_atom.data() + 2 * N,
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + N,
    atom.velocity_per_atom.data() + 2 * N);
  GPU_CHECK_KERNEL
}

void Lincs::apply_velocity_correction(Atom& atom)
{
  const int K = number_of_constraints;
  const int N = atom.number_of_atoms;
  gpu_apply_velocity_correction<<<(K - 1) / 128 + 1, 128>>>(
    K,
    atom_i.data(),
    atom_j.data(),
    atom.mass.data(),
    bond_vec.data(),
    lagrange.data(),
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + N,
    atom.velocity_per_atom.data() + 2 * N);
  GPU_CHECK_KERNEL
}

void Lincs::detect_bonds(
  Atom& atom,
  const Box& box,
  const std::vector<double>& position,
  std::vector<int>& cpu_atom_i,
  std::vector<int>& cpu_atom_j,
  std::vector<double>& cpu_target)
{
  const int N = atom.number_of_atoms;
  if (static_cast<int>(atom.cpu_atom_symbol.size()) != N) {
    PRINT_INPUT_ERROR("Element symbols in model.xyz are needed for constraint detection.");
  }

  // atomic numbers and covalent radii
  std::vector<int> atomic_number(N);
  std::vector<double> radius(N);
  double max_radius = 0.0;
  for (int n = 0; n < N; ++n) {
    atomic_number[n] = get_atomic_number(atom.cpu_atom_symbol[n]);
    if (atomic_number[n] < 1) {
      PRINT_INPUT_ERROR("Unknown element symbol in model.xyz for constraint detection.");
    }
    radius[n] = COVALENT_RADII[atomic_number[n] - 1];
    if (radius[n] > max_radius) {
      max_radius = radius[n];
    }
  }

  const double rcut = 2.0 * max_radius * cutoff_factor;

  // fractional coordinates
  const double* hi = box.cpu_h + 9;
  std::vector<double> frac(3 * N);
  for (int n = 0; n < N; ++n) {
    frac[n] = hi[0] * position[n] + hi[1] * position[N + n] + hi[2] * position[2 * N + n];
    frac[N + n] = hi[3] * position[n] + hi[4] * position[N + n] + hi[5] * position[2 * N + n];
    frac[2 * N + n] = hi[6] * position[n] + hi[7] * position[N + n] + hi[8] * position[2 * N + n];
  }

  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  const double volume = box.get_volume();
  const double thickness[3] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};

  // wrap fractional coordinates into [0, 1) for periodic directions and
  // shift them to start from zero for non-periodic directions
  double range[3];
  for (int d = 0; d < 3; ++d) {
    double* f = frac.data() + d * N;
    if (pbc[d]) {
      for (int n = 0; n < N; ++n) {
        f[n] -= floor(f[n]);
      }
      range[d] = 1.0;
    } else {
      double fmin = f[0];
      for (int n = 1; n < N; ++n) {
        if (f[n] < fmin) {
          fmin = f[n];
        }
      }
      double fmax = fmin;
      for (int n = 0; n < N; ++n) {
        f[n] -= fmin;
        if (f[n] > fmax) {
          fmax = f[n];
        }
      }
      range[d] = fmax > 1.0e-6 ? fmax : 1.0e-6;
    }
  }

  // cell list in fractional space; a pair with distance < rcut has
  // |delta_s_d| < rcut / thickness_d in every direction
  int ncell[3];
  for (int d = 0; d < 3; ++d) {
    const double fcut = rcut / thickness[d];
    ncell[d] = static_cast<int>(range[d] / fcut);
    if (ncell[d] < 1) {
      ncell[d] = 1;
    }
  }
  const int total_cells = ncell[0] * ncell[1] * ncell[2];

  std::vector<int> cell_of(N);
  std::vector<int> cell_count(total_cells + 1, 0);
  for (int n = 0; n < N; ++n) {
    int c[3];
    for (int d = 0; d < 3; ++d) {
      c[d] = static_cast<int>(frac[d * N + n] / range[d] * ncell[d]);
      if (c[d] >= ncell[d]) {
        c[d] = ncell[d] - 1;
      }
      if (c[d] < 0) {
        c[d] = 0;
      }
    }
    const int cell = c[0] + ncell[0] * (c[1] + ncell[1] * c[2]);
    cell_of[n] = cell;
    ++cell_count[cell + 1];
  }
  for (int c = 0; c < total_cells; ++c) {
    cell_count[c + 1] += cell_count[c];
  }
  std::vector<int> cell_contents(N);
  {
    std::vector<int> filled(total_cells, 0);
    for (int n = 0; n < N; ++n) {
      const int cell = cell_of[n];
      cell_contents[cell_count[cell] + filled[cell]] = n;
      ++filled[cell];
    }
  }

  // detect bonds
  const double* px = position.data();
  const double* py = position.data() + N;
  const double* pz = position.data() + 2 * N;
  for (int i = 0; i < N; ++i) {
    const int ci = cell_of[i];
    int c[3];
    c[0] = ci % ncell[0];
    c[1] = (ci / ncell[0]) % ncell[1];
    c[2] = ci / (ncell[0] * ncell[1]);
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int dd[3] = {dx, dy, dz};
          int cn[3];
          bool valid = true;
          for (int d = 0; d < 3; ++d) {
            if (pbc[d]) {
              // with ncell 1 or 2, only scan +1 to avoid duplicated cells
              if (ncell[d] < 3 && dd[d] < 0) {
                valid = false;
                break;
              }
              cn[d] = (c[d] + dd[d] + ncell[d]) % ncell[d];
            } else {
              cn[d] = c[d] + dd[d];
              if (cn[d] < 0 || cn[d] >= ncell[d]) {
                valid = false;
                break;
              }
            }
          }
          if (!valid) {
            continue;
          }
          const int cell = cn[0] + ncell[0] * (cn[1] + ncell[1] * cn[2]);
          for (int p = cell_count[cell]; p < cell_count[cell + 1]; ++p) {
            const int j = cell_contents[p];
            if (j <= i) {
              continue;
            }
            if (mode == 1 && atomic_number[i] != 1 && atomic_number[j] != 1) {
              continue;
            }
            double dx = px[i] - px[j];
            double dy = py[i] - py[j];
            double dz2 = pz[i] - pz[j];
            apply_mic(box, dx, dy, dz2);
            const double r2 = dx * dx + dy * dy + dz2 * dz2;
            const double rmax = (radius[i] + radius[j]) * cutoff_factor;
            if (r2 < rmax * rmax) {
              cpu_atom_i.push_back(i);
              cpu_atom_j.push_back(j);
              cpu_target.push_back(sqrt(r2));
            }
          }
        }
      }
    }
  }
}

void Lincs::read_constraints_from_file(
  Atom& atom,
  const Box& box,
  const std::vector<double>& position,
  std::vector<int>& cpu_atom_i,
  std::vector<int>& cpu_atom_j,
  std::vector<double>& cpu_target)
{
  const int N = atom.number_of_atoms;
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open the constraint file.");
  }

  const double* px = position.data();
  const double* py = position.data() + N;
  const double* pz = position.data() + 2 * N;

  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    std::vector<std::string> tokens_without_comments;
    for (const auto& t : tokens) {
      if (t[0] != '#') {
        tokens_without_comments.emplace_back(t);
      } else {
        break;
      }
    }
    if (tokens_without_comments.size() == 0) {
      continue;
    }
    if (tokens_without_comments.size() < 2 || tokens_without_comments.size() > 3) {
      PRINT_INPUT_ERROR("Each line of the constraint file should be: atom_i atom_j [length].");
    }
    int i, j;
    if (!is_valid_int(tokens_without_comments[0].c_str(), &i) ||
        !is_valid_int(tokens_without_comments[1].c_str(), &j)) {
      PRINT_INPUT_ERROR("Atom indices in the constraint file should be integers.");
    }
    // atom indices in the file start from 1
    --i;
    --j;
    if (i < 0 || i >= N || j < 0 || j >= N) {
      PRINT_INPUT_ERROR("Atom index in the constraint file is out of range.");
    }
    if (i == j) {
      PRINT_INPUT_ERROR("A constraint cannot connect an atom to itself.");
    }
    double d0;
    if (tokens_without_comments.size() == 3) {
      if (!is_valid_real(tokens_without_comments[2].c_str(), &d0)) {
        PRINT_INPUT_ERROR("Target length in the constraint file should be a number.");
      }
      if (d0 <= 0.0) {
        PRINT_INPUT_ERROR("Target length in the constraint file should > 0.");
      }
    } else {
      // default target length: the distance in the initial structure
      double dx = px[i] - px[j];
      double dy = py[i] - py[j];
      double dz = pz[i] - pz[j];
      apply_mic(box, dx, dy, dz);
      d0 = sqrt(dx * dx + dy * dy + dz * dz);
    }
    cpu_atom_i.push_back(i);
    cpu_atom_j.push_back(j);
    cpu_target.push_back(d0);
  }
  input.close();
}

void Lincs::setup(Atom& atom, Box& box)
{
  if (!enabled || setup_done) {
    return;
  }
  setup_done = true;

  const int N = atom.number_of_atoms;

  // positions on the host
  std::vector<double> position(3 * N);
  atom.position_per_atom.copy_to_host(position.data());

  std::vector<int> cpu_atom_i;
  std::vector<int> cpu_atom_j;
  std::vector<double> cpu_target;
  if (mode == 3) {
    read_constraints_from_file(atom, box, position, cpu_atom_i, cpu_atom_j, cpu_target);
  } else {
    detect_bonds(atom, box, position, cpu_atom_i, cpu_atom_j, cpu_target);
  }

  number_of_constraints = static_cast<int>(cpu_atom_i.size());
  printf("    number of constrained bonds = %d.\n", number_of_constraints);
  if (number_of_constraints == 0) {
    printf("Warning: no bonds detected; bond constraints have no effect.\n");
    return;
  }

  // coupling structure: two constraints couple if they share an atom
  std::vector<std::vector<std::pair<int, int>>> atom_to_constraints(N);
  for (int k = 0; k < number_of_constraints; ++k) {
    atom_to_constraints[cpu_atom_i[k]].emplace_back(k, 0);
    atom_to_constraints[cpu_atom_j[k]].emplace_back(k, 1);
  }
  std::vector<int> cpu_coupl_offset(number_of_constraints + 1, 0);
  std::vector<int> cpu_coupl_index;
  std::vector<int> cpu_coupl_atom;
  std::vector<double> cpu_coupl_sign;
  for (int k = 0; k < number_of_constraints; ++k) {
    const int atoms_of_k[2] = {cpu_atom_i[k], cpu_atom_j[k]};
    for (int a = 0; a < 2; ++a) {
      const int shared = atoms_of_k[a];
      for (const auto& kl : atom_to_constraints[shared]) {
        if (kl.first == k) {
          continue;
        }
        // same role (both first or both second) -> +1, otherwise -> -1
        cpu_coupl_index.push_back(kl.first);
        cpu_coupl_atom.push_back(shared);
        cpu_coupl_sign.push_back(kl.second == a ? 1.0 : -1.0);
      }
    }
    cpu_coupl_offset[k + 1] = static_cast<int>(cpu_coupl_index.size());
  }

  // upload to the GPU
  atom_i.resize(number_of_constraints);
  atom_j.resize(number_of_constraints);
  target.resize(number_of_constraints);
  atom_i.copy_from_host(cpu_atom_i.data());
  atom_j.copy_from_host(cpu_atom_j.data());
  target.copy_from_host(cpu_target.data());
  coupl_offset.resize(number_of_constraints + 1);
  coupl_index.resize(cpu_coupl_index.size());
  coupl_atom.resize(cpu_coupl_atom.size());
  coupl_sign.resize(cpu_coupl_sign.size());
  coupl_offset.copy_from_host(cpu_coupl_offset.data());
  coupl_index.copy_from_host(cpu_coupl_index.data());
  coupl_atom.copy_from_host(cpu_coupl_atom.data());
  coupl_sign.copy_from_host(cpu_coupl_sign.data());
  bond_vec.resize(3 * number_of_constraints);
  rhs.resize(number_of_constraints);
  diag_inv.resize(number_of_constraints);
  lagrange.resize(number_of_constraints);
  matvec_a.resize(number_of_constraints);
  matvec_b.resize(number_of_constraints);

  // bond vectors at the initial (constrained) positions
  compute_bond_vectors(box, atom);
}

void Lincs::compute1(const double time_step, const Box& box, Atom& atom)
{
  if (!enabled || number_of_constraints == 0) {
    return;
  }

  // first solve uses the bond vectors at the previous constrained positions
  compute_diag_inv(atom);
  compute_rhs(box, atom);
  solve_linear_system(atom);
  apply_position_correction(time_step, atom);

  // re-linearization iterations
  for (int it = 0; it < num_iterations; ++it) {
    compute_bond_vectors(box, atom);
    compute_diag_inv(atom);
    compute_rhs(box, atom);
    solve_linear_system(atom);
    apply_position_correction(time_step, atom);
  }
}

void Lincs::compute2(const Box& box, Atom& atom)
{
  if (!enabled || number_of_constraints == 0) {
    return;
  }

  // refresh the bond vectors at the final constrained positions; they are
  // also the linearization point for the next step
  compute_bond_vectors(box, atom);
  compute_diag_inv(atom);
  compute_velocity_rhs(atom);
  solve_linear_system(atom);
  apply_velocity_correction(atom);
}
