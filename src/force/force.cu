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
The driver class calculating force and related quantities.
------------------------------------------------------------------------------*/

#include "eam.cuh"
#include "fcp.cuh"
#include "force.cuh"
#include "lj.cuh"
#include "nep3.cuh"
#include "nep3_multigpu.cuh"
#include "potential.cuh"
#include "tersoff1988.cuh"
#include "tersoff1989.cuh"
#include "tersoff_mini.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <vector>

#define BLOCK_SIZE 128

Force::Force(void) { is_fcp = false; }

void Force::parse_potential(
  char** param, int num_param, char* input_dir, const Box& box, const int number_of_atoms)
{
  static int num_calls = 0;
  if (num_calls++ != 0) {
    PRINT_INPUT_ERROR("potential keyword can only be used once.\n");
  }

  if (num_param != 2) {
    PRINT_INPUT_ERROR("potential should have 1 parameter.\n");
  }

  FILE* fid_potential = my_fopen(param[1], "r");
  char potential_name[20];
  int count = fscanf(fid_potential, "%s", potential_name);
  if (count != 1) {
    PRINT_INPUT_ERROR("reading error for potential file.");
  }
  int num_types = get_number_of_types(fid_potential);

  // determine the potential
  if (strcmp(potential_name, "tersoff_1989") == 0) {
    potential.reset(new Tersoff1989(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "tersoff_1988") == 0) {
    potential.reset(new Tersoff1988(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "tersoff_mini") == 0) {
    potential.reset(new Tersoff_mini(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "eam_zhou_2004") == 0) {
    potential.reset(new EAM(fid_potential, potential_name, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "eam_dai_2006") == 0) {
    potential.reset(new EAM(fid_potential, potential_name, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "fcp") == 0) {
    potential.reset(new FCP(fid_potential, input_dir, num_types, number_of_atoms, box));
    is_fcp = true;
  } else if (
    strcmp(potential_name, "nep") == 0 || strcmp(potential_name, "nep_zbl") == 0 ||
    strcmp(potential_name, "nep3") == 0 || strcmp(potential_name, "nep3_zbl") == 0) {
    int num_gpus;
    CHECK(cudaGetDeviceCount(&num_gpus));
#ifdef ZHEYONG
    num_gpus = 3;
#endif
    if (num_gpus == 1) {
      potential.reset(new NEP3(param[1], number_of_atoms));
    } else {
      potential.reset(new NEP3_MULTIGPU(num_gpus, param[1], number_of_atoms));
    }
  } else if (strcmp(potential_name, "lj") == 0) {
    potential.reset(new LJ(fid_potential, num_types, number_of_atoms));
  } else {
    PRINT_INPUT_ERROR("illegal potential model.\n");
  }

  fclose(fid_potential);

  potential->N1 = 0;
  potential->N2 = number_of_atoms;
}

int Force::get_number_of_types(FILE* fid_potential)
{
  int num_of_types;
  int count = fscanf(fid_potential, "%d", &num_of_types);
  PRINT_SCANF_ERROR(count, 1, "Reading error for number of types.");
  return num_of_types;
}

static __global__ void gpu_add_driving_force(
  int N,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] += fe_x * g_sxx[i] + fe_y * g_syx[i] + fe_z * g_szx[i];
    g_fy[i] += fe_x * g_sxy[i] + fe_y * g_syy[i] + fe_z * g_szy[i];
    g_fz[i] += fe_x * g_sxz[i] + fe_y * g_syz[i] + fe_z * g_szz[i];
  }
}

// get the total force
static __global__ void gpu_sum_force(int N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_f[1024];
  double f = 0.0;

  switch (bid) {
    case 0:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fx[n];
      }
      break;
    case 1:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fy[n];
      }
      break;
    case 2:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fz[n];
      }
      break;
  }
  s_f[tid] = f;
  __syncthreads();

#pragma unroll
  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
  }
  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    g_f[bid] = s_f[0];
  }
}

// correct the total force
static __global__ void
gpu_correct_force(int N, double one_over_N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] -= g_f[0] * one_over_N;
    g_fy[i] -= g_f[1] * one_over_N;
    g_fz[i] -= g_f[2] * one_over_N;
  }
}

static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    g_virial[n1 + 0 * N] = 0.0;
    g_virial[n1 + 1 * N] = 0.0;
    g_virial[n1 + 2 * N] = 0.0;
    g_virial[n1 + 3 * N] = 0.0;
    g_virial[n1 + 4 * N] = 0.0;
    g_virial[n1 + 5 * N] = 0.0;
    g_virial[n1 + 6 * N] = 0.0;
    g_virial[n1 + 7 * N] = 0.0;
    g_virial[n1 + 8 * N] = 0.0;
  }
}

void Force::set_hnemd_parameters(
  const bool compute_hnemd,
  const double hnemd_fe_x,
  const double hnemd_fe_y,
  const double hnemd_fe_z)
{
  compute_hnemd_ = compute_hnemd;
  if (compute_hnemd) {
    hnemd_fe_[0] = hnemd_fe_x;
    hnemd_fe_[1] = hnemd_fe_y;
    hnemd_fe_[2] = hnemd_fe_z;
  }
}

void Force::set_hnemdec_parameters(
  const int compute_hnemdec,
  const double hnemd_fe_x,
  const double hnemd_fe_y,
  const double hnemd_fe_z,
  const std::vector<double>& mass,
  const std::vector<int>& type,
  const std::vector<int>& type_size,
  const double T)
{
  int N = mass.size();
  int N1 = type_size[0];
  int N2 = type_size[1];
  int number_of_types = type_size.size();
  compute_hnemdec_ = compute_hnemdec;
  temperature = T;
  std::vector<double> cpu_coefficient;

  // find 2 atom types' mass or fraction
  if (compute_hnemdec_ == 1) {
    std::vector<double> mass_type;
    mass_type.resize(number_of_types);
    int find_mass_type = 0;
    for (int i = 0; i < mass_type.size(); i++) {
      mass_type[i] = 0;
    }
    for (int i = 0; i < N; i++) {
      if (mass_type[type[i]] != mass[i]) {
        mass_type[type[i]] = mass[i];
        find_mass_type += 1;
      }
      if (find_mass_type == 2) {
        break;
      }
    }

    int COEFF_NUM = 4;
    cpu_coefficient.resize(COEFF_NUM);
    coefficient.resize(COEFF_NUM);

    double m1 = mass_type[0];
    double m2 = mass_type[1];
    double miu = 1 / m1 - 1 / m2;
    double tmp = m1 * m2 / (m1 * N1 + m2 * N2);
    double c1 = miu * N2 * tmp;
    double c2 = -1 * miu * N1 * tmp;
    double c11 = (c1 - 1) / N;
    double c12 = c1 * K_B * temperature;
    double c21 = (c2 - 1) / N;
    double c22 = c2 * K_B * temperature;

    cpu_coefficient[0] = c11;
    cpu_coefficient[1] = c21;
    cpu_coefficient[2] = c12;
    cpu_coefficient[3] = c22;
    coefficient.copy_from_host(cpu_coefficient.data());
  } else if (compute_hnemdec_ == 2) {
    int COEFF_NUM = 2;
    cpu_coefficient.resize(COEFF_NUM);
    cpu_coefficient[0] = N2 / double(N);
    cpu_coefficient[1] = -1 * N1 / double(N);
    coefficient.resize(COEFF_NUM);
    coefficient.copy_from_host(cpu_coefficient.data());
  }

  hnemd_fe_[0] = hnemd_fe_x;
  hnemd_fe_[1] = hnemd_fe_y;
  hnemd_fe_[2] = hnemd_fe_z;
}

static __global__ void gpu_apply_pbc(int N, Box box, double* g_x, double* g_y, double* g_z)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    if (box.triclinic == 0) {
      double lx = box.cpu_h[0];
      double ly = box.cpu_h[1];
      double lz = box.cpu_h[2];
      if (box.pbc_x == 1) {
        if (g_x[n] < 0) {
          g_x[n] += lx;
        } else if (g_x[n] > lx) {
          g_x[n] -= lx;
        }
      }
      if (box.pbc_y == 1) {
        if (g_y[n] < 0) {
          g_y[n] += ly;
        } else if (g_y[n] > ly) {
          g_y[n] -= ly;
        }
      }
      if (box.pbc_z == 1) {
        if (g_z[n] < 0) {
          g_z[n] += lz;
        } else if (g_z[n] > lz) {
          g_z[n] -= lz;
        }
      }
    } else {
      double x = g_x[n];
      double y = g_y[n];
      double z = g_z[n];
      double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
      double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
      double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
      if (box.pbc_x == 1) {
        if (sx < 0.0) {
          sx += 1.0;
        } else if (sx > 1.0) {
          sx -= 1.0;
        }
      }
      if (box.pbc_y == 1) {
        if (sy < 0.0) {
          sy += 1.0;
        } else if (sy > 1.0) {
          sy -= 1.0;
        }
      }
      if (box.pbc_z == 1) {
        if (sz < 0.0) {
          sz += 1.0;
        } else if (sz > 1.0) {
          sz -= 1.0;
        }
      }
      g_x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
      g_y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
      g_z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
    }
  }
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  if (!is_fcp) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, box, position_per_atom.data(), position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2);
  }

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(), virial_per_atom.data());
  CUDA_CHECK_KERNEL

  potential->compute(
    box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);

  if (compute_hnemd_) {
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, hnemd_fe_[0], hnemd_fe_[1], hnemd_fe_[2],
      virial_per_atom.data() + 0 * number_of_atoms, virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms, virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms, virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms, virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms, force_per_atom.data(),
      force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms);

    GPU_Vector<double> ftot(3); // total force vector of the system

    gpu_sum_force<<<3, 1024>>>(
      number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms, ftot.data());
    CUDA_CHECK_KERNEL

    gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, 1.0 / number_of_atoms, force_per_atom.data(),
      force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    CUDA_CHECK_KERNEL
  }

  // always correct the force when using the FCP potential
  if (is_fcp) {
    if (!compute_hnemd_) {
      GPU_Vector<double> ftot(3); // total force vector of the system
      gpu_sum_force<<<3, 1024>>>(
        number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms, ftot.data());
      CUDA_CHECK_KERNEL

      gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms, 1.0 / number_of_atoms, force_per_atom.data(),
        force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
        ftot.data());
      CUDA_CHECK_KERNEL
    }
  }
}

static __global__ void gpu_find_per_atom_tensor(
  int N,
  double* g_mass,
  double* g_potential,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_tensor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double mass = g_mass[i];
    double potential = g_potential[i];
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    double energy = mass * (vx * vx + vy * vy + vz * vz) * 0.5 + potential;
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_tensor[i] = energy + g_sxx[i];
    g_tensor[i + 3 * N] = g_sxy[i];
    g_tensor[i + 4 * N] = g_sxz[i];
    g_tensor[i + 6 * N] = g_syx[i];
    g_tensor[i + N] = energy + g_syy[i];
    g_tensor[i + 5 * N] = g_syz[i];
    g_tensor[i + 7 * N] = g_szx[i];
    g_tensor[i + 8 * N] = g_szy[i];
    g_tensor[i + 2 * N] = energy + g_szz[i];
  }
}

static __global__ void gpu_sum_tensor(int N, double* g_tensor, double* g_sum_tensor)
{
  //<<<9,1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_t[1024];
  double t = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 1024;
    if (n < N)
      t += g_tensor[bid * N + n];
  }
  s_t[tid] = t;
  __syncthreads();

#pragma unroll
  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_t[tid] += s_t[tid + offset];
    }
    __syncthreads();
  }
  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_t[tid] += s_t[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    g_sum_tensor[bid] = s_t[0];
  }
}

static __global__ void gpu_add_driving_force(
  int N,
  const double* coefficient,
  const int* g_type,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_tensor_tot,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  // heat flow algorithm
  // coefficient: c11,c21,c12,c22
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int type = g_type[i];
    double coefficient1 = coefficient[type];
    double coefficient2 = coefficient[type + 2];

    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_fx[i] += fe_x * (g_sxx[i] + coefficient1 * g_tensor_tot[0] + coefficient2) +
               fe_y * (g_syx[i] + coefficient1 * g_tensor_tot[6]) +
               fe_z * (g_szx[i] + coefficient1 * g_tensor_tot[7]);

    g_fy[i] += fe_x * (g_sxy[i] + coefficient1 * g_tensor_tot[3]) +
               fe_y * (g_syy[i] + coefficient1 * g_tensor_tot[1] + coefficient2) +
               fe_z * (g_szy[i] + coefficient1 * g_tensor_tot[8]);

    g_fz[i] += fe_x * (g_sxz[i] + coefficient1 * g_tensor_tot[4]) +
               fe_y * (g_syz[i] + coefficient1 * g_tensor_tot[5]) +
               fe_z * (g_szz[i] + coefficient1 * g_tensor_tot[2] + coefficient2);
  }
}

static __global__ void gpu_add_driving_force(
  int N,
  const double* g_coefficient,
  const int* g_type,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  // color conductivity algorithm
  // coefficient: c1,c2
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int type = g_type[i];
    double coefficient = g_coefficient[type];

    g_fx[i] += fe_x * coefficient;
    g_fy[i] += fe_y * coefficient;
    g_fz[i] += fe_z * coefficient;
  }
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& mass_per_atom)
{
  const int number_of_atoms = type.size();
  if (!is_fcp) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, box, position_per_atom.data(), position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2);
  }

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(), virial_per_atom.data());
  CUDA_CHECK_KERNEL

  potential->compute(
    box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);

  if (compute_hnemd_) {
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, hnemd_fe_[0], hnemd_fe_[1], hnemd_fe_[2],
      virial_per_atom.data() + 0 * number_of_atoms, virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms, virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms, virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms, virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms, force_per_atom.data(),
      force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms);

    GPU_Vector<double> ftot(3); // total force vector of the system

    gpu_sum_force<<<3, 1024>>>(
      number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms, ftot.data());
    CUDA_CHECK_KERNEL

    gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, 1.0 / number_of_atoms, force_per_atom.data(),
      force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    CUDA_CHECK_KERNEL
  } else if (compute_hnemdec_ == 1) {
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    GPU_Vector<double> tensor_per_atom(number_of_atoms * 9);
    GPU_Vector<double> tensor_tot(9);

    gpu_find_per_atom_tensor<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, mass_per_atom.data(), potential_per_atom.data(), velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms,
      virial_per_atom.data() + 0 * number_of_atoms, virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms, virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms, virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms, virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms, tensor_per_atom.data());
    CUDA_CHECK_KERNEL

    gpu_sum_tensor<<<9, 1024>>>(number_of_atoms, tensor_per_atom.data(), tensor_tot.data());
    CUDA_CHECK_KERNEL

    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, coefficient.data(), type.data(), hnemd_fe_[0], hnemd_fe_[1], hnemd_fe_[2],
      tensor_per_atom.data() + 0 * number_of_atoms, tensor_per_atom.data() + 3 * number_of_atoms,
      tensor_per_atom.data() + 4 * number_of_atoms, tensor_per_atom.data() + 6 * number_of_atoms,
      tensor_per_atom.data() + 1 * number_of_atoms, tensor_per_atom.data() + 5 * number_of_atoms,
      tensor_per_atom.data() + 7 * number_of_atoms, tensor_per_atom.data() + 8 * number_of_atoms,
      tensor_per_atom.data() + 2 * number_of_atoms, tensor_tot.data(), force_per_atom.data(),
      force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms);
    CUDA_CHECK_KERNEL

  } else if (compute_hnemdec_ == 2) {
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, coefficient.data(), type.data(), hnemd_fe_[0], hnemd_fe_[1], hnemd_fe_[2],
      force_per_atom.data(), force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);
  }

  // always correct the force when using the FCP potential
  if (is_fcp) {
    if (!compute_hnemd_) {
      GPU_Vector<double> ftot(3); // total force vector of the system
      gpu_sum_force<<<3, 1024>>>(
        number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
        force_per_atom.data() + 2 * number_of_atoms, ftot.data());
      CUDA_CHECK_KERNEL

      gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
        number_of_atoms, 1.0 / number_of_atoms, force_per_atom.data(),
        force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms,
        ftot.data());
      CUDA_CHECK_KERNEL
    }
  }
}
