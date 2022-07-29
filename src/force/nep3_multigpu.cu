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

This is the multi-GPU (single-node) version. It has good parallel efficiency
when there is NVlink, but is also not very bad when there is only PCI-E.
------------------------------------------------------------------------------*/

#include "nep3_multigpu.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include <iostream>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
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

NEP3_MULTIGPU::NEP3_MULTIGPU(const int num_gpus, char* file_potential, const int num_atoms)
{

  printf("Try to use %d GPUs for the NEP part.\n", num_gpus);

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep") {
    paramb.version = 2;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3") {
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep_zbl") {
    paramb.version = 2;
    zbl.enabled = true;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.version = 3;
    zbl.enabled = true;
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.version == 2) {
    if (paramb.num_types == 1) {
      printf("Use the NEP2 potential with %d atom type.\n", paramb.num_types);
    } else {
      printf("Use the NEP2 potential with %d atom types.\n", paramb.num_types);
    }
  } else {
    if (paramb.num_types == 1) {
      printf("Use the NEP3 potential with %d atom type.\n", paramb.num_types);
    } else {
      printf("Use the NEP3 potential with %d atom types.\n", paramb.num_types);
    }
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %g).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be zbl rc_inner rc_outer." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_float_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_float_from_token(tokens[2], __FILE__, __LINE__);
    printf(
      "    has ZBL with inner cutoff %g A and outer cutoff %g A.\n", zbl.rc_inner, zbl.rc_outer);
  }

  // cutoff 4.2 3.7 80 47
  tokens = get_tokens(input);
  if (tokens.size() != 3 && tokens.size() != 5) {
    std::cout << "This line should be cutoff rc_radial rc_angular [MN_radial] [MN_angular].\n";
    exit(1);
  }
  paramb.rc_radial = get_float_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_float_from_token(tokens[2], __FILE__, __LINE__);
  printf("    radial cutoff = %g A.\n", paramb.rc_radial);
  printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  paramb.MN_radial = 500;
  paramb.MN_angular = 100;

  if (tokens.size() == 5) {
    int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
    printf("    MN_radial = %d.\n", MN_radial);
    printf("    MN_angular = %d.\n", MN_angular);
    paramb.MN_radial = int(ceil(MN_radial * 1.25));
    paramb.MN_angular = int(ceil(MN_angular * 1.25));
    printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
    printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);
  }

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  if (paramb.version == 3) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
                << std::endl;
      exit(1);
    }
    paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
    paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
    printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
    printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);
  }

  // l_max
  tokens = get_tokens(input);
  if (paramb.version == 2) {
    if (tokens.size() != 2) {
      std::cout << "This line should be l_max l_max_3body." << std::endl;
      exit(1);
    }
  } else if (paramb.version == 3) {
    if (tokens.size() != 4) {
      std::cout << "This line should be l_max l_max_3body l_max_4body l_max_5body." << std::endl;
      exit(1);
    }
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  if (paramb.version == 3) {
    int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
    int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
    printf("    l_max_4body = %d.\n", L_max_4body);
    printf("    l_max_5body = %d.\n", L_max_5body);
    if (L_max_4body == 2) {
      paramb.num_L += 1;
    }
    if (L_max_5body == 1) {
      paramb.num_L += 1;
    }
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb[0].num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb[0].dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  printf("    ANN = %d-%d-1.\n", annmb[0].dim, annmb[0].num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb[0].num_para = (annmb[0].dim + 2) * annmb[0].num_neurons1 + 1;
  printf("    number of neural network parameters = %d.\n", annmb[0].num_para);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  if (paramb.version == 2) {
    num_para_descriptor =
      (paramb.num_types == 1)
        ? 0
        : paramb.num_types_sq * (paramb.n_max_radial + paramb.n_max_angular + 2);
  }
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb[0].num_para += num_para_descriptor;
  printf("    total number of parameters = %d\n", annmb[0].num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb[0].num_para);
  for (int n = 0; n < annmb[0].num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }
  for (int d = 0; d < annmb[0].dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }

  paramb.num_gpus = num_gpus;
  nep_temp_data.num_atoms_per_gpu = num_atoms;
  if (num_gpus > 1) {
    nep_temp_data.num_atoms_per_gpu = (num_atoms * 1.25) / num_gpus;
  }

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    annmb[gpu].num_para = annmb[0].num_para;
    annmb[gpu].dim = annmb[0].dim;
    annmb[gpu].num_neurons1 = annmb[0].num_neurons1;
#ifndef ZHEYONG
    CHECK(cudaSetDevice(gpu));
#endif

    nep_data[gpu].parameters.resize(annmb[gpu].num_para);
    nep_data[gpu].parameters.copy_from_host(parameters.data());

    update_potential(nep_data[gpu].parameters.data(), annmb[gpu]);

    nep_data[gpu].cell_count.resize(num_atoms);
    nep_data[gpu].cell_count_sum.resize(num_atoms);
    nep_data[gpu].cell_contents.resize(num_atoms);

    CHECK(cudaStreamCreate(&nep_data[gpu].stream));
  }

  CHECK(cudaSetDevice(0));

  nep_temp_data.cell_count_sum_cpu.resize(num_atoms);
  nep_temp_data.cell_count.resize(num_atoms);
  nep_temp_data.cell_count_sum.resize(num_atoms);
  nep_temp_data.cell_contents.resize(num_atoms);

  allocate_memory();
}

void NEP3_MULTIGPU::allocate_memory()
{
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
    CHECK(cudaSetDevice(gpu));
#endif

    nep_data[gpu].f12x.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].f12y.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].f12z.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].NN_radial.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].NL_radial.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_radial);
    nep_data[gpu].NN_angular.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].NL_angular.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].Fp.resize(nep_temp_data.num_atoms_per_gpu * annmb[gpu].dim);
    nep_data[gpu].sum_fxyz.resize(
      nep_temp_data.num_atoms_per_gpu * (paramb.n_max_angular + 1) * NUM_OF_ABC);

    nep_data[gpu].type.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].position.resize(nep_temp_data.num_atoms_per_gpu * 3);
    nep_data[gpu].potential.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].force.resize(nep_temp_data.num_atoms_per_gpu * 3);
    nep_data[gpu].virial.resize(nep_temp_data.num_atoms_per_gpu * 9);
  }

  CHECK(cudaSetDevice(0));

  nep_temp_data.type.resize(nep_temp_data.num_atoms_per_gpu);
  nep_temp_data.position.resize(nep_temp_data.num_atoms_per_gpu * 3);
  nep_temp_data.potential.resize(nep_temp_data.num_atoms_per_gpu);
  nep_temp_data.force.resize(nep_temp_data.num_atoms_per_gpu * 3);
  nep_temp_data.virial.resize(nep_temp_data.num_atoms_per_gpu * 9);
}

NEP3_MULTIGPU::~NEP3_MULTIGPU(void)
{
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(cudaStreamDestroy(nep_data[gpu].stream));
  }
}

void NEP3_MULTIGPU::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  ann.b1 = ann.w1 + ann.num_neurons1;
  ann.c = ann.b1 + 1;
}

static __device__ void find_cell_id(
  const int partition_direction,
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  if (box.triclinic == 0) {
    cell_id_x = floor(x * rc_inv);
    cell_id_y = floor(y * rc_inv);
    cell_id_z = floor(z * rc_inv);
  } else {
    const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    cell_id_x = floor(sx * box.thickness_x * rc_inv);
    cell_id_y = floor(sy * box.thickness_y * rc_inv);
    cell_id_z = floor(sz * box.thickness_z * rc_inv);
  }
  if (cell_id_x < 0)
    cell_id_x += nx;
  if (cell_id_x >= nx)
    cell_id_x -= nx;
  if (cell_id_y < 0)
    cell_id_y += ny;
  if (cell_id_y >= ny)
    cell_id_y -= ny;
  if (cell_id_z < 0)
    cell_id_z += nz;
  if (cell_id_z >= nz)
    cell_id_z -= nz;
  if (partition_direction == 0) {
    cell_id = cell_id_y + ny * (cell_id_z + nz * cell_id_x);
  } else if (partition_direction == 1) {
    cell_id = cell_id_x + nx * (cell_id_z + nz * cell_id_y);
  } else {
    cell_id = cell_id_x + nx * (cell_id_y + ny * cell_id_z);
  }
}

static __device__ void find_cell_id(
  const int partition_direction,
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(
    partition_direction, box, x, y, z, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z,
    cell_id);
}

static __global__ void find_cell_counts(
  const int partition_direction,
  const Box box,
  const int N,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(partition_direction, box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents(
  const int partition_direction,
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(partition_direction, box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

static void __global__ set_to_zero(int size, int* data)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    data[n] = 0;
  }
}

static void find_cell_list(
  cudaStream_t& stream,
  const int partition_direction,
  const double rc,
  const int* num_bins,
  Box& box,
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int offset = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + offset;
  const double* z = position_per_atom.data() + offset * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  CUDA_CHECK_KERNEL

  set_to_zero<<<(cell_count_sum.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count_sum.size(), cell_count_sum.data());
  CUDA_CHECK_KERNEL

  set_to_zero<<<(cell_contents.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_contents.size(), cell_contents.data());
  CUDA_CHECK_KERNEL

  find_cell_counts<<<grid_size, block_size, 0, stream>>>(
    partition_direction, box, N, cell_count.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2],
    rc_inv);
  CUDA_CHECK_KERNEL

  thrust::exclusive_scan(
    thrust::cuda::par.on(stream), cell_count.data(), cell_count.data() + N_cells,
    cell_count_sum.data());

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  CUDA_CHECK_KERNEL

  find_cell_contents<<<grid_size, block_size, 0, stream>>>(
    partition_direction, box, N, cell_count.data(), cell_count_sum.data(), cell_contents.data(), x,
    y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv);
  CUDA_CHECK_KERNEL
}

static __global__ void find_neighbor_list_large_box(
  NEP3_MULTIGPU::ParaMB paramb,
  const int partition_direction,
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
    partition_direction, box, x1, y1, z1, 2.0f * paramb.rcinv_radial, nx, ny, nz, cell_id_x,
    cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int xxx = xx;
        int yyy = yy;
        int zzz = zz;
        if (cell_id_x + xx < 0)
          xxx += nx;
        if (cell_id_x + xx >= nx)
          xxx -= nx;
        if (cell_id_y + yy < 0)
          yyy += ny;
        if (cell_id_y + yy >= ny)
          yyy -= ny;
        if (cell_id_z + zz < 0)
          zzz += nz;
        if (cell_id_z + zz >= nz)
          zzz -= nz;

        int neighbor_cell = cell_id;
        if (partition_direction == 0) {
          neighbor_cell += (xxx * nz + zzz) * ny + yyy;
        } else if (partition_direction == 1) {
          neighbor_cell += (yyy * nz + zzz) * nx + xxx;
        } else {
          neighbor_cell += (zzz * ny + yyy) * nx + xxx;
        }

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n1 == n2) {
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
  NEP3_MULTIGPU::ParaMB paramb,
  NEP3_MULTIGPU::ANN annmb,
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
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  NEP3_MULTIGPU::ParaMB paramb,
  NEP3_MULTIGPU::ANN annmb,
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
    g_fx[n1] = s_fx;
    g_fy[n1] = s_fy;
    g_fz[n1] = s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] = s_sxx;
    g_virial[n1 + 1 * N] = s_syy;
    g_virial[n1 + 2 * N] = s_szz;
    g_virial[n1 + 3 * N] = s_sxy;
    g_virial[n1 + 4 * N] = s_sxz;
    g_virial[n1 + 5 * N] = s_syz;
    g_virial[n1 + 6 * N] = s_syx;
    g_virial[n1 + 7 * N] = s_szx;
    g_virial[n1 + 8 * N] = s_szy;
  }
}

static __global__ void find_partial_force_angular(
  NEP3_MULTIGPU::ParaMB paramb,
  NEP3_MULTIGPU::ANN annmb,
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
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
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
  const NEP3_MULTIGPU::ZBL zbl,
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

static __global__ void distribute_position(
  const int num_atoms_gobal,
  const int num_atoms_local,
  const int N1,
  const int N2,
  const int N3,
  const int M0,
  const int M1,
  const int M2,
  const int* cell_contents,
  const int* g_type_global,
  const double* g_position_global,
  int* g_type_local,
  double* g_position_local)
{
  int n_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_local < N3) {
    int n_global;
    if (n_local < N1) { // left
      n_global = cell_contents[n_local + M0];
    } else if (n_local < N2) { // middle
      n_global = cell_contents[n_local - N1 + M1];
    } else { // right
      n_global = cell_contents[n_local - N2 + M2];
    }

    g_type_local[n_local] = g_type_global[n_global];
    for (int d = 0; d < 3; ++d) {
      g_position_local[n_local + d * num_atoms_local] =
        g_position_global[n_global + d * num_atoms_gobal];
    }
  }
}

static __global__ void collect_properties(
  const int num_atoms_global,
  const int num_atoms_local,
  const int N1,
  const int N2,
  const int M1,
  const int* cell_contents,
  const double* g_force_local,
  const double* g_potential_local,
  const double* g_virial_local,
  double* g_force_global,
  double* g_potential_global,
  double* g_virial_global)
{
  int n_local = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n_local < N2) {
    int n_global = cell_contents[n_local - N1 + M1];
    for (int d = 0; d < 3; ++d) {
      g_force_global[n_global + d * num_atoms_global] =
        g_force_local[n_local + d * num_atoms_local];
    }
    g_potential_global[n_global] = g_potential_local[n_local];
    for (int d = 0; d < 9; ++d) {
      g_virial_global[n_global + d * num_atoms_global] =
        g_virial_local[n_local + d * num_atoms_local];
    }
  }
}

static __global__ void gpu_find_force_many_body(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f; // virial_stress_xx
  float s_sxy = 0.0f; // virial_stress_xy
  float s_sxz = 0.0f; // virial_stress_xz
  float s_syx = 0.0f; // virial_stress_yx
  float s_syy = 0.0f; // virial_stress_yy
  float s_syz = 0.0f; // virial_stress_yz
  float s_szx = 0.0f; // virial_stress_zx
  float s_szy = 0.0f; // virial_stress_zy
  float s_szz = 0.0f; // virial_stress_zz

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = offset * number_of_particles + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

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
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}

void NEP3_MULTIGPU::compute(
  const int group_method,
  std::vector<Group>& group,
  const int type_begin,
  const int type_end,
  const int type_shift,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int N = type.size();
  const double rc_cell_list = 0.5 * rc;
  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  int partition_direction = 2;
  int num_bins_longitudinal = num_bins[2] / paramb.num_gpus;
  int num_bins_transverse = num_bins[0] * num_bins[1];
  if (num_bins[0] > num_bins[1] && num_bins[0] > num_bins[2]) {
    partition_direction = 0;
    num_bins_longitudinal = num_bins[0] / paramb.num_gpus;
    num_bins_transverse = num_bins[1] * num_bins[2];
  }
  if (num_bins[1] > num_bins[0] && num_bins[1] > num_bins[2]) {
    partition_direction = 1;
    num_bins_longitudinal = num_bins[1] / paramb.num_gpus;
    num_bins_transverse = num_bins[0] * num_bins[2];
  }

  if (num_bins_longitudinal < 10) {
    printf("The longest direction has less than 5 times of the NEP cutoff per GPU.\n");
    printf("Please reduce the number of GPUs or increase the simulation cell size.\n");
    exit(1);
  }

  find_cell_list(
    nep_data[0].stream, partition_direction, rc_cell_list, num_bins, box, N, position,
    nep_temp_data.cell_count, nep_temp_data.cell_count_sum, nep_temp_data.cell_contents);
  nep_temp_data.cell_count_sum.copy_to_host(
    nep_temp_data.cell_count_sum_cpu.data(), num_bins[0] * num_bins[1] * num_bins[2]);

  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    if (paramb.num_gpus == 1) {
      nep_data[gpu].N1 = 0;
      nep_data[gpu].N4 = 0;
      nep_data[gpu].N2 = N;
      nep_data[gpu].N5 = N;
      nep_data[gpu].N3 = N;
      nep_data[gpu].M0 = 0;
      nep_data[gpu].M1 = 0;
      nep_data[gpu].M2 = N;
    } else {
      if (gpu == 0) {
        nep_data[gpu].M0 =
          nep_temp_data.cell_count_sum_cpu[(num_bins[2] - 4) * num_bins_transverse];
        nep_data[gpu].M1 = 0;
        nep_data[gpu].M2 =
          nep_temp_data.cell_count_sum_cpu[num_bins_longitudinal * num_bins_transverse];
        nep_data[gpu].N1 = N - nep_data[gpu].M0;
        nep_data[gpu].N4 =
          nep_temp_data.cell_count_sum_cpu[(num_bins[2] - 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N2 = nep_data[gpu].N1 + nep_data[gpu].M2;
        nep_data[gpu].N5 =
          nep_data[gpu].N1 +
          nep_temp_data.cell_count_sum_cpu[(num_bins_longitudinal + 2) * num_bins_transverse];
        nep_data[gpu].N3 =
          nep_data[gpu].N1 +
          nep_temp_data.cell_count_sum_cpu[(num_bins_longitudinal + 4) * num_bins_transverse];
      } else if (gpu == paramb.num_gpus - 1) {
        nep_data[gpu].M0 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal - 4) * num_bins_transverse];
        nep_data[gpu].M1 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal) * num_bins_transverse];
        nep_data[gpu].M2 = 0;
        nep_data[gpu].N1 = nep_data[gpu].M1 - nep_data[gpu].M0;
        nep_data[gpu].N4 =
          nep_temp_data
            .cell_count_sum_cpu[(gpu * num_bins_longitudinal - 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N2 = N - nep_data[gpu].M0;
        nep_data[gpu].N5 =
          nep_data[gpu].N2 + nep_temp_data.cell_count_sum_cpu[2 * num_bins_transverse];
        nep_data[gpu].N3 =
          nep_data[gpu].N2 + nep_temp_data.cell_count_sum_cpu[4 * num_bins_transverse];
      } else {
        nep_data[gpu].M0 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal - 4) * num_bins_transverse];
        nep_data[gpu].M1 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal) * num_bins_transverse];
        nep_data[gpu].M2 =
          nep_temp_data
            .cell_count_sum_cpu[((gpu + 1) * num_bins_longitudinal) * num_bins_transverse];
        nep_data[gpu].N1 = nep_data[gpu].M1 - nep_data[gpu].M0;
        nep_data[gpu].N4 =
          nep_temp_data
            .cell_count_sum_cpu[(gpu * num_bins_longitudinal - 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N2 = nep_data[gpu].M2 - nep_data[gpu].M0;
        nep_data[gpu].N5 =
          nep_temp_data
            .cell_count_sum_cpu[((gpu + 1) * num_bins_longitudinal + 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N3 =
          nep_temp_data
            .cell_count_sum_cpu[((gpu + 1) * num_bins_longitudinal + 4) * num_bins_transverse] -
          nep_data[gpu].M0;
      }
    }
    if (nep_data[gpu].N3 > nep_temp_data.num_atoms_per_gpu) {
      nep_temp_data.num_atoms_per_gpu = nep_data[gpu].N3 * 1.1;
      allocate_memory();
    }
  }

  // serial
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    distribute_position<<<(nep_data[gpu].N3 - 1) / 64 + 1, 64>>>(
      N, nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N1, nep_data[gpu].N2, nep_data[gpu].N3,
      nep_data[gpu].M0, nep_data[gpu].M1, nep_data[gpu].M2, nep_temp_data.cell_contents.data(),
      type.data(), position.data(), nep_temp_data.type.data(), nep_temp_data.position.data());
    CUDA_CHECK_KERNEL

    CHECK(cudaMemcpy(
      nep_data[gpu].type.data(), nep_temp_data.type.data(), sizeof(int) * nep_data[gpu].N3,
      cudaMemcpyDeviceToDevice));
    for (int d = 0; d < 3; ++d) {
      CHECK(cudaMemcpy(
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * d,
        nep_temp_data.position.data() + nep_temp_data.num_atoms_per_gpu * d,
        sizeof(double) * nep_data[gpu].N3, cudaMemcpyDeviceToDevice));
    }
  }

  // parallel
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
    CHECK(cudaSetDevice(gpu));
#endif

    find_cell_list(
      nep_data[gpu].stream, partition_direction, rc_cell_list, num_bins, box, nep_data[gpu].N3,
      nep_data[gpu].position, nep_data[gpu].cell_count, nep_data[gpu].cell_count_sum,
      nep_data[gpu].cell_contents);

    find_neighbor_list_large_box<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1, 64, 0, nep_data[gpu].stream>>>(
      paramb, partition_direction, nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N4,
      nep_data[gpu].N5, num_bins[0], num_bins[1], num_bins[2], box, nep_data[gpu].cell_count.data(),
      nep_data[gpu].cell_count_sum.data(), nep_data[gpu].cell_contents.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].NN_radial.data(), nep_data[gpu].NL_radial.data(),
      nep_data[gpu].NN_angular.data(), nep_data[gpu].NL_angular.data());
    CUDA_CHECK_KERNEL

    find_descriptor<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1, 64, 0, nep_data[gpu].stream>>>(
      paramb, annmb[gpu], nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N4, nep_data[gpu].N5, box,
      nep_data[gpu].NN_radial.data(), nep_data[gpu].NL_radial.data(),
      nep_data[gpu].NN_angular.data(), nep_data[gpu].NL_angular.data(), nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].potential.data(), nep_data[gpu].Fp.data(), nep_data[gpu].sum_fxyz.data());
    CUDA_CHECK_KERNEL

    find_force_radial<<<
      (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1, 64, 0, nep_data[gpu].stream>>>(
      paramb, annmb[gpu], nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N1, nep_data[gpu].N2, box,
      nep_data[gpu].NN_radial.data(), nep_data[gpu].NL_radial.data(), nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2, nep_data[gpu].Fp.data(),
      nep_data[gpu].force.data(), nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].virial.data());
    CUDA_CHECK_KERNEL

    find_partial_force_angular<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1, 64, 0, nep_data[gpu].stream>>>(
      paramb, annmb[gpu], nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N4, nep_data[gpu].N5, box,
      nep_data[gpu].NN_angular.data(), nep_data[gpu].NL_angular.data(), nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2, nep_data[gpu].Fp.data(),
      nep_data[gpu].sum_fxyz.data(), nep_data[gpu].f12x.data(), nep_data[gpu].f12y.data(),
      nep_data[gpu].f12z.data());
    CUDA_CHECK_KERNEL

    gpu_find_force_many_body<<<
      (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1, 64, 0, nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N1, nep_data[gpu].N2, box,
      nep_data[gpu].NN_angular.data(), nep_data[gpu].NL_angular.data(), nep_data[gpu].f12x.data(),
      nep_data[gpu].f12y.data(), nep_data[gpu].f12z.data(), nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].force.data(), nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].virial.data());
    CUDA_CHECK_KERNEL

    if (zbl.enabled) {
      find_force_ZBL<<<
        (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1, 64, 0, nep_data[gpu].stream>>>(
        nep_temp_data.num_atoms_per_gpu, zbl, nep_data[gpu].N1, nep_data[gpu].N2, box,
        nep_data[gpu].NN_angular.data(), nep_data[gpu].NL_angular.data(), nep_data[gpu].type.data(),
        nep_data[gpu].position.data(),
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
        nep_data[gpu].force.data(), nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
        nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
        nep_data[gpu].virial.data(), nep_data[gpu].potential.data());
      CUDA_CHECK_KERNEL
    }
  }

  CHECK(cudaSetDevice(0));

  // serial
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(cudaMemcpy(
      nep_temp_data.potential.data() + nep_data[gpu].N1,
      nep_data[gpu].potential.data() + nep_data[gpu].N1,
      sizeof(double) * (nep_data[gpu].N2 - nep_data[gpu].N1), cudaMemcpyDeviceToDevice));

    for (int d = 0; d < 3; ++d) {
      CHECK(cudaMemcpy(
        nep_temp_data.force.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        nep_data[gpu].force.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        sizeof(double) * (nep_data[gpu].N2 - nep_data[gpu].N1), cudaMemcpyDeviceToDevice));
    }

    for (int d = 0; d < 9; ++d) {
      CHECK(cudaMemcpy(
        nep_temp_data.virial.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        nep_data[gpu].virial.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        sizeof(double) * (nep_data[gpu].N2 - nep_data[gpu].N1), cudaMemcpyDeviceToDevice));
    }

    collect_properties<<<(nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1, 64>>>(
      N, nep_temp_data.num_atoms_per_gpu, nep_data[gpu].N1, nep_data[gpu].N2, nep_data[gpu].M1,
      nep_temp_data.cell_contents.data(), nep_temp_data.force.data(),
      nep_temp_data.potential.data(), nep_temp_data.virial.data(), force.data(), potential.data(),
      virial.data());
    CUDA_CHECK_KERNEL
  }
}
