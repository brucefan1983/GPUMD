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
The class dealing with the Deep Potential(DP).
------------------------------------------------------------------------------*/


#include "dp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <sstream>


#define BLOCK_SIZE_FORCE 128
#define MAX_NEIGH_NUM_DP 512    // max neighbor number of an atom for DP



DP::DP(const char* filename_dp, int num_atoms)
{
  // DP setting
  set_dp_coeff();

  // init DP from potential file
  initialize_dp(filename_dp);


  dp_data.NN.resize(num_atoms);
  dp_data.NL.resize(num_atoms * MAX_NEIGH_NUM_DP); // the largest supported by CUDA
  dp_data.cell_count.resize(num_atoms);
  dp_data.cell_count_sum.resize(num_atoms);
  dp_data.cell_contents.resize(num_atoms);
  type_cpu.resize(num_atoms);
  e_f_v_gpu.resize(num_atoms * (1 + 3 + 9));    // energy: 1; force: 3; virial: 9

  // init dp neighbor list
  dp_nl.inum = num_atoms;
  dp_nl.ilist.resize(num_atoms);
  dp_nl.numneigh.resize(num_atoms);
  dp_nl.firstneigh.resize(num_atoms);
  dp_nl.neigh_storage.resize(num_atoms * MAX_NEIGH_NUM_DP);
  
  // init ghost lists
  ghost_list.resize(num_atoms);
  ghost_count.resize(num_atoms);
  ghost_sum.resize(num_atoms);
  ghost_flag.resize(num_atoms);

  // init dp nghost temporary vector
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;
  nghost_tmp.resize(grid_size);
  

}


void DP::initialize_dp(const char* filename_dp)
{
  int num_gpus;
  CHECK(gpuGetDeviceCount(&num_gpus));
  printf("\nInitialize deep potential by the file: %s and %d gpu(s).\n\n", filename_dp, num_gpus);
  deep_pot.init(filename_dp, num_gpus);
  rc = deep_pot.cutoff();
  int numb_types = deep_pot.numb_types();
  int numb_types_spin = deep_pot.numb_types_spin();
  int dim_fparam = deep_pot.dim_fparam();
  int dim_aparam = deep_pot.dim_aparam();

  char* type_map[numb_types];
  std::string type_map_str;
  deep_pot.get_type_map(type_map_str);
  // convert the string to a vector of strings
  std::istringstream iss(type_map_str);
  std::string type_name;
  int i = 0;
  while (iss >> type_name) {
    if (i >= numb_types) break;
    type_map[i] = strdup(type_name.c_str());
    i++;
  }

  printf("=======================================================\n");
  printf("  ++ cutoff: %f ++ \n", rc);
  printf("  ++ numb_types: %d ++ \n", numb_types);
  printf("  ++ numb_types_spin: %d ++ \n", numb_types_spin);
  printf("  ++ dim_fparam: %d ++ \n", dim_fparam);
  printf("  ++ dim_aparam: %d ++ \n  ++ ", dim_aparam);
  for (int i = 0; i < numb_types; ++i)
  {
    printf("%s ", type_map[i]);
  }
  printf("++\n=======================================================\n");
}

DP::~DP(void)
{
  // none
}

void DP::set_dp_coeff(void) {
  ener_unit_cvt_factor=1;      // 1.0 / 8.617343e-5;
  dist_unit_cvt_factor=1;      // gpumd: angstrom, dp: angstrom;
  force_unit_cvt_factor=ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor=1;    // ener_unit_cvt_factor
  single_model = true;
  atom_spin_flag = false;
}


static __global__ void create_dp_position(
  const double* gpumd_position,
  double* dp_position,
  int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    dp_position[n1 * 3] = gpumd_position[n1];
    dp_position[n1 * 3 + 1] = gpumd_position[n1 + N];
    dp_position[n1 * 3 + 2] = gpumd_position[n1 + 2 * N];
  }
}


// force and virial need transpose from dp to gpumd
// TODO: use share memory to speed up
static __global__ void transpose_and_update_unit(
  const double* e_f_v_in,
  double* e_out,
  double* f_out,
  double* v_out,
  double e_factor,
  double f_factor,
  double v_factor,
  const int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    const int f_in_offset = N;
    const int v_in_offset = N * 4;
    e_out[n1] = e_f_v_in[n1] * e_factor;
    f_out[n1] = e_f_v_in[f_in_offset + n1 * 3] * f_factor;                // fx
    f_out[n1 + N] = e_f_v_in[f_in_offset + n1 * 3 + 1] * f_factor;        // fy
    f_out[n1 + N * 2] = e_f_v_in[f_in_offset + n1 * 3 + 2] * f_factor;    // fz
    // virial
    v_out[n1] = e_f_v_in[v_in_offset + n1 * 9] * v_factor;
    v_out[n1 + N] = e_f_v_in[v_in_offset + n1 * 9 + 4] * v_factor;
    v_out[n1 + N * 2] = e_f_v_in[v_in_offset + n1 * 9 + 8] * v_factor;
    v_out[n1 + N * 3] = e_f_v_in[v_in_offset + n1 * 9 + 3] * v_factor;
    v_out[n1 + N * 4] = e_f_v_in[v_in_offset + n1 * 9 + 6] * v_factor;
    v_out[n1 + N * 5] = e_f_v_in[v_in_offset + n1 * 9 + 7] * v_factor;
    v_out[n1 + N * 6] = e_f_v_in[v_in_offset + n1 * 9 + 1] * v_factor;
    v_out[n1 + N * 7] = e_f_v_in[v_in_offset + n1 * 9 + 2] * v_factor;
    v_out[n1 + N * 8] = e_f_v_in[v_in_offset + n1 * 9 + 5] * v_factor;
  }
}


static __device__ void warp_reduce(volatile int* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

static __global__ void calc_ghost_atom_number_each_block(
  const int N,
  const double rc,
  const double* x,
  const double* y,
  const double* z,
  int* ghost_count,
  int* ghost_flag,
  int* nghost_tmp,
  const Box& box)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  int tid = threadIdx.x;
  extern __shared__ int nghost_block[];
  // init shared memory
  nghost_block[tid] = 0;
  if (n1 < N) {
    int nghost = 0;
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    if (box.triclinic == 0) {
      // orthogonal box
      if (box.pbc_x == 1 && (x1 < rc || x1 > box.cpu_h[3] - rc)) {
        ++nghost;
      }
      if (box.pbc_y == 1 && (y1 < rc || y1 > box.cpu_h[4] - rc)) {
        ++nghost;
      }
      if (box.pbc_z == 1 && (z1 < rc || z1 > box.cpu_h[5] - rc)) {
        ++nghost;
      }
    } else {
      // triclinic box
      // TODO
      printf("TODO: triclinc box\n");
      return;
    }
    nghost_block[tid] = nghost;
    ghost_count[n1] = nghost;
    ghost_flag[n1] = nghost != 0;
    
  }
  __syncthreads();

  // reduce
  for (int s = blockDim.x >> 1; s > 32; s >>= 1) {
    if (tid < s) {
      nghost_block[tid] += nghost_block[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warp_reduce(nghost_block, tid);
  }

  // save to nghost_tmp
  if (tid == 0) {
    nghost_tmp[blockIdx.x] = nghost_block[0];
  }
  
}

static __global__ void reduce_nghost(int* idata, int* odata) {
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = i < gridDim.x ? idata[i] : 0;
  __syncthreads();

  // reduce
  for (int s = blockDim.x >> 1; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid < 32) {
    warp_reduce(sdata, tid);
  }

  // save to nghost_tmp
  if (tid == 0) {
    odata[blockIdx.x] = sdata[0];
  }
}


// this function has two step to calculate ghost atom number
// step 1: calculate nghost for each atom and reduce in each block
// step 2: reduce the nghost_tmp to get the total nghost
static int calc_ghost_atom_number(
  const int block_size,
  const int grid_size,
  const int N,
  const double rc,
  const double* position,
  int* ghost_count,
  int* ghost_flag,
  GPU_Vector<int>& nghost_tmp,
  const Box& box)
{
  calc_ghost_atom_number_each_block<<<grid_size, block_size, block_size * sizeof(int)>>>(
    N,
    rc,
    position,
    position + N,
    position + 2 * N,
    ghost_count,
    ghost_flag,
    nghost_tmp.data(),
    box);
  GPU_CHECK_KERNEL

  int nghost = 0;
  if (grid_size == 1) {
    nghost_tmp.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128 atoms
  int new_grid_size = (grid_size - 1) / block_size + 1;
  GPU_Vector<int> tmp1(new_grid_size);
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    nghost_tmp.data(),
    tmp1.data());
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    tmp1.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128x128 atoms
  new_grid_size = (new_grid_size - 1) / block_size + 1;
  GPU_Vector<int> tmp2(new_grid_size);
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    tmp1.data(),
    tmp2.data());
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    int nghost = 0;
    tmp2.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128x128x128 atoms
  new_grid_size = (new_grid_size - 1) / block_size + 1;
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    tmp2.data(),
    tmp1.data());
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    int nghost = 0;
    tmp1.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128x128x128x128 atoms
  new_grid_size = (new_grid_size - 1) / block_size + 1;
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    tmp1.data(),
    tmp2.data());
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    int nghost = 0;
    tmp2.copy_to_host(&nghost, 1);
    return nghost;
  }

  printf("\nTO MANY ATOMS!!!\n\n");
  return 0;


}


static __global__ void create_ghost_map(
  const int N,
  const int nghost,
  const double rc,
  const int* ghost_count,
  const int* ghost_sum,
  int* ghost_list,
  int* ghost_id_map,
  int* type_ghost,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  double* dp_position,
  Box& box)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];

    if (ghost_count[n1] == 0) {
      ghost_list[n1] = 0;
      return;
      // TODO: may use less threads? use more memory to save messages
    }
    int ghost_id = N + ghost_sum[n1];
    int ghost_idx = ghost_list[n1];
    type_ghost[ghost_idx] = type[n1];
    if (box.triclinic == 0) {
      // orthogonal box
      if (box.pbc_x == 1 && (x1 < rc || x1 > box.cpu_h[3] - rc)) {
        ghost_id_map[ghost_idx] = ghost_id++;
        dp_position[ghost_idx * 3] = x1 < rc ? x1 + box.cpu_h[3] : x1 - box.cpu_h[3];
        dp_position[ghost_idx * 3 + 1] = y1;
        dp_position[ghost_idx * 3 + 2] = z1;
      }
      if (box.pbc_y == 1 && (y1 < rc || y1 > box.cpu_h[4] - rc)) {
        ghost_id_map[ghost_idx + nghost] = ghost_id++;
        dp_position[ghost_idx * 3] = x1;
        dp_position[ghost_idx * 3 + 1] = y1 < rc ? y1 + box.cpu_h[4] : y1 - box.cpu_h[4];
        dp_position[ghost_idx * 3 + 2] = z1;
      }
      if (box.pbc_z == 1 && (z1 < rc || z1 > box.cpu_h[5] - rc)) {
        ghost_id_map[ghost_idx + nghost * 2] = ghost_id++;
        dp_position[ghost_idx * 3] = x1;
        dp_position[ghost_idx * 3 + 1] = y1;
        dp_position[ghost_idx * 3 + 2] = z1 < rc ? z1 + box.cpu_h[5] : z1 - box.cpu_h[5];
      }
    } else {
      // triclinic box
      // TODO
      printf("TODO: triclinc box\n");
      return;
    }


  }

}


static __global__ void gpu_find_neighbor_ON1_dp(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              if (d2 < cutoff_square) {
                NL[count++ * N + n1] = n2;
              }
            }
          }
        }
      }
    }
    NN[n1] = count;
  }
}


static void find_neighbor_dp(
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL,
  GPU_Vector<double>& dp_position_gpu,
  int* ghost_id_map,
  int* ghost_list,
  int* type_ghost)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_dp<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}


void DP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

  // get ghost atom number
  nghost = calc_ghost_atom_number(
    BLOCK_SIZE_FORCE,
    grid_size,
    number_of_atoms,
    rc,
    position_per_atom.data(),
    ghost_count.data(),
    ghost_flag.data(),
    nghost_tmp,
    box);

  thrust::exclusive_scan(
    thrust::device, ghost_count.data(), ghost_count.data() + number_of_atoms, ghost_sum.data());

  thrust::exclusive_scan(
    thrust::device, ghost_flag.data(), ghost_flag.data() + number_of_atoms, ghost_list.data());

  // resize the ghost vectors
  int num_all_atoms = number_of_atoms + nghost; // all atoms include ghost atoms
  ghost_id_map.resize(nghost * 3, -1);
  type_ghost.resize(nghost);
  dp_position_gpu.resize(num_all_atoms * 3);
  create_ghost_map<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    nghost,
    rc,
    ghost_count.data(),
    ghost_sum.data(),
    ghost_list.data(),
    ghost_id_map.data(),
    type_ghost.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    dp_position_gpu.data() + number_of_atoms * 3,
    box);
  GPU_CHECK_KERNEL



#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor_dp(
      N1,
      N2,
      rc,
      box,
      type,
      position_per_atom,
      dp_data.cell_count,
      dp_data.cell_count_sum,
      dp_data.cell_contents,
      dp_data.NN,
      dp_data.NL,
      dp_position_gpu,
      ghost_id_map.data(),
      ghost_list.data(),
      type_ghost.data());
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  // create dp position from gpumd
  create_dp_position<<<grid_size, BLOCK_SIZE_FORCE>>>(
    position_per_atom.data(),
    dp_position_gpu.data(),
    number_of_atoms
  );
  GPU_CHECK_KERNEL

  // Initialize DeepPot computation variables
  std::vector<double> dp_ene_all(1, 0.0);
  std::vector<double> dp_ene_atom(number_of_atoms, 0.0);
  std::vector<double> dp_force(number_of_atoms * 3, 0.0);
  std::vector<double> dp_vir_all(9, 0.0);
  std::vector<double> dp_vir_atom(number_of_atoms * 9, 0.0);


  // copy position and type to CPU
  std::vector<double> dp_position_cpu(number_of_atoms * 3);
  dp_position_gpu.copy_to_host(dp_position_cpu.data());
  // TODO: BUG! argument list does not match, because type is const int?
  // type.copy_to_host(type_cpu.data(), number_of_atoms);
  CHECK(gpuMemcpy(type_cpu.data(), type.data(), number_of_atoms * sizeof(int), gpuMemcpyDeviceToHost));


  // create dp box
  std::vector<double> dp_box(9, 0.0);
  if (box.triclinic == 0) {
    dp_box[0] = box.cpu_h[0];
    dp_box[4] = box.cpu_h[1];
    dp_box[8] = box.cpu_h[2];
  } else {
    dp_box[0] = box.cpu_h[0];
    dp_box[4] = box.cpu_h[1];
    dp_box[8] = box.cpu_h[2];
    dp_box[7] = box.cpu_h[7];
    dp_box[6] = box.cpu_h[6];
    dp_box[3] = box.cpu_h[3];
  }

  // Allocate lmp_ilist and lmp_numneigh
  dp_data.NN.copy_to_host(dp_nl.numneigh.data());
  // gpuMemcpy(lmp_numneigh, deepmd_ghost_data.NN.data(), num_of_all_atoms*sizeof(int), gpuMemcpyDeviceToHost);
  std::vector<int> cpu_NL(dp_data.NL.size());
  dp_data.NL.copy_to_host(cpu_NL.data());
  // gpuMemcpy(cpu_NL.data(), deepmd_ghost_data.NL.data(), total_all_neighs * sizeof(int), gpuMemcpyDeviceToHost);

  int offset = 0;
  for (int i = 0; i < number_of_atoms; ++i) {
    dp_nl.ilist[i] = i;
    dp_nl.firstneigh[i] = dp_nl.neigh_storage.data() + offset;
    for (int j = 0; j < dp_nl.numneigh[i]; ++j) {
        dp_nl.neigh_storage[offset + j] = cpu_NL[i + j * number_of_atoms]; // Copy in column-major order
    }
    offset += dp_nl.numneigh[i];
  }

  // Constructing a neighbor list in LAMMPS format
  // deepmd_compat::InputNlist lmp_list(num_of_all_atoms, lmp_ilist, lmp_numneigh, lmp_firstneigh);
  deepmd_compat::InputNlist lmp_list(dp_nl.inum, dp_nl.ilist.data(), dp_nl.numneigh.data(), dp_nl.firstneigh.data());



  // to calculate the atomic force and energy from deepot
  if (single_model) {
    if (! atom_spin_flag) {
        //deep_pot.compute(dp_ene_all, dp_force, dp_vir_all,dp_cpu_ghost_position, gpumd_cpu_ghost_type,dp_box);
        deep_pot.compute(dp_ene_all, dp_force, dp_vir_all, dp_ene_atom, dp_vir_atom, 
            dp_position_cpu, type_cpu, dp_box,
            0, lmp_list, 0);
    }
  }


  // copy dp output energy, force, and virial to gpu
  size_t size_tmp = number_of_atoms; // size of number_of_atom * 1 in double
  // memory distribution of e_f_v_gpu: e1, e2 ... en, fx1, fy1, fz1, fx2 ... fzn, vxx1 ...
  e_f_v_gpu.copy_from_host(dp_ene_atom.data(), size_tmp, 0);
  e_f_v_gpu.copy_from_host(dp_force.data(), size_tmp * 3, number_of_atoms);
  e_f_v_gpu.copy_from_host(dp_vir_atom.data(), size_tmp * 9, number_of_atoms * 4);

  // std::vector<double> gpumd_ene_atom(number_of_atoms, 0.0);
  // std::vector<double> gpumd_force(number_of_atoms * 3, 0.0);
  // std::vector<double> virial_per_atom_cpu(number_of_atoms * 9, 0.0);
  // const int const_cell = half_const_cell * 2 + 1;
  // for (int g = 0; g < number_of_atoms; g++) {
  //   gpumd_ene_atom[g] += dp_ene_atom[i * real_num_of_atoms + g] * ener_unit_cvt_factor;
  //   for (int o = 0; o < 3; o++)
  //     gpumd_force.data()[g + o * real_num_of_atoms] += dp_force.data()[3*g+o] * force_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 0 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 0] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 1 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 4] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 2 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 8] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 3 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 3] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 4 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 6] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 5 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 7] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 6 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 1] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 7 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 2] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 8 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 5] * virial_unit_cvt_factor;
  // }
  transpose_and_update_unit<<<grid_size, BLOCK_SIZE_FORCE>>>(
    e_f_v_gpu.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    ener_unit_cvt_factor,
    force_unit_cvt_factor,
    virial_unit_cvt_factor,
    number_of_atoms
  );
  GPU_CHECK_KERNEL

}