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

#ifdef USE_TENSORFLOW
#include "dp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <sstream>

#define BLOCK_SIZE_FORCE 128
#define MAX_NEIGH_NUM_DP 512    // max neighbor number of an atom for DP
#define MAX_GHOST_NUM_EACH_DANGER 7

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
  danger_list.resize(num_atoms);
  ghost_count.resize(num_atoms);
  ghost_sum.resize(num_atoms);
  danger_flag.resize(num_atoms);

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

  printf("---------------------------------------------------------------\n");
  printf("  ++ cutoff: %f ++ \n", rc);
  printf("  ++ numb_types: %d ++ \n", numb_types);
  printf("  ++ numb_types_spin: %d ++ \n", numb_types_spin);
  printf("  ++ dim_fparam: %d ++ \n", dim_fparam);
  printf("  ++ dim_aparam: %d ++ \n  ++ ", dim_aparam);
  for (int i = 0; i < numb_types; ++i)
  {
    printf("%s ", type_map[i]);
  }
  printf("++\n---------------------------------------------------------------\n");
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

static __global__ void dp_position_transpose(
  const double* position,
  double* position_trans,
  int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    position_trans[n1 * 3] = position[n1];
    position_trans[n1 * 3 + 1] = position[n1 + N];
    position_trans[n1 * 3 + 2] = position[n1 + 2 * N];
  }
}

// force and virial need transpose from dp to gpumd
// TODO: use share memory to speed up
static __global__ void transpose_and_update_unit(
  const double* e_f_v_in,
  double* e_out,
  double* f_out,
  double* v_out,
  double* f_ghost_in,
  double* v_ghost_in,
  int* danger_list,
  int* ghost_id_map,
  double e_factor,
  double f_factor,
  double v_factor,
  const int N,
  const int ndanger)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    const int f_in_offset = N;
    const int v_in_offset = N * 4;
    e_out[n1] = e_f_v_in[n1] * e_factor;

    double fx = e_f_v_in[f_in_offset + n1 * 3];
    double fy = e_f_v_in[f_in_offset + n1 * 3 + 1];
    double fz = e_f_v_in[f_in_offset + n1 * 3 + 2];

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    double vxx = e_f_v_in[v_in_offset + n1 * 9] * v_factor;
    double vyy = e_f_v_in[v_in_offset + n1 * 9 + 4] * v_factor;
    double vzz = e_f_v_in[v_in_offset + n1 * 9 + 8] * v_factor;
    double vxy = e_f_v_in[v_in_offset + n1 * 9 + 3] * v_factor;
    double vxz = e_f_v_in[v_in_offset + n1 * 9 + 6] * v_factor;
    double vyz = e_f_v_in[v_in_offset + n1 * 9 + 7] * v_factor;
    double vyx = e_f_v_in[v_in_offset + n1 * 9 + 1] * v_factor;
    double vzx = e_f_v_in[v_in_offset + n1 * 9 + 2] * v_factor;
    double vzy = e_f_v_in[v_in_offset + n1 * 9 + 5] * v_factor;
    int ghost_idx = danger_list[n1];
    if (ghost_idx != -1) {
      for (int i = 0; i < MAX_GHOST_NUM_EACH_DANGER; ++i) {
        int ghost_id = ghost_id_map[ghost_idx + ndanger * i];
        if (ghost_id != -1) {
          ghost_id -= N;
          fx += f_ghost_in[ghost_id * 3];
          fy += f_ghost_in[ghost_id * 3 + 1];
          fz += f_ghost_in[ghost_id * 3 + 2];

          vxx +=  v_ghost_in[ghost_id * 9];
          vyy +=  v_ghost_in[ghost_id * 9 + 4];
          vzz +=  v_ghost_in[ghost_id * 9 + 8];
          vxy +=  v_ghost_in[ghost_id * 9 + 3];
          vxz +=  v_ghost_in[ghost_id * 9 + 6];
          vyz +=  v_ghost_in[ghost_id * 9 + 7];
          vyx +=  v_ghost_in[ghost_id * 9 + 1];
          vzx +=  v_ghost_in[ghost_id * 9 + 2];
          vzy +=  v_ghost_in[ghost_id * 9 + 5];
        }
      }
    }
    f_out[n1] = fx * f_factor;            // fx
    f_out[n1 + N] = fy * f_factor;        // fy
    f_out[n1 + N * 2] = fz * f_factor;    // fz

    v_out[n1] = vxx * v_factor;
    v_out[n1 + N] = vyy * v_factor;
    v_out[n1 + N * 2] = vzz * v_factor;
    v_out[n1 + N * 3] = vxy * v_factor;
    v_out[n1 + N * 4] = vxz * v_factor;
    v_out[n1 + N * 5] = vyz * v_factor;
    v_out[n1 + N * 6] = vyx * v_factor;
    v_out[n1 + N * 7] = vzx * v_factor;
    v_out[n1 + N * 8] = vzy * v_factor;
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
  int* danger_flag,
  int* nghost_tmp,
  const Box box)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  int tid = threadIdx.x;
  extern __shared__ int nghost_block[];
  // init shared memory
  nghost_block[tid] = 0;
  if (n1 < N) {
    int nghost = 1;
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    if (box.triclinic == 0) {
      // orthogonal box
      if (box.pbc_x == 1 && (x1 < rc || x1 > box.cpu_h[0] - rc)) {
        nghost <<= 1;
      }
      if (box.pbc_y == 1 && (y1 < rc || y1 > box.cpu_h[1] - rc)) {
        nghost <<= 1;
      }
      if (box.pbc_z == 1 && (z1 < rc || z1 > box.cpu_h[2] - rc)) {
        nghost <<= 1;
      }
    } else {
      // triclinic box
      // TODO
      printf("TODO: triclinc box\n");
      return;
    }
    // ghost boudary | nghost
    // x, y, z       | 1
    // xy, xz, yz    | 3
    // xyz           | 7
    --nghost;
    nghost_block[tid] = nghost;
    ghost_count[n1] = nghost;
    danger_flag[n1] = nghost != 0;
    
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

static __global__ void reduce_nghost(int* idata, int* odata, int N) {
  extern __shared__ int sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = i < N ? idata[i] : 0;
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
  int* danger_flag,
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
    danger_flag,
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
  int old_grid_size = grid_size;
  GPU_Vector<int> tmp1(new_grid_size);
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    nghost_tmp.data(),
    tmp1.data(),
    old_grid_size);
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    tmp1.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128x128 atoms
  old_grid_size = new_grid_size;
  new_grid_size = (new_grid_size - 1) / block_size + 1;
  GPU_Vector<int> tmp2(new_grid_size);
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    tmp1.data(),
    tmp2.data(),
    old_grid_size);
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    int nghost = 0;
    tmp2.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128x128x128 atoms
  old_grid_size = new_grid_size;
  new_grid_size = (new_grid_size - 1) / block_size + 1;
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    tmp2.data(),
    tmp1.data(),
    old_grid_size);
  GPU_CHECK_KERNEL

  if (new_grid_size == 1) {
    int nghost = 0;
    tmp1.copy_to_host(&nghost, 1);
    return nghost;
  }

  // more than 128x128x128x128 atoms
  old_grid_size = new_grid_size;
  new_grid_size = (new_grid_size - 1) / block_size + 1;
  reduce_nghost<<<new_grid_size, block_size, block_size * sizeof(int)>>>(
    tmp1.data(),
    tmp2.data(),
    old_grid_size);
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
  const int ndanger,
  const double rc,
  const int* ghost_count,
  const int* ghost_sum,
  int* danger_list,
  int* ghost_id_map,
  int* type_ghost,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  double* dp_position,
  Box box)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    int nall = N + nghost;
    int nall_2 = nall * 2;

    dp_position[n1] = x1;
    dp_position[n1 + nall] = y1;
    dp_position[n1 + nall_2] = z1;
    type_ghost[n1] = type[n1];
    if (ghost_count[n1] == 0) {
      danger_list[n1] = -1;
      return;
      // TODO: may use less threads? use more memory to save messages
    }
    int ghost_id = N + ghost_sum[n1];
    int ghost_idx = danger_list[n1];
    int ghost_x_flag = 0;
    int ghost_y_flag = 0;
    if (box.triclinic == 0) {
      // orthogonal box
      if (box.pbc_x == 1 && (x1 < rc || x1 > box.cpu_h[0] - rc)) {
        // x
        ghost_x_flag = 1;
        ghost_id_map[ghost_idx + ndanger * GHOST_X] = ghost_id;
        type_ghost[ghost_id] = type[n1];
        dp_position[ghost_id] = x1 < rc ? x1 + box.cpu_h[0] : x1 - box.cpu_h[0];
        dp_position[ghost_id + nall] = y1;
        dp_position[ghost_id + nall_2] = z1;
        ++ghost_id;
      }

      if (box.pbc_y == 1 && (y1 < rc || y1 > box.cpu_h[1] - rc)) {
        // y
        ghost_y_flag = 1;
        ghost_id_map[ghost_idx + ndanger * GHOST_Y] = ghost_id;
        type_ghost[ghost_id] = type[n1];
        dp_position[ghost_id] = x1;
        dp_position[ghost_id + nall] = y1 < rc ? y1 + box.cpu_h[1] : y1 - box.cpu_h[1];
        dp_position[ghost_id + nall_2] = z1;
        ++ghost_id;

        if (ghost_x_flag == 1) {
          // xy
          ghost_id_map[ghost_idx + ndanger * GHOST_XY] = ghost_id;
          type_ghost[ghost_id] = type[n1];
          dp_position[ghost_id] = x1 < rc ? x1 + box.cpu_h[0] : x1 - box.cpu_h[0];
          dp_position[ghost_id + nall] = y1 < rc ? y1 + box.cpu_h[1] : y1 - box.cpu_h[1];
          dp_position[ghost_id + nall_2] = z1;
          ++ghost_id;
        }
      }

      if (box.pbc_z == 1 && (z1 < rc || z1 > box.cpu_h[2] - rc)) {
        // z
        ghost_id_map[ghost_idx + ndanger * GHOST_Z] = ghost_id;
        type_ghost[ghost_id] = type[n1];
        dp_position[ghost_id] = x1;
        dp_position[ghost_id + nall] = y1;
        dp_position[ghost_id + nall_2] = z1 < rc ? z1 + box.cpu_h[2] : z1 - box.cpu_h[2];
        ++ghost_id;

        if (ghost_x_flag == 1) {
          // xz
          ghost_id_map[ghost_idx + ndanger * GHOST_XZ] = ghost_id;
          type_ghost[ghost_id] = type[n1];
          dp_position[ghost_id] = x1 < rc ? x1 + box.cpu_h[0] : x1 - box.cpu_h[0];
          dp_position[ghost_id + nall] = y1;
          dp_position[ghost_id + nall_2] = z1 < rc ? z1 + box.cpu_h[2] : z1 - box.cpu_h[2];
          ++ghost_id;

          if (ghost_y_flag == 1) {
            // xyz
            ghost_id_map[ghost_idx + ndanger * GHOST_XYZ] = ghost_id;
            type_ghost[ghost_id] = type[n1];
            dp_position[ghost_id] = x1 < rc ? x1 + box.cpu_h[0] : x1 - box.cpu_h[0];
            dp_position[ghost_id + nall] = y1 < rc ? y1 + box.cpu_h[1] : y1 - box.cpu_h[1];
            dp_position[ghost_id + nall_2] = z1 < rc ? z1 + box.cpu_h[2] : z1 - box.cpu_h[2];
            ++ghost_id;
          }
        }

        if (ghost_y_flag == 1) {
          // yz
          ghost_id_map[ghost_idx + ndanger * GHOST_YZ] = ghost_id;
          type_ghost[ghost_id] = type[n1];
          dp_position[ghost_id] = x1;
          dp_position[ghost_id + nall] = y1 < rc ? y1 + box.cpu_h[1] : y1 - box.cpu_h[1];
          dp_position[ghost_id + nall_2] = z1 < rc ? z1 + box.cpu_h[2] : z1 - box.cpu_h[2];
          ++ghost_id;
        }
      }
    } else {
      // triclinic box
      // TODO
      printf("TODO: triclinc box\n");
      return;
    }
  }
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
    danger_flag.data(),
    nghost_tmp,
    box);

  thrust::exclusive_scan(
    thrust::device, ghost_count.data(), ghost_count.data() + number_of_atoms, ghost_sum.data());
  thrust::exclusive_scan(
    thrust::device, danger_flag.data(), danger_flag.data() + number_of_atoms, danger_list.data());

  // get the number of dangerous atoms from the last number of danger_list
  int last_atom_danger_flag = 0;
  danger_flag.copy_to_host(&last_atom_danger_flag, 1, number_of_atoms - 1);
  danger_list.copy_to_host(&ndanger, 1, number_of_atoms - 1);
  ndanger += last_atom_danger_flag;

  // check_ghost<<<grid_size, BLOCK_SIZE_FORCE>>>(ghost_count.data(), ghost_sum.data(), ghost_flag.data(), ghost_list.data(), number_of_atoms);
  // resize the ghost vectors
  int num_all_atoms = number_of_atoms + nghost; // all atoms include ghost atoms
  int grid_size_ghost = (num_all_atoms - 1) / BLOCK_SIZE_FORCE + 1;
  ghost_id_map.resize(ndanger * 7, -1);
  type_ghost.resize(num_all_atoms);
  dp_position_gpu.resize(num_all_atoms * 3);
  
  create_ghost_map<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms,
    nghost,
    ndanger,
    rc,
    ghost_count.data(),
    ghost_sum.data(),
    danger_list.data(),
    ghost_id_map.data(),
    type_ghost.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2,
    dp_position_gpu.data(),
    box);
  GPU_CHECK_KERNEL

  dp_data.NN.resize(num_all_atoms);
  dp_data.NL.resize(num_all_atoms * MAX_NEIGH_NUM_DP);
  dp_data.cell_contents.resize(num_all_atoms);
  dp_data.cell_count.resize(num_all_atoms);
  dp_data.cell_count_sum.resize(num_all_atoms);

  Box box_ghost;
  box_ghost.pbc_x = 0;
  box_ghost.pbc_y = 0;
  box_ghost.pbc_z = 0;
  // TODO: triclinic
  // TODO: use periodic box when find neigh
  box_ghost.triclinic = box.triclinic;
  box_ghost.cpu_h[0] = box.cpu_h[0] + box.pbc_x ? 2 * rc : 0;
  box_ghost.cpu_h[1] = box.cpu_h[1] + box.pbc_y ? 2 * rc : 0;
  box_ghost.cpu_h[2] = box.cpu_h[2] + box.pbc_z ? 2 * rc : 0;

  find_neighbor(
    N1,
    num_all_atoms,
    rc,
    box_ghost,
    type_ghost,
    dp_position_gpu,
    dp_data.cell_count,
    dp_data.cell_count_sum,
    dp_data.cell_contents,
    dp_data.NN,
    dp_data.NL);

  // Initialize DeepPot computation variables
  dp_ene_all.resize(1, 0.0);
  dp_ene_atom.resize(num_all_atoms, 0.0);
  dp_force.resize(num_all_atoms * 3, 0.0);
  dp_vir_all.resize(9, 0.0);
  dp_vir_atom.resize(num_all_atoms * 9, 0.0);

  // copy position and type to CPU
  dp_position_gpu_trans.resize(num_all_atoms * 3);
  dp_position_transpose<<<grid_size_ghost, BLOCK_SIZE_FORCE>>>(
    dp_position_gpu.data(),
    dp_position_gpu_trans.data(),
    num_all_atoms);
  dp_position_cpu.resize(num_all_atoms * 3);
  dp_position_gpu_trans.copy_to_host(dp_position_cpu.data());
  type_cpu.resize(num_all_atoms);
  type_ghost.copy_to_host(type_cpu.data());

  // create dp box
  std::vector<double> dp_box(9, 0.0);
  if (box.triclinic == 0) {
    dp_box[0] = box.cpu_h[0] + box.pbc_x ? 2 * rc : 0;
    dp_box[4] = box.cpu_h[1] + box.pbc_y ? 2 * rc : 0;
    dp_box[8] = box.cpu_h[2] + box.pbc_z ? 2 * rc : 0;
  } else {
    dp_box[0] = box.cpu_h[0];
    dp_box[4] = box.cpu_h[1];
    dp_box[8] = box.cpu_h[2];
    dp_box[7] = box.cpu_h[7];
    dp_box[6] = box.cpu_h[6];
    dp_box[3] = box.cpu_h[3];
  }

  dp_nl.ilist.resize(num_all_atoms, 0);
  dp_nl.numneigh.resize(num_all_atoms, 0);
  dp_nl.firstneigh.resize(num_all_atoms, nullptr);

  // Allocate lmp_ilist and lmp_numneigh
  dp_data.NN.copy_to_host(dp_nl.numneigh.data());
  cpu_NL.resize(dp_data.NL.size());
  dp_data.NL.copy_to_host(cpu_NL.data());

  int offset = 0;
  dp_nl.neigh_storage.resize(dp_data.NL.size());
  for (int i = 0; i < num_all_atoms; ++i) {
    dp_nl.ilist[i] = i;
    dp_nl.firstneigh[i] = dp_nl.neigh_storage.data() + offset;
    for (int j = 0; j < dp_nl.numneigh[i]; ++j) {
        dp_nl.neigh_storage[offset + j] = cpu_NL[i + j * num_all_atoms]; // Copy in column-major order
    }
    offset += dp_nl.numneigh[i];
  }

  // Constructing a neighbor list in LAMMPS format
  // inum: number of local atoms
  // the neighbor list record the message of ghost atoms, so len(numneigh) = nlocal + nghost 
  // deepmd_compat::InputNlist lmp_list(nlocal, lmp_ilist, lmp_numneigh, lmp_firstneigh);
  deepmd_compat::InputNlist lmp_list(dp_nl.inum, dp_nl.ilist.data(), dp_nl.numneigh.data(), dp_nl.firstneigh.data());

  // to calculate the atomic force and energy from deepot
  if (single_model) {
    if (! atom_spin_flag) {
        deep_pot.compute(dp_ene_all, dp_force, dp_vir_all, dp_ene_atom, dp_vir_atom, 
            dp_position_cpu, type_cpu, dp_box,
            nghost, lmp_list, 0);
    }
  }

  // copy dp output energy, force, and virial to gpu
  // memory distribution of e_f_v_gpu: e1, e2 ... en, fx1, fy1, fz1, fx2 ... fzn, vxx1 ...
  e_f_v_gpu.copy_from_host(dp_ene_atom.data(), number_of_atoms, 0);
  e_f_v_gpu.copy_from_host(dp_force.data(), number_of_atoms * 3, number_of_atoms);
  e_f_v_gpu.copy_from_host(dp_vir_atom.data(), number_of_atoms * 9, number_of_atoms * 4);
  
  // copy ghost atom force and virial to modify the local atoms' force and virial
  f_ghost.resize(nghost * 3);
  v_ghost.resize(nghost * 9);
  f_ghost.copy_from_host(dp_force.data() + number_of_atoms * 3, nghost * 3);
  v_ghost.copy_from_host(dp_vir_atom.data() + number_of_atoms * 9, nghost * 9);

  // transpose dp vectors
  transpose_and_update_unit<<<grid_size, BLOCK_SIZE_FORCE>>>(
    e_f_v_gpu.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    f_ghost.data(),
    v_ghost.data(),
    danger_list.data(),
    ghost_id_map.data(),
    ener_unit_cvt_factor,
    force_unit_cvt_factor,
    virial_unit_cvt_factor,
    number_of_atoms,
    ndanger);
  GPU_CHECK_KERNEL
}
#endif
