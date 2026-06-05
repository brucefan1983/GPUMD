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

#ifdef USE_DEEPMD
#include "dp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <cmath>
#include <sstream>
#include <cstring>

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
  danger_list.resize(num_atoms);
  ghost_count.resize(num_atoms);
  ghost_sum.resize(num_atoms);
  danger_flag.resize(num_atoms);
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

namespace {
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
  const int ndanger,
  const int nghost,
  const int max_ghost_num_each_danger)
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
      for (int i = 0; i < max_ghost_num_each_danger; ++i) {
        int ghost_id = ghost_id_map[ghost_idx + ndanger * i];
        if (ghost_id != -1) {
          ghost_id -= N;
          if ((unsigned int) ghost_id < (unsigned int) nghost) {
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

static __host__ __device__ double get_dp_padding_fraction(
  const int pbc, const double rc, const double thickness)
{
  return (pbc == 1 && thickness > 0.0) ? rc / thickness : 0.0;
}

static __host__ __device__ void get_fractional_position(
  const Box& box,
  const double x,
  const double y,
  const double z,
  double& sx,
  double& sy,
  double& sz)
{
  sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;

  if (box.pbc_x == 1) sx -= floor(sx);
  if (box.pbc_y == 1) sy -= floor(sy);
  if (box.pbc_z == 1) sz -= floor(sz);
}

static __device__ void get_ghost_shift_bounds(
  const int pbc,
  const double s,
  const double padding,
  int& min_shift,
  int& max_shift)
{
  if (pbc == 1) {
    min_shift = static_cast<int>(ceil(-padding - s));
    max_shift = static_cast<int>(floor(1.0 + padding - s));
  } else {
    min_shift = 0;
    max_shift = 0;
  }
}

static __host__ int get_max_ghost_num_each_danger(const Box& box, const double rc)
{
  const double thickness[3] = {box.thickness_x, box.thickness_y, box.thickness_z};
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  int max_num_images = 1;
  for (int d = 0; d < 3; ++d) {
    int max_num_shifts = 1;
    if (pbc[d] == 1 && thickness[d] > 0.0) {
      const double padding = rc / thickness[d];
      max_num_shifts = static_cast<int>(floor(1.0 + 2.0 * padding)) + 2;
    }
    max_num_images *= max_num_shifts;
  }
  return max_num_images - 1;
}

static __host__ __device__ void get_padded_position(
  const Box& box,
  const double rc,
  const double x,
  const double y,
  const double z,
  const int shift_x,
  const int shift_y,
  const int shift_z,
  double& px,
  double& py,
  double& pz)
{
  const double padding_x = get_dp_padding_fraction(box.pbc_x, rc, box.thickness_x);
  const double padding_y = get_dp_padding_fraction(box.pbc_y, rc, box.thickness_y);
  const double padding_z = get_dp_padding_fraction(box.pbc_z, rc, box.thickness_z);
  const double offset_x = shift_x + padding_x;
  const double offset_y = shift_y + padding_y;
  const double offset_z = shift_z + padding_z;

  px = x + box.cpu_h[0] * offset_x + box.cpu_h[1] * offset_y + box.cpu_h[2] * offset_z;
  py = y + box.cpu_h[3] * offset_x + box.cpu_h[4] * offset_y + box.cpu_h[5] * offset_z;
  pz = z + box.cpu_h[6] * offset_x + box.cpu_h[7] * offset_y + box.cpu_h[8] * offset_z;
}

static void create_dp_ghost_box(const Box& box, const double rc, Box& box_ghost)
{
  const double padding_x = get_dp_padding_fraction(box.pbc_x, rc, box.thickness_x);
  const double padding_y = get_dp_padding_fraction(box.pbc_y, rc, box.thickness_y);
  const double padding_z = get_dp_padding_fraction(box.pbc_z, rc, box.thickness_z);
  const double scale_x = 1.0 + 2.0 * padding_x;
  const double scale_y = 1.0 + 2.0 * padding_y;
  const double scale_z = 1.0 + 2.0 * padding_z;

  box_ghost.pbc_x = 0;
  box_ghost.pbc_y = 0;
  box_ghost.pbc_z = 0;

  box_ghost.cpu_h[0] = box.cpu_h[0] * scale_x;
  box_ghost.cpu_h[3] = box.cpu_h[3] * scale_x;
  box_ghost.cpu_h[6] = box.cpu_h[6] * scale_x;

  box_ghost.cpu_h[1] = box.cpu_h[1] * scale_y;
  box_ghost.cpu_h[4] = box.cpu_h[4] * scale_y;
  box_ghost.cpu_h[7] = box.cpu_h[7] * scale_y;

  box_ghost.cpu_h[2] = box.cpu_h[2] * scale_z;
  box_ghost.cpu_h[5] = box.cpu_h[5] * scale_z;
  box_ghost.cpu_h[8] = box.cpu_h[8] * scale_z;

  box_ghost.get_inverse();
  box_ghost.set_is_orthogonal();
}

static void set_deepmd_box(const Box& box, std::vector<double>& dp_box)
{
  dp_box[0] = box.cpu_h[0];
  dp_box[1] = box.cpu_h[3];
  dp_box[2] = box.cpu_h[6];
  dp_box[3] = box.cpu_h[1];
  dp_box[4] = box.cpu_h[4];
  dp_box[5] = box.cpu_h[7];
  dp_box[6] = box.cpu_h[2];
  dp_box[7] = box.cpu_h[5];
  dp_box[8] = box.cpu_h[8];
}

static __global__ void calc_ghost_atom_number_each_atom(
  const int N,
  const double rc,
  const double* x,
  const double* y,
  const double* z,
  int* ghost_count,
  int* danger_flag,
  const Box box)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    double sx, sy, sz;
    get_fractional_position(box, x1, y1, z1, sx, sy, sz);

    const double padding_x = get_dp_padding_fraction(box.pbc_x, rc, box.thickness_x);
    const double padding_y = get_dp_padding_fraction(box.pbc_y, rc, box.thickness_y);
    const double padding_z = get_dp_padding_fraction(box.pbc_z, rc, box.thickness_z);
    int min_x, max_x, min_y, max_y, min_z, max_z;
    get_ghost_shift_bounds(box.pbc_x, sx, padding_x, min_x, max_x);
    get_ghost_shift_bounds(box.pbc_y, sy, padding_y, min_y, max_y);
    get_ghost_shift_bounds(box.pbc_z, sz, padding_z, min_z, max_z);

    const int nghost =
      (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1) - 1;
    ghost_count[n1] = nghost;
    danger_flag[n1] = nghost != 0;
  }
}

// this function calculates ghost atom number for each atom and then reduces on device
static int calc_ghost_atom_number(
  const int block_size,
  const int grid_size,
  const int N,
  const double rc,
  const double* position,
  int* ghost_count,
  int* danger_flag,
  const Box& box)
{
  calc_ghost_atom_number_each_atom<<<grid_size, block_size>>>(
    N,
    rc,
    position,
    position + N,
    position + 2 * N,
    ghost_count,
    danger_flag,
    box);
  GPU_CHECK_KERNEL

  return thrust::reduce(
    thrust::device,
    ghost_count,
    ghost_count + N,
    0,
    thrust::plus<int>());
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
  Box box,
  const int max_ghost_num_each_danger)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    int nall = N + nghost;
    int nall_2 = nall * 2;

    double px, py, pz;
    get_padded_position(box, rc, x1, y1, z1, 0, 0, 0, px, py, pz);
    dp_position[n1] = px;
    dp_position[n1 + nall] = py;
    dp_position[n1 + nall_2] = pz;
    type_ghost[n1] = type[n1];
    if (ghost_count[n1] == 0) {
      danger_list[n1] = -1;
      return;
      // TODO: may use less threads? use more memory to save messages
    }
    int ghost_id = N + ghost_sum[n1];
    int ghost_idx = danger_list[n1];

    double sx, sy, sz;
    get_fractional_position(box, x1, y1, z1, sx, sy, sz);
    const double padding_x = get_dp_padding_fraction(box.pbc_x, rc, box.thickness_x);
    const double padding_y = get_dp_padding_fraction(box.pbc_y, rc, box.thickness_y);
    const double padding_z = get_dp_padding_fraction(box.pbc_z, rc, box.thickness_z);
    int min_x, max_x, min_y, max_y, min_z, max_z;
    get_ghost_shift_bounds(box.pbc_x, sx, padding_x, min_x, max_x);
    get_ghost_shift_bounds(box.pbc_y, sy, padding_y, min_y, max_y);
    get_ghost_shift_bounds(box.pbc_z, sz, padding_z, min_z, max_z);

    int ghost_slot = 0;
    for (int iz = min_z; iz <= max_z; ++iz) {
      for (int iy = min_y; iy <= max_y; ++iy) {
        for (int ix = min_x; ix <= max_x; ++ix) {
          if (ix == 0 && iy == 0 && iz == 0) continue;
          if (ghost_slot < max_ghost_num_each_danger) {
            ghost_id_map[ghost_idx + ndanger * ghost_slot] = ghost_id;
          }
          type_ghost[ghost_id] = type[n1];
          get_padded_position(box, rc, x1, y1, z1, ix, iy, iz, px, py, pz);
          dp_position[ghost_id] = px;
          dp_position[ghost_id + nall] = py;
          dp_position[ghost_id + nall_2] = pz;
          ++ghost_id;
          ++ghost_slot;
        }
      }
    }
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
  if (number_of_atoms <= 0) return;
  dp_nl.inum = number_of_atoms;
  int grid_size = (number_of_atoms - 1) / BLOCK_SIZE_FORCE + 1;
  int num_bins_unused[3];
  box.get_num_bins(rc, num_bins_unused);
  const int max_ghost_num_each_danger = get_max_ghost_num_each_danger(box, rc);

  // get ghost atom number
nghost = calc_ghost_atom_number(
  BLOCK_SIZE_FORCE,
  grid_size,
  number_of_atoms,
  rc,
  position_per_atom.data(),
  ghost_count.data(),
  danger_flag.data(),
  box);

  thrust::exclusive_scan(
    thrust::device, ghost_count.data(), ghost_count.data() + number_of_atoms, ghost_sum.data());
  thrust::exclusive_scan(
    thrust::device, danger_flag.data(), danger_flag.data() + number_of_atoms, danger_list.data());

  ndanger = thrust::reduce(
    thrust::device,
    danger_flag.data(),
    danger_flag.data() + number_of_atoms,
    0,
    thrust::plus<int>());

  // check_ghost<<<grid_size, BLOCK_SIZE_FORCE>>>(ghost_count.data(), ghost_sum.data(), ghost_flag.data(), ghost_list.data(), number_of_atoms);
  // resize the ghost vectors
  int num_all_atoms = number_of_atoms + nghost; // all atoms include ghost atoms
  int grid_size_ghost = (num_all_atoms - 1) / BLOCK_SIZE_FORCE + 1;

  // Prevent ndanger == 0 from causing an error.
  if ( ndanger == 0 ) ghost_id_map.resize(1, -1);
  else ghost_id_map.resize(ndanger * max_ghost_num_each_danger, -1);

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
    box,
    max_ghost_num_each_danger);
  GPU_CHECK_KERNEL

  dp_data.NN.resize(num_all_atoms);
  dp_data.NL.resize(num_all_atoms * MAX_NEIGH_NUM_DP);
  dp_data.cell_contents.resize(num_all_atoms);
  dp_data.cell_count.resize(num_all_atoms);
  dp_data.cell_count_sum.resize(num_all_atoms);

  Box box_ghost;
  create_dp_ghost_box(box, rc, box_ghost);

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
  GPU_CHECK_KERNEL
  dp_position_cpu.resize(num_all_atoms * 3);
  dp_position_gpu_trans.copy_to_host(dp_position_cpu.data());
  type_cpu.resize(num_all_atoms);
  type_ghost.copy_to_host(type_cpu.data());

  // create dp box
  std::vector<double> dp_box(9, 0.0);
  set_deepmd_box(box_ghost, dp_box);

  dp_nl.ilist.resize(num_all_atoms, 0);
  dp_nl.numneigh.resize(num_all_atoms, 0);
  dp_nl.firstneigh.resize(num_all_atoms, nullptr);

  // Allocate lmp_ilist and lmp_numneigh
  dp_data.NN.copy_to_host(dp_nl.numneigh.data());
  int max_numneigh = 0;
  for (int i = 0; i < num_all_atoms; ++i) {
    if (dp_nl.numneigh[i] > max_numneigh) max_numneigh = dp_nl.numneigh[i];
  }
  if (max_numneigh > MAX_NEIGH_NUM_DP) {
    printf("Error: DP neighbor overflow. max_numneigh = %d, limit = %d\n", max_numneigh, MAX_NEIGH_NUM_DP);
    exit(1);
  }
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

  // Build atom mapping (local + ghost -> local index) for DeePMD-kit C++ API.
  // Required by .pt2 (AOTInductor/DPA4) models; harmless no-op for .pth (TorchScript/DPA2/DPA3).
  // NOTE: atom_mapping must outlive lmp_list usage in deep_pot.compute() below,
  // because set_mapping() stores a raw pointer — not a copy.
  std::vector<int> atom_mapping(num_all_atoms);
  for (int i = 0; i < number_of_atoms; ++i) atom_mapping[i] = i;
  if (nghost > 0) {
    std::vector<int> gc(number_of_atoms), gs(number_of_atoms);
    ghost_count.copy_to_host(gc.data());
    ghost_sum.copy_to_host(gs.data());
    for (int i = 0; i < number_of_atoms; ++i)
      for (int g = 0; g < gc[i]; ++g)
        atom_mapping[number_of_atoms + gs[i] + g] = i;
  }
  lmp_list.set_mapping(atom_mapping.data());


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
  if (nghost > 0) {
    f_ghost.resize(nghost * 3);
    v_ghost.resize(nghost * 9);
    f_ghost.copy_from_host(dp_force.data() + number_of_atoms * 3, nghost * 3);
    v_ghost.copy_from_host(dp_vir_atom.data() + number_of_atoms * 9, nghost * 9);
  }

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
    ndanger,
    nghost,
    max_ghost_num_each_danger);
  GPU_CHECK_KERNEL
}
#endif
