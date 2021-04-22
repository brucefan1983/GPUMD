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
#include "limt.cuh"
#include "lj.cuh"
#include "nep.cuh"
#include "potential.cuh"
#include "rebo_mos2.cuh"
#include "ri.cuh"
#include "sw.cuh"
#include "tersoff1988.cuh"
#include "tersoff1989.cuh"
#include "tersoff_mini.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "vashishta.cuh"
#include <vector>

#define BLOCK_SIZE 128

Force::Force(void)
{
  num_of_potentials = 0;
  rc_max = 0.0;
  group_method = -1;
}

void Force::parse_potential(
  char** param,
  int num_param,
  char* input_dir,
  const Box& box,
  const Neighbor& neighbor,
  const std::vector<int>& cpu_type,
  const std::vector<int>& cpu_type_size)
{
  // check for at least the file path
  if (num_param < 2) {
    PRINT_INPUT_ERROR("potential should have at least 1 parameter.\n");
  }
  strcpy(file_potential[num_of_potentials], param[1]);

  // open file to check number of types used in potential
  char potential_name[20];
  FILE* fid_potential = my_fopen(file_potential[num_of_potentials], "r");
  int count = fscanf(fid_potential, "%s", potential_name);
  PRINT_SCANF_ERROR(count, 1, "Reading error for number of types.");
  int num_types = get_number_of_types(fid_potential);
  fclose(fid_potential);

  if (strcmp(potential_name, "lj") == 0) {
    is_lj[num_of_potentials] = true;
    if (num_param == 3) {
      if (!is_valid_int(param[2], &group_method)) {
        PRINT_INPUT_ERROR("Group method for LJ potential should be an integer.\n");
      }
    }
    atom_begin[num_of_potentials] = 0;
    atom_end[num_of_potentials] = num_types - 1;
  } else {
    is_lj[num_of_potentials] = false;
    if (num_param != num_types + 2) {
      PRINT_INPUT_ERROR("potential has incorrect number of types defined.\n");
    }

    std::vector<int> atom_type(num_types);

    for (int i = 0; i < num_types; i++) {
      if (!is_valid_int(param[i + 2], &atom_type[i])) {
        PRINT_INPUT_ERROR("type should be an integer.\n");
      }
      if (i != 0 && atom_type[i] < atom_type[i - 1]) {
        PRINT_INPUT_ERROR("potential types must be in ascending order.\n");
      }
    }
    atom_begin[num_of_potentials] = atom_type[0];
    atom_end[num_of_potentials] = atom_type[num_types - 1];

    if (atom_type[num_types - 1] - atom_type[0] + 1 > num_types) {
      PRINT_INPUT_ERROR("Error: types for one potential must be contiguous.\n");
    }
  }

  num_of_potentials++;

  add_potential(input_dir, box, neighbor, cpu_type, cpu_type_size);
}

int Force::get_number_of_types(FILE* fid_potential)
{
  int num_of_types;
  int count = fscanf(fid_potential, "%d", &num_of_types);
  PRINT_SCANF_ERROR(count, 1, "Reading error for number of types.");
  return num_of_types;
}

void Force::initialize_potential(
  char* input_dir,
  const Box& box,
  const Neighbor& neighbor,
  const std::vector<int>& cpu_type_size,
  const int m)
{
  const int number_of_atoms = neighbor.NN.size();
  FILE* fid_potential = my_fopen(file_potential[m], "r");
  char potential_name[20];
  int count = fscanf(fid_potential, "%s", potential_name);
  if (count != 1) {
    PRINT_INPUT_ERROR("reading error for potential file.");
  }

  int num_types = get_number_of_types(fid_potential);

  // determine the potential
  if (strcmp(potential_name, "tersoff_1989") == 0) {
    potential[m].reset(new Tersoff1989(fid_potential, num_types, neighbor));
  } else if (strcmp(potential_name, "tersoff_1988") == 0) {
    potential[m].reset(new Tersoff1988(fid_potential, num_types, neighbor));
  } else if (strcmp(potential_name, "tersoff_mini") == 0) {
    potential[m].reset(new Tersoff_mini(fid_potential, num_types, neighbor));
  } else if (strcmp(potential_name, "limt") == 0) {
    potential[m].reset(new LIMT(fid_potential, num_types, neighbor));
  } else if (strcmp(potential_name, "sw_1985") == 0) {
    potential[m].reset(new SW2(fid_potential, num_types, neighbor));
  } else if (strcmp(potential_name, "rebo_mos2") == 0) {
    potential[m].reset(new REBO_MOS(neighbor));
  } else if (strcmp(potential_name, "eam_zhou_2004") == 0) {
    potential[m].reset(new EAM(fid_potential, potential_name, number_of_atoms));
  } else if (strcmp(potential_name, "eam_dai_2006") == 0) {
    potential[m].reset(new EAM(fid_potential, potential_name, number_of_atoms));
  } else if (strcmp(potential_name, "vashishta") == 0) {
    potential[m].reset(new Vashishta(fid_potential, neighbor));
  } else if (strcmp(potential_name, "fcp") == 0) {
    potential[m].reset(new FCP(fid_potential, input_dir, number_of_atoms, box));
  } else if (strcmp(potential_name, "nep") == 0) {
    potential[m].reset(new NEP2(fid_potential, neighbor));
  } else if (strcmp(potential_name, "lj") == 0) {
    potential[m].reset(new LJ(fid_potential, num_types));
  } else if (strcmp(potential_name, "ri") == 0) {
    potential[m].reset(new RI(fid_potential));
  } else {
    PRINT_INPUT_ERROR("illegal potential model.\n");
  }

  potential[m]->N1 = 0;
  potential[m]->N2 = 0;

  for (int n = 0; n < atom_begin[m]; ++n) {
    potential[m]->N1 += cpu_type_size[n];
  }
  for (int n = 0; n <= atom_end[m]; ++n) {
    potential[m]->N2 += cpu_type_size[n];
  }

  printf(
    "    applies to atoms [%d, %d) from type %d to type %d.\n", potential[m]->N1, potential[m]->N2,
    atom_begin[m], atom_end[m]);

  fclose(fid_potential);
}

void Force::add_potential(
  char* input_dir,
  const Box& box,
  const Neighbor& neighbor,
  const std::vector<int>& cpu_type,
  const std::vector<int>& cpu_type_size)
{
  int m = num_of_potentials - 1; // current potential ID
  initialize_potential(input_dir, box, neighbor, cpu_type_size, m);

  if (rc_max < potential[m]->rc)
    rc_max = potential[m]->rc;

  // check the atom types in xyz.in
  for (int n = potential[m]->N1; n < potential[m]->N2; ++n) {
    if (cpu_type[n] < atom_begin[m] || cpu_type[n] > atom_end[m]) {
      printf("ERROR: type for potential # %d not from %d to %d.", m, atom_begin[m], atom_end[m]);
      exit(1);
    }
  }
  type_shift_[m] = atom_begin[m];
}

// Construct the local neighbor list from the global one (Kernel)
static __global__ void gpu_find_neighbor_local(
  const bool use_group,
  int* group_label,
  Box box,
  int type_begin,
  int type_end,
  int* type,
  int N,
  int N1,
  int N2,
  double cutoff_square,
  int* NN,
  int* NL,
  int* NN_local,
  int* NL_local,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  int layer_n1;

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = NN[n1];
    if (use_group)
      layer_n1 = group_label[n1];
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = NL[n1 + N * i1];

      if (use_group) {
        if (layer_n1 == group_label[n2])
          continue;
      }

      // only include neighbors with the correct types
      int type_n2 = type[n2];
      if (type_n2 < type_begin || type_n2 > type_end)
        continue;

      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double z12 = z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (distance_square < cutoff_square) {
        NL_local[count * N + n1] = n2;
        ++count;
      }
    }
    NN_local[n1] = count;
  }
}

// Construct the local neighbor list from the global one (Wrapper)
void Force::find_neighbor_local(
  const int m,
  std::vector<Group>& group,
  GPU_Vector<int>& atom_type,
  const GPU_Vector<double>& position_per_atom,
  const Box& box,
  Neighbor& neighbor)
{
  const int number_of_atoms = neighbor.NN.size();
  int grid_size = (potential[m]->N2 - potential[m]->N1 - 1) / 128 + 1;

  const bool use_group = is_lj[m] && (group_method > -1);
  int* group_label = nullptr;
  if (use_group)
    group_label = group[group_method].label.data();

  gpu_find_neighbor_local<<<grid_size, 128>>>(
    use_group, group_label, box, atom_begin[m], atom_end[m], atom_type.data(), number_of_atoms,
    potential[m]->N1, potential[m]->N2, potential[m]->rc * potential[m]->rc, neighbor.NN.data(),
    neighbor.NL.data(), neighbor.NN_local.data(), neighbor.NL_local.data(),
    position_per_atom.data(), position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + number_of_atoms * 2);
  CUDA_CHECK_KERNEL
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

void Force::compute(
  const Box& box,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  Neighbor& neighbor,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, force_per_atom.data(), force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2, potential_per_atom.data(), virial_per_atom.data());
  CUDA_CHECK_KERNEL

  for (int m = 0; m < num_of_potentials; m++) {
    // first build a local neighbor list
#ifndef USE_FCP // the FCP does not use a neighbor list at all
    find_neighbor_local(m, group, type, position_per_atom, box, neighbor);
#endif
    // and then calculate the forces and related quantities
    potential[m]->compute(
      type_shift_[m], box, neighbor, type, position_per_atom, potential_per_atom, force_per_atom,
      virial_per_atom);
  }

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
#ifdef USE_FCP
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
#endif
}
