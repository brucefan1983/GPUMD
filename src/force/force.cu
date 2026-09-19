/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version. GPUMD is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have
   received a copy of the GNU General Public License along with GPUMD.  If not, see
   <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The driver class calculating force and related quantities.
------------------------------------------------------------------------------*/

#ifdef USE_DEEPMD
#include "dp.cuh"
#endif
#ifdef USE_NNAP
#include "nnap.cuh"
#endif
#include "adp.cuh"
#include "eam.cuh"
#include "eam_alloy.cuh"
#include "fcp.cuh"
#include "force.cuh"
#include "ilp_nep.cuh"
#include "ilp_tmd_sw.cuh"
#include "ilp_tersoff.cuh"
#include "lj.cuh"
#include "nep.cuh"
#include "nep_multigpu.cuh"
#include "nep_charge.cuh"
#include "potential.cuh"
#include "tersoff1988.cuh"
#include "tersoff1989.cuh"
#include "tersoff_mini.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <cstring>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 128

Force::Force(void)
{
  is_fcp = false;
  has_non_nep = false;
}

void Force::check_types(const std::string& file_potential)
{
  std::ifstream input(file_potential);
  std::vector<std::string> tokens = get_tokens(input);
  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  for (int n = 0; n < num_types; ++n) {
    std::string token = tokens[2 + n];
    if (potentials.size() == 0) {
      atom_types[n] = token;
    } else {
      if (token != atom_types[n]) {
        PRINT_INPUT_ERROR(
          "The atomic species and/or the order of the species are not consistent "
          "between the multiple potentials.\n");
      }
    }
  }
}

std::unique_ptr<Potential> Force::create_potential(
  const std::vector<std::string>& tokens,
  FILE* fid_potential,
  char* potential_name,
  const int num_types,
  const Box& box,
  const int number_of_atoms,
  const RunInput& run_input,
  bool& is_nep)
{
  const int num_param = tokens.size();
  std::unique_ptr<Potential> potential;

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
  } else if (strcmp(potential_name, "eam/alloy") == 0) {
    int max_neigh = 400;
    if (num_param == 3) {
      if (!is_valid_int(tokens[2], &max_neigh) || max_neigh <= 0 || max_neigh > 1024) {
        PRINT_INPUT_ERROR(
          "max_neighbor for eam/alloy must be a positive integer in (0, 1024].");
      }
    }
    potential.reset(new EAMAlloy(tokens[1].c_str(), number_of_atoms, max_neigh));
  } else if (strcmp(potential_name, "adp") == 0) {
    potential.reset(new ADP(tokens[1].c_str(), number_of_atoms));
  } else if (strcmp(potential_name, "fcp") == 0) {
    potential.reset(new FCP(fid_potential, num_types, number_of_atoms, box));
    is_fcp = true;
  } else if (
    strcmp(potential_name, "nep4_charge1") == 0 ||
    strcmp(potential_name, "nep4_charge2") == 0 ||
    strcmp(potential_name, "nep4_charge3") == 0 ||
    strcmp(potential_name, "nep4_zbl_charge1") == 0 ||
    strcmp(potential_name, "nep4_zbl_charge2") == 0 ||
    strcmp(potential_name, "nep4_zbl_charge3") == 0) {
    potential.reset(new NEP_Charge(tokens[1].c_str(), number_of_atoms, run_input));
    is_nep = true;
  } else if (
    strcmp(potential_name, "nep4") == 0 || strcmp(potential_name, "nep4_zbl") == 0 ||
    strcmp(potential_name, "nep4_temperature") == 0 ||
    strcmp(potential_name, "nep4_zbl_temperature") == 0) {
    int num_gpus;
    CHECK(gpuGetDeviceCount(&num_gpus));
#ifdef ZHEYONG
    num_gpus = 3;
#endif
    if (num_gpus == 1) {
      potential.reset(new NEP(tokens[1].c_str(), number_of_atoms, run_input));
    } else {
      int partition_direction = -1;
      if (num_param == 3) {
        if (tokens[2] == "x") {
          partition_direction = 0;
        } else if (tokens[2] == "y") {
          partition_direction = 1;
        } else if (tokens[2] == "z") {
          partition_direction = 2;
        } else {
          PRINT_INPUT_ERROR("partition direction for multi-GPU NEP can only be x or y or z.\n");
        }
      }
      potential.reset(
        new NEP_MULTIGPU(
          num_gpus, tokens[1].c_str(), number_of_atoms, partition_direction, run_input));
    }
    is_nep = true;
#ifdef USE_DEEPMD
  } else if (strcmp(potential_name, "dp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR(
        "The potential command should contain two parameters, the setting file and the DP potential file.\n");
    }
    potential.reset(new DP(tokens[2].c_str(), number_of_atoms));
#endif
#ifdef USE_NNAP
  } else if (strcmp(potential_name, "nnap") == 0 || strcmp(potential_name, "nnap_zbl") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR(
        "The potential command should contain two parameters, the setting file and the NNAP potential file.\n");
    }
    potential.reset(new NNAP(tokens[1].c_str(), tokens[2].c_str(), number_of_atoms));
#endif
  } else if (strcmp(potential_name, "lj") == 0) {
    potential.reset(new LJ(fid_potential, num_types, number_of_atoms));
  } else if (strcmp(potential_name, "nep_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain an ILP potential file and a NEP map file.\n");
    }
    FILE* fid_nep_map = my_fopen(tokens[2].c_str(), "r");
    potential.reset(new ILP_NEP(fid_potential, fid_nep_map, num_types, number_of_atoms));
    fclose(fid_nep_map);
  } else if (strcmp(potential_name, "tersoff_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain ILP potential file and Tersoff potential file.\n");
    }
    FILE* fid_tersoff = my_fopen(tokens[2].c_str(), "r");
    potential.reset(new ILP_TERSOFF(fid_potential, fid_tersoff, num_types, number_of_atoms));
    fclose(fid_tersoff);
  } else if (strcmp(potential_name, "sw_ilp") == 0) {
    if (num_param != 3) {
      PRINT_INPUT_ERROR("potential should contain ILP potential file and SW potential file.\n");
    }
    FILE* fid_sw = my_fopen(tokens[2].c_str(), "r");
    potential.reset(new ILP_TMD_SW(fid_potential, fid_sw, num_types, number_of_atoms));
    fclose(fid_sw);
  } else {
    PRINT_INPUT_ERROR("illegal potential model.\n");
  }

  return potential;
}

void Force::parse_potential(
  const std::vector<std::string>& tokens,
  const Box& box,
  const int number_of_atoms,
  const RunInput& run_input)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("potential should have 1 or 2 parameters.\n");
  }

  FILE* fid_potential = my_fopen(tokens[1].c_str(), "r");
  char potential_name[100];
  int count = fscanf(fid_potential, "%s", potential_name);
  if (count != 1) {
    PRINT_INPUT_ERROR("reading error for potential file.");
  }
  int num_types = get_number_of_types(fid_potential);
  number_of_atoms_ = number_of_atoms;
  bool is_nep = false;
  std::unique_ptr<Potential> potential = create_potential(
    tokens,
    fid_potential,
    potential_name,
    num_types,
    box,
    number_of_atoms,
    run_input,
    is_nep);

  if (is_nep) {
    // Check if the types for this potential are compatible with the possibly other potentials
    check_types(tokens[1]);
  }
  fclose(fid_potential);

  potential->N1 = 0;
  potential->N2 = number_of_atoms;

  // Move the pointer into the list of potentials
  potentials.push_back(std::move(potential));
  // Check if a non-NEP potential has previously been defined
  has_non_nep = has_non_nep || !is_nep;
  if (potentials.size() > 1 && has_non_nep) {
    PRINT_INPUT_ERROR("Multiple potentials may only be used with NEP potentials.\n");
  }
}

int Force::get_number_of_types(FILE* fid_potential)
{
  int num_of_types;
  int count = fscanf(fid_potential, "%d", &num_of_types);
  PRINT_SCANF_ERROR(count, 1, "Reading error for number of types.");
  return num_of_types;
}

// get the total force
static __global__ void gpu_sum_force(int N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_f[1024];
  double f = 0.0;

  switch (bid) {
    case 0:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        int n = tid + batch * 1024;
        if (n < N)
          f += g_fx[n];
      }
      break;
    case 1:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        int n = tid + batch * 1024;
        if (n < N)
          f += g_fy[n];
      }
      break;
    case 2:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        int n = tid + batch * 1024;
        if (n < N)
          f += g_fz[n];
      }
      break;
  }
  s_f[tid] = f;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
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

void Force::finalize()
{
  compute_hnemdec_ = -1;
  multiple_potentials_mode_ = "observe";
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
  if (compute_hnemdec_ >= 0) {
    PRINT_INPUT_ERROR("Cannot have more than one HNEMD method within one run.");
  }

  int N = mass.size();
  int number_of_types = type_size.size();
  compute_hnemdec_ = compute_hnemdec;
  temperature = T;

  double total_mass = 0;
  std::vector<double> cpu_coefficient;
  std::vector<double> mass_type;
  mass_type.resize(number_of_types);
  int find_mass_type = 0;
  for (int i = 0; i < N; i++) {
    if (mass_type[type[i]] != mass[i]) {
      mass_type[type[i]] = mass[i];
      find_mass_type += 1;
    }
    total_mass += mass[i];
  }
  if (find_mass_type != number_of_types) {
    PRINT_INPUT_ERROR("mass type and element type do not match.\n");
  }

  // find atom types' fraction
  if (compute_hnemdec_ == 0) {
    cpu_coefficient.resize(number_of_types * 2);
    coefficient.resize(number_of_types * 2);
    const size_t tensor_size = static_cast<size_t>(N) * 9;
    if (hnemdec_tensor_per_atom_.size() != tensor_size) {
      hnemdec_tensor_per_atom_.resize(tensor_size);
    }
    if (hnemdec_tensor_sum_.size() != 9) {
      hnemdec_tensor_sum_.resize(9);
    }

    for (int i = 0; i < number_of_types; i++) {
      double c_hv = (total_mass - N * mass_type[i]) / total_mass;
      cpu_coefficient[i * 2] = (c_hv - 1) / N;
      cpu_coefficient[i * 2 + 1] = K_B * temperature * c_hv;
    }

    coefficient.copy_from_host(cpu_coefficient.data());
  } else if ((compute_hnemdec_ > 0) && (compute_hnemdec_ <= number_of_types)) {
    int element_index = compute_hnemdec_ - 1;
    cpu_coefficient.resize(number_of_types);
    cpu_coefficient[element_index] = double(N) / type_size[element_index];
    double partial_mass = 0;
    for (int i = 0; i < number_of_types; i++) {
      if (i != element_index) {
        partial_mass += mass_type[i] * type_size[i];
      }
    }
    for (int i = 0; i < number_of_types; i++) {
      if (i != element_index) {
        cpu_coefficient[i] = -1 * N * mass_type[i] / partial_mass;
      }
    }
    coefficient.resize(number_of_types);
    coefficient.copy_from_host(cpu_coefficient.data());
  }

  hnemd_fe_[0] = hnemd_fe_x;
  hnemd_fe_[1] = hnemd_fe_y;
  hnemd_fe_[2] = hnemd_fe_z;
}

static __global__ void gpu_apply_pbc(
  int N, Box box, double* g_x, double* g_y, double* g_z, int* g_position_image)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    if (box.pbc_x == 1) {
      if (sx < 0.0) {
        sx += 1.0;
        if (g_position_image != nullptr)
          g_position_image[n]--;
      } else if (sx > 1.0) {
        sx -= 1.0;
        if (g_position_image != nullptr)
          g_position_image[n]++;
      }
    }
    if (box.pbc_y == 1) {
      if (sy < 0.0) {
        sy += 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N]--;
      } else if (sy > 1.0) {
        sy -= 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N]++;
      }
    }
    if (box.pbc_z == 1) {
      if (sz < 0.0) {
        sz += 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N * 2]--;
      } else if (sz > 1.0) {
        sz -= 1.0;
        if (g_position_image != nullptr)
          g_position_image[n + N * 2]++;
      }
    }
    g_x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
    g_y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
    g_z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
  }
}

static __global__ void gpu_average_properties(
  int N, double* g_potential, double* g_force, double* g_virial, double denominator)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_potential[n1] /= denominator;
    g_force[n1 + 0 * N] /= denominator;
    g_force[n1 + 1 * N] /= denominator;
    g_force[n1 + 2 * N] /= denominator;
    g_virial[n1 + 0 * N] /= denominator;
    g_virial[n1 + 1 * N] /= denominator;
    g_virial[n1 + 2 * N] /= denominator;
    g_virial[n1 + 3 * N] /= denominator;
    g_virial[n1 + 4 * N] /= denominator;
    g_virial[n1 + 5 * N] /= denominator;
    g_virial[n1 + 6 * N] /= denominator;
    g_virial[n1 + 7 * N] /= denominator;
    g_virial[n1 + 8 * N] /= denominator;
  }
}

void Force::set_multiple_potentials_mode(std::string mode) { multiple_potentials_mode_ = mode; }

void Force::set_temperature_range(
  const double temperature1, const double temperature2, const int number_of_steps)
{
  temperature = temperature1;
  delta_T = (temperature2 - temperature1) / number_of_steps;
}

void Force::advance_temperature() { temperature += delta_T; }

int Force::get_number_of_potentials() const { return potentials.size(); }

Potential& Force::get_potential(const int index) { return *potentials[index]; }

void Force::prepare_compute(
  const int number_of_atoms,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  int* position_image)
{
  box.set_is_orthogonal();

  if (!is_fcp) {
    gpu_apply_pbc<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      box,
      position_per_atom.data(),
      position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2,
      position_image);
  }

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + number_of_atoms * 2,
    potential_per_atom.data(),
    virial_per_atom.data());
  GPU_CHECK_KERNEL
}

void Force::compute_single_potential(
  Potential& potential,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (3 == potential.nep_model_type) {
    potential.compute(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else if (1 == potential.ilp_flag) {
    potential.compute_ilp(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom, group);
  } else {
    potential.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

void Force::compute_potentials(
  const int number_of_atoms,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (multiple_potentials_mode_.compare("observe") == 0) {
    // If observing, calculate using main potential only
    compute_single_potential(
      *potentials[0],
      box,
      position_per_atom,
      type,
      group,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else if (multiple_potentials_mode_.compare("average") == 0) {
    // Calculate average potential, force and virial per atom.
    for (int i = 0; i < potentials.size(); i++) {
      // potential->compute automatically adds the properties
      compute_single_potential(
        *potentials[i],
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);
    }
    // Compute average and copy properties back into original vectors.
    gpu_average_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      potential_per_atom.data(),
      force_per_atom.data(),
      virial_per_atom.data(),
      (double)potentials.size());
    GPU_CHECK_KERNEL
  } else {
    PRINT_INPUT_ERROR("Invalid mode for multiple potentials.\n");
  }
}

void Force::correct_fcp_force(
  const int number_of_atoms, GPU_Vector<double>& force_per_atom)
{
  // always correct the force when using the FCP potential
  if (is_fcp) {
    GPU_Vector<double> ftot(3); // total force vector of the system
    gpu_sum_force<<<3, 1024>>>(
      number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    GPU_CHECK_KERNEL

    gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      1.0 / number_of_atoms,
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms,
      ftot.data());
    GPU_CHECK_KERNEL
  }
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();
  prepare_compute(
    number_of_atoms,
    box,
    position_per_atom,
    potential_per_atom,
    force_per_atom,
    virial_per_atom,
    nullptr);
  compute_potentials(
    number_of_atoms,
    box,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

  correct_fcp_force(number_of_atoms, force_per_atom);
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
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_t[1024];
  double t = 0.0;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N)
      t += g_tensor[bid * N + n];
  }
  s_t[tid] = t;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_t[tid] += s_t[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_sum_tensor[bid] = s_t[0];
  }
}

static __global__ void gpu_add_driving_force(
  int N,
  const double* g_coefficient,
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int type2 = g_type[i] * 2;
    double coefficient1 = g_coefficient[type2];
    double coefficient2 = g_coefficient[type2 + 1];
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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double coefficient = g_coefficient[g_type[i]];
    g_fx[i] += fe_x * coefficient;
    g_fy[i] += fe_y * coefficient;
    g_fz[i] += fe_z * coefficient;
  }
}

void Force::apply_hnemdec(
  const int number_of_atoms,
  GPU_Vector<int>& type,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& mass_per_atom)
{
  if (compute_hnemdec_ == 0) {
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    gpu_find_per_atom_tensor<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      mass_per_atom.data(),
      potential_per_atom.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms,
      velocity_per_atom.data() + 2 * number_of_atoms,
      virial_per_atom.data() + 0 * number_of_atoms,
      virial_per_atom.data() + 3 * number_of_atoms,
      virial_per_atom.data() + 4 * number_of_atoms,
      virial_per_atom.data() + 6 * number_of_atoms,
      virial_per_atom.data() + 1 * number_of_atoms,
      virial_per_atom.data() + 5 * number_of_atoms,
      virial_per_atom.data() + 7 * number_of_atoms,
      virial_per_atom.data() + 8 * number_of_atoms,
      virial_per_atom.data() + 2 * number_of_atoms,
      hnemdec_tensor_per_atom_.data());
    GPU_CHECK_KERNEL

    gpu_sum_tensor<<<9, 1024>>>(
      number_of_atoms, hnemdec_tensor_per_atom_.data(), hnemdec_tensor_sum_.data());
    GPU_CHECK_KERNEL

    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      coefficient.data(),
      type.data(),
      hnemd_fe_[0],
      hnemd_fe_[1],
      hnemd_fe_[2],
      hnemdec_tensor_per_atom_.data() + 0 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 3 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 4 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 6 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 1 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 5 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 7 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 8 * number_of_atoms,
      hnemdec_tensor_per_atom_.data() + 2 * number_of_atoms,
      hnemdec_tensor_sum_.data(),
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);
    GPU_CHECK_KERNEL
  } else {
    gpu_add_driving_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      coefficient.data(),
      type.data(),
      hnemd_fe_[0],
      hnemd_fe_[1],
      hnemd_fe_[2],
      force_per_atom.data(),
      force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);
  }
}

void Force::compute(
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  const std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& mass_per_atom,
  int* position_image)
{
  const int number_of_atoms = type.size();
  prepare_compute(
    number_of_atoms,
    box,
    position_per_atom,
    potential_per_atom,
    force_per_atom,
    virial_per_atom,
    position_image);
  compute_potentials(
    number_of_atoms,
    box,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

  if (compute_hnemdec_ != -1) {
    apply_hnemdec(
      number_of_atoms,
      type,
      potential_per_atom,
      force_per_atom,
      virial_per_atom,
      velocity_per_atom,
      mass_per_atom);
  }

  correct_fcp_force(number_of_atoms, force_per_atom);
}
