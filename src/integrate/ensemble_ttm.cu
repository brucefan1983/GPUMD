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
Two-Temperature Model (TTM) for metals, with an optional heat_lan source/sink
channel for heat transport across metal-nonmetal heterointerfaces.

Three atom categories:
  1. Source/sink atoms (grouping method 0): Langevin thermostat at T+dT / T-dT
  2. Metal atoms (ttm_grouping_method): TTM Langevin coupled to electron grid
  3. All other atoms: NVE (no thermostat)

The electron subsystem is modeled as a 3D grid with finite-difference heat
diffusion. Metal atoms exchange energy with the local electron temperature
via a Langevin-like coupling (friction + stochastic force).

References:
[1] D.M. Duffy and A.M. Rutherford, J. Phys.: Condens. Matter 19, 016207 (2007).
[2] A.M. Rutherford and D.M. Duffy, J. Phys.: Condens. Matter 19, 496201 (2007).
------------------------------------------------------------------------------*/

#include "ensemble_ttm.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#ifdef USE_HIP
  #define TTM_RAND_UNIFORM(a) hiprand_uniform(a)
#else
  #define TTM_RAND_UNIFORM(a) curand_uniform(a)
#endif

static void parse_ttm_active_range(
  const char* text,
  const char* axis_name,
  const int upper_bound,
  int& lower,
  int& upper)
{
  if (strcmp(text, "all") == 0) {
    lower = 1;
    upper = upper_bound;
    return;
  }

  int value = 0;
  if (is_valid_int(text, &value)) {
    lower = value;
    upper = value;
  } else {
    const char* separator = strchr(text, ':');
    if (separator == nullptr) {
      separator = strchr(text, '-');
    }
    if (separator == nullptr) {
      PRINT_INPUT_ERROR("TTM active range should be an integer, all, or min:max.");
    }

    const std::string lower_text(text, separator - text);
    const std::string upper_text(separator + 1);
    if (!is_valid_int(lower_text.c_str(), &lower) || !is_valid_int(upper_text.c_str(), &upper)) {
      PRINT_INPUT_ERROR("TTM active range bounds should be integers.");
    }
  }

  if (lower < 1 || upper < 1 || lower > upper_bound || upper > upper_bound || lower > upper) {
    if (strcmp(axis_name, "x") == 0) {
      PRINT_INPUT_ERROR("ttm_active_x is out of range.");
    } else if (strcmp(axis_name, "y") == 0) {
      PRINT_INPUT_ERROR("ttm_active_y is out of range.");
    } else {
      PRINT_INPUT_ERROR("ttm_active_z is out of range.");
    }
  }
}

void parse_ttm_parameters(
  const int type,
  const char** param,
  const int num_param,
  const Atom& atom,
  const Box& box,
  const std::vector<Group>& group,
  const int source,
  const int sink,
  TTM_Parameters& ttm_parameters)
{
  ttm_parameters = TTM_Parameters();

  if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
    PRINT_INPUT_ERROR("ensemble ttm/heat_ttm requires periodic boundary conditions in all directions.");
  }
  if (
    box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 || box.cpu_h[5] != 0 ||
    box.cpu_h[6] != 0 || box.cpu_h[7] != 0) {
    PRINT_INPUT_ERROR("ensemble ttm/heat_ttm only supports orthogonal boxes.");
  }
  if (group.empty()) {
    PRINT_INPUT_ERROR("ensemble ttm/heat_ttm requires at least one grouping method.");
  }

  const int ttm_offset = (type == 24) ? 7 : 2;

  if (!is_valid_int(param[ttm_offset], &ttm_parameters.grouping_method)) {
    PRINT_INPUT_ERROR("TTM grouping method should be an integer.");
  }
  if (ttm_parameters.grouping_method < 0 || ttm_parameters.grouping_method >= group.size()) {
    PRINT_INPUT_ERROR("TTM grouping method out of range.");
  }

  if (!is_valid_int(param[ttm_offset + 1], &ttm_parameters.group_id)) {
    PRINT_INPUT_ERROR("TTM group ID should be an integer.");
  }
  if (
    ttm_parameters.group_id < 0 ||
    ttm_parameters.group_id >= group[ttm_parameters.grouping_method].number) {
    PRINT_INPUT_ERROR("TTM group ID out of range.");
  }
  if (group[ttm_parameters.grouping_method].cpu_size[ttm_parameters.group_id] <= 0) {
    PRINT_INPUT_ERROR("TTM metal group cannot be empty.");
  }

  if (type == 24) {
    if (group[0].cpu_size[source] <= 0) {
      PRINT_INPUT_ERROR("Heat source group for ensemble heat_ttm cannot be empty.");
    }
    if (group[0].cpu_size[sink] <= 0) {
      PRINT_INPUT_ERROR("Heat sink group for ensemble heat_ttm cannot be empty.");
    }
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      if (
        group[ttm_parameters.grouping_method].cpu_label[n] == ttm_parameters.group_id &&
        (group[0].cpu_label[n] == source || group[0].cpu_label[n] == sink)) {
        PRINT_INPUT_ERROR("TTM metal group cannot overlap with the heat source or sink group.");
      }
    }
  }

  if (!is_valid_real(param[ttm_offset + 2], &ttm_parameters.Ce)) {
    PRINT_INPUT_ERROR("Ce (electronic specific heat) should be a number.");
  }
  if (ttm_parameters.Ce <= 0.0) {
    PRINT_INPUT_ERROR("Ce should > 0.");
  }

  if (!is_valid_real(param[ttm_offset + 3], &ttm_parameters.rho_e)) {
    PRINT_INPUT_ERROR("rho_e (electronic density) should be a number.");
  }
  if (ttm_parameters.rho_e <= 0.0) {
    PRINT_INPUT_ERROR("rho_e should > 0.");
  }

  if (!is_valid_real(param[ttm_offset + 4], &ttm_parameters.kappa_e)) {
    PRINT_INPUT_ERROR("kappa_e (electronic thermal conductivity) should be a number.");
  }
  if (ttm_parameters.kappa_e < 0.0) {
    PRINT_INPUT_ERROR("kappa_e should >= 0.");
  }

  if (!is_valid_real(param[ttm_offset + 5], &ttm_parameters.gamma_p)) {
    PRINT_INPUT_ERROR("gamma_p (e-ph coupling friction) should be a number.");
  }
  if (ttm_parameters.gamma_p <= 0.0) {
    PRINT_INPUT_ERROR("gamma_p should > 0.");
  }

  if (!is_valid_real(param[ttm_offset + 6], &ttm_parameters.gamma_s)) {
    PRINT_INPUT_ERROR("gamma_s (stopping power friction) should be a number.");
  }
  if (ttm_parameters.gamma_s < 0.0) {
    PRINT_INPUT_ERROR("gamma_s should >= 0.");
  }

  if (!is_valid_real(param[ttm_offset + 7], &ttm_parameters.v_0)) {
    PRINT_INPUT_ERROR("v_0 (velocity threshold) should be a number.");
  }
  if (ttm_parameters.v_0 < 0.0) {
    PRINT_INPUT_ERROR("v_0 should >= 0.");
  }

  if (!is_valid_int(param[ttm_offset + 8], &ttm_parameters.nx)) {
    PRINT_INPUT_ERROR("nx (electron grid x) should be an integer.");
  }
  if (!is_valid_int(param[ttm_offset + 9], &ttm_parameters.ny)) {
    PRINT_INPUT_ERROR("ny (electron grid y) should be an integer.");
  }
  if (!is_valid_int(param[ttm_offset + 10], &ttm_parameters.nz)) {
    PRINT_INPUT_ERROR("nz (electron grid z) should be an integer.");
  }
  if (ttm_parameters.nx <= 0 || ttm_parameters.ny <= 0 || ttm_parameters.nz <= 0) {
    PRINT_INPUT_ERROR("Electron grid sizes must all be > 0.");
  }

  ttm_parameters.active_x_max = ttm_parameters.nx;
  ttm_parameters.active_y_max = ttm_parameters.ny;
  ttm_parameters.active_z_max = ttm_parameters.nz;

  const long long ttm_ngrid_total =
    1LL * ttm_parameters.nx * ttm_parameters.ny * ttm_parameters.nz;
  if (ttm_ngrid_total > 2147483647LL) {
    PRINT_INPUT_ERROR("Too many electron grid points for ensemble ttm/heat_ttm.");
  }

  if (!is_valid_real(param[ttm_offset + 11], &ttm_parameters.T_e_init)) {
    PRINT_INPUT_ERROR("T_e_init (initial electron temperature) should be a number.");
  }
  if (ttm_parameters.T_e_init <= 0.0) {
    PRINT_INPUT_ERROR("T_e_init should > 0.");
  }

  int i = ttm_offset + 12;
  while (i < num_param) {
    if (strcmp(param[i], "ttm_out_interval") == 0) {
      if (!is_valid_int(param[i + 1], &ttm_parameters.out_interval)) {
        PRINT_INPUT_ERROR("ttm_out_interval should be an integer.");
      }
      if (ttm_parameters.out_interval <= 0) {
        PRINT_INPUT_ERROR("ttm_out_interval should > 0.");
      }
    } else if (strcmp(param[i], "ttm_infile") == 0) {
      ttm_parameters.infile = param[i + 1];
      if (ttm_parameters.infile.empty()) {
        PRINT_INPUT_ERROR("ttm_infile should be a valid file path.");
      }
    } else if (strcmp(param[i], "ttm_properties_file") == 0) {
      ttm_parameters.properties_file = param[i + 1];
      if (ttm_parameters.properties_file.empty()) {
        PRINT_INPUT_ERROR("ttm_properties_file should be a valid file path.");
      }
    } else if (strcmp(param[i], "ttm_source") == 0) {
      if (!is_valid_real(param[i + 1], &ttm_parameters.source)) {
        PRINT_INPUT_ERROR("ttm_source should be a number.");
      }
    } else if (strcmp(param[i], "ttm_active_x") == 0) {
      parse_ttm_active_range(
        param[i + 1], "x", ttm_parameters.nx, ttm_parameters.active_x_min, ttm_parameters.active_x_max);
    } else if (strcmp(param[i], "ttm_active_y") == 0) {
      parse_ttm_active_range(
        param[i + 1], "y", ttm_parameters.ny, ttm_parameters.active_y_min, ttm_parameters.active_y_max);
    } else if (strcmp(param[i], "ttm_active_z") == 0) {
      parse_ttm_active_range(
        param[i + 1], "z", ttm_parameters.nz, ttm_parameters.active_z_min, ttm_parameters.active_z_max);
    } else {
      PRINT_INPUT_ERROR("Unknown ensemble ttm/heat_ttm optional keyword.");
    }
    i += 2;
  }
}

void print_ttm_settings(const TTM_Parameters& ttm_parameters)
{
  printf(
    "    TTM metal group is group %d in grouping method %d.\n",
    ttm_parameters.group_id,
    ttm_parameters.grouping_method);
  printf(
    "    Ce = %g, rho_e = %g, kappa_e = %g.\n",
    ttm_parameters.Ce,
    ttm_parameters.rho_e,
    ttm_parameters.kappa_e);
  printf(
    "    gamma_p = %g, gamma_s = %g, v_0 = %g.\n",
    ttm_parameters.gamma_p,
    ttm_parameters.gamma_s,
    ttm_parameters.v_0);
  printf(
    "    electron grid: %d x %d x %d.\n",
    ttm_parameters.nx,
    ttm_parameters.ny,
    ttm_parameters.nz);
  printf(
    "    active electron cells: x %d:%d, y %d:%d, z %d:%d.\n",
    ttm_parameters.active_x_min,
    ttm_parameters.active_x_max,
    ttm_parameters.active_y_min,
    ttm_parameters.active_y_max,
    ttm_parameters.active_z_min,
    ttm_parameters.active_z_max);
  if (ttm_parameters.infile.empty()) {
    printf("    uniform initial electron temperature is %g K.\n", ttm_parameters.T_e_init);
  } else {
    printf("    initial electron temperature is read from %s.\n", ttm_parameters.infile.c_str());
  }
  if (ttm_parameters.properties_file.empty()) {
    printf("    electron properties are spatially uniform.\n");
  } else {
    printf(
      "    electron cell properties are read from %s.\n", ttm_parameters.properties_file.c_str());
  }
  if (ttm_parameters.source != 0.0) {
    printf("    electron volumetric source is %g.\n", ttm_parameters.source);
  }
  printf(
    "    electron temperature snapshots are written every %d step(s) to "
    "ttm_electron_temperature.out.\n",
    ttm_parameters.out_interval);
}

// Map a metal atom to its electron grid cell index.
// For orthogonal GPUMD boxes, positions are wrapped into [0, L).
static __global__ void gpu_map_atoms_to_grid(
  const int N_metal,
  const int offset,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const double dxinv,
  const double dyinv,
  const double dzinv,
  const int nx,
  const int ny,
  const int nz,
  int* __restrict__ g_atom_grid)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < N_metal) {
    int n = g_group_contents[offset + m];
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    int ix = static_cast<int>(floor(x * dxinv));
    int iy = static_cast<int>(floor(y * dyinv));
    int iz = static_cast<int>(floor(z * dzinv));

    ix %= nx;
    iy %= ny;
    iz %= nz;
    if (ix < 0) ix += nx;
    if (iy < 0) iy += ny;
    if (iz < 0) iz += nz;

    g_atom_grid[m] = iz * ny * nx + iy * nx + ix;
  }
}

// Apply a stored TTM force for half a timestep.
static __global__ void gpu_apply_ttm_force_half(
  const int N_metal,
  const int offset,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_mass,
  const double* __restrict__ g_ttm_force,
  const double half_dt,
  double* __restrict__ g_vx,
  double* __restrict__ g_vy,
  double* __restrict__ g_vz)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < N_metal) {
    int n = g_group_contents[offset + m];
    double inv_mass = 1.0 / g_mass[n];
    g_vx[n] += g_ttm_force[m] * half_dt * inv_mass;
    g_vy[n] += g_ttm_force[m + N_metal] * half_dt * inv_mass;
    g_vz[n] += g_ttm_force[m + 2 * N_metal] * half_dt * inv_mass;
  }
}

// Compute the current TTM Langevin force from the electron grid.
static __global__ void gpu_update_ttm_force(
  gpurandState* __restrict__ g_state,
  const int N_metal,
  const int offset,
  const int* __restrict__ g_group_contents,
  const double* __restrict__ g_Te,
  const int* __restrict__ g_active,
  const double* __restrict__ g_gamma_p,
  const int* __restrict__ g_atom_grid,
  const double gamma_s,
  const double time_step,
  const double v_0_sq,
  double* __restrict__ g_ttm_force,
  double* __restrict__ g_vx,
  double* __restrict__ g_vy,
  double* __restrict__ g_vz)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < N_metal) {
    int n = g_group_contents[offset + m];
    int grid_idx = g_atom_grid[m];
    double Te = g_Te[grid_idx];
    if (g_active[grid_idx] == 0 || Te <= 0.0) {
      g_ttm_force[m] = 0.0;
      g_ttm_force[m + N_metal] = 0.0;
      g_ttm_force[m + 2 * N_metal] = 0.0;
      return;
    }
    if (Te < 0.0) Te = 0.0;
    const double gamma_p = g_gamma_p[grid_idx];
    if (gamma_p <= 0.0) {
      g_ttm_force[m] = 0.0;
      g_ttm_force[m + N_metal] = 0.0;
      g_ttm_force[m + 2 * N_metal] = 0.0;
      return;
    }

    double vx = g_vx[n];
    double vy = g_vy[n];
    double vz = g_vz[n];

    double vsq = vx * vx + vy * vy + vz * vz;
    double gamma = gamma_p;
    if (vsq > v_0_sq) gamma = gamma_p + gamma_s;

    // Use zero-mean uniform random numbers for the stochastic force.
    double gfactor = sqrt(Te * 24.0 * K_B * gamma_p / time_step);

    gpurandState state = g_state[m];
    double fx = -gamma * vx + gfactor * (TTM_RAND_UNIFORM(&state) - 0.5);
    double fy = -gamma * vy + gfactor * (TTM_RAND_UNIFORM(&state) - 0.5);
    double fz = -gamma * vz + gfactor * (TTM_RAND_UNIFORM(&state) - 0.5);
    g_state[m] = state;

    g_ttm_force[m] = fx;
    g_ttm_force[m + N_metal] = fy;
    g_ttm_force[m + 2 * N_metal] = fz;
  }
}

// Accumulate the instantaneous electron-to-atom power using the stored
// Langevin force and the final velocity of the MD step.
static __global__ void gpu_accumulate_ttm_power(
  const int N_metal,
  const int offset,
  const int* __restrict__ g_group_contents,
  const int* __restrict__ g_active,
  const int* __restrict__ g_atom_grid,
  const double* __restrict__ g_ttm_force,
  const double* __restrict__ g_vx,
  const double* __restrict__ g_vy,
  const double* __restrict__ g_vz,
  const double time_unit_conversion,
  double* __restrict__ g_net_energy)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < N_metal) {
    int n = g_group_contents[offset + m];
    int grid_idx = g_atom_grid[m];
    if (g_active[grid_idx] == 0) {
      return;
    }
    double vx = g_vx[n];
    double vy = g_vy[n];
    double vz = g_vz[n];
    double fx = g_ttm_force[m];
    double fy = g_ttm_force[m + N_metal];
    double fz = g_ttm_force[m + 2 * N_metal];
    const double power_fs = (fx * vx + fy * vy + fz * vz) / time_unit_conversion;
    atomicAdd(&g_net_energy[grid_idx], power_fs);
  }
}

void Ensemble_TTM::update_box_geometry(const Box& box)
{
  box_length[0] = box.cpu_h[0];
  box_length[1] = box.cpu_h[4];
  box_length[2] = box.cpu_h[8];
  dx = box_length[0] / nx;
  dy = box_length[1] / ny;
  dz = box_length[2] / nz;
}

void Ensemble_TTM::open_electron_temperature_file()
{
  electron_temperature_file = my_fopen("ttm_electron_temperature.out", "w");
  fprintf(
    electron_temperature_file,
    "# electron temperature snapshots for TTM\n"
    "# nx %d ny %d nz %d\n"
    "# active_x %d %d active_y %d %d active_z %d %d\n"
    "# properties_file %s\n"
    "# electron_source %.10e\n"
    "# output_interval %d step(s)\n"
    "# columns: ix iy iz T_e[K]\n",
    nx,
    ny,
    nz,
    active_x_min,
    active_x_max,
    active_y_min,
    active_y_max,
    active_z_min,
    active_z_max,
    use_electron_properties ? "yes" : "no",
    electron_source,
    electron_temperature_output_interval);
}

void Ensemble_TTM::initialize_active_cells()
{
  electron_cell_active.resize(ngrid_total, 0);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const bool active_x = ix + 1 >= active_x_min && ix + 1 <= active_x_max;
        const bool active_y = iy + 1 >= active_y_min && iy + 1 <= active_y_max;
        const bool active_z = iz + 1 >= active_z_min && iz + 1 <= active_z_max;
        if (active_x && active_y && active_z) {
          electron_cell_active[iz * ny * nx + iy * nx + ix] = 1;
        }
      }
    }
  }
}

void Ensemble_TTM::apply_active_cell_mask()
{
  for (int i = 0; i < ngrid_total; ++i) {
    if (electron_cell_active[i] == 0) {
      T_electron[i] = 0.0;
      T_electron_old[i] = 0.0;
      if (i < static_cast<int>(electron_heat_capacity.size())) {
        electron_heat_capacity[i] = 0.0;
        electron_thermal_conductivity[i] = 0.0;
        electron_gamma_p_nat[i] = 0.0;
        electron_eta[i] = 0.0;
      }
    }
  }
}

void Ensemble_TTM::load_initial_electron_temperatures(const std::string& filename)
{
  if (filename.empty()) {
    apply_active_cell_mask();
    return;
  }

  std::ifstream input(filename);
  if (!input) {
    std::string error = "Could not open ttm_infile.";
    PRINT_INPUT_ERROR(error.c_str());
  }

  std::vector<int> is_set(ngrid_total, 0);
  int num_set = 0;
  std::string line;
  while (std::getline(input, line)) {
    const size_t first_nonblank = line.find_first_not_of(" \t\r\n");
    if (first_nonblank == std::string::npos) {
      continue;
    }
    if (line[first_nonblank] == '#') {
      continue;
    }

    std::istringstream iss(line.substr(first_nonblank));
    int ix = 0;
    int iy = 0;
    int iz = 0;
    double Te = 0.0;
    if (!(iss >> ix >> iy >> iz >> Te)) {
      std::string error = "Invalid line in ttm_infile.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    if (ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz) {
      std::string error = "Grid index in ttm_infile is out of range.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    const int idx = (iz - 1) * ny * nx + (iy - 1) * nx + (ix - 1);
    if (is_set[idx]) {
      std::string error = "Duplicate grid point found in ttm_infile.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    if (Te < 0.0) {
      std::string error = "Electron temperatures in ttm_infile should >= 0.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    if (electron_cell_active[idx] == 0 && Te != 0.0) {
      std::string error = "Electron temperatures in inactive ttm cells should be 0.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    T_electron[idx] = Te;
    T_electron_old[idx] = Te;
    is_set[idx] = 1;
    ++num_set;
  }

  if (num_set != ngrid_total) {
    std::string error = "ttm_infile did not set all electron grid temperatures.";
    PRINT_INPUT_ERROR(error.c_str());
  }

  apply_active_cell_mask();
}

void Ensemble_TTM::load_electron_properties(const std::string& filename)
{
  if (filename.empty()) {
    return;
  }

  std::ifstream input(filename);
  if (!input) {
    std::string error = "Could not open ttm_properties_file.";
    PRINT_INPUT_ERROR(error.c_str());
  }

  std::vector<int> is_set(ngrid_total, 0);
  int num_set = 0;
  std::string line;
  while (std::getline(input, line)) {
    const size_t first_nonblank = line.find_first_not_of(" \t\r\n");
    if (first_nonblank == std::string::npos) {
      continue;
    }
    if (line[first_nonblank] == '#') {
      continue;
    }

    std::istringstream iss(line.substr(first_nonblank));
    int ix = 0;
    int iy = 0;
    int iz = 0;
    double c_vol = 0.0;
    double kappa = 0.0;
    double gamma_p_local = 0.0;
    double eta = 0.0;
    if (!(iss >> ix >> iy >> iz >> c_vol >> kappa >> gamma_p_local >> eta)) {
      std::string error = "Invalid line in ttm_properties_file.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    if (ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz) {
      std::string error = "Grid index in ttm_properties_file is out of range.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    if (c_vol < 0.0 || kappa < 0.0 || gamma_p_local < 0.0) {
      std::string error = "Cell properties in ttm_properties_file should be >= 0.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    if (eta < 0.0 || eta > 1.0) {
      std::string error = "eta in ttm_properties_file should be between 0 and 1.";
      PRINT_INPUT_ERROR(error.c_str());
    }

    const int idx = (iz - 1) * ny * nx + (iy - 1) * nx + (ix - 1);
    if (is_set[idx]) {
      std::string error = "Duplicate grid point found in ttm_properties_file.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    electron_heat_capacity[idx] = c_vol;
    electron_thermal_conductivity[idx] = kappa / 1000.0;
    electron_gamma_p_nat[idx] = gamma_p_local * TIME_UNIT_CONVERSION / 1000.0;
    electron_eta[idx] = eta;
    is_set[idx] = 1;
    ++num_set;
  }

  if (num_set != ngrid_total) {
    std::string error = "ttm_properties_file did not set all electron grid properties.";
    PRINT_INPUT_ERROR(error.c_str());
  }
}

static inline double safe_effective_kappa(const double a, const double b)
{
  if (a <= 0.0 || b <= 0.0) {
    return 0.0;
  }
  return 2.0 * a * b / (a + b);
}

void Ensemble_TTM::write_electron_temperature_snapshot(const int step)
{
  fprintf(electron_temperature_file, "# step %d\n", step);
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int idx = iz * ny * nx + iy * nx + ix;
        fprintf(
          electron_temperature_file, "%d %d %d %.10e\n", ix + 1, iy + 1, iz + 1, T_electron[idx]);
      }
    }
  }
  fflush(electron_temperature_file);
}

void Ensemble_TTM::close_electron_temperature_file()
{
  if (electron_temperature_file != nullptr) {
    fclose(electron_temperature_file);
    electron_temperature_file = nullptr;
  }
}

void Ensemble_TTM::initialize_ttm_random_states()
{
  curand_states_metal.resize(N_metal);
  int grid_size_metal = (N_metal - 1) / 128 + 1;
  initialize_curand_states<<<grid_size_metal, 128>>>(curand_states_metal.data(), N_metal, rand());
  GPU_CHECK_KERNEL
}

void Ensemble_TTM::initialize_electron_grid(
  double T_e_init,
  const std::string& electron_temperature_init_file_input,
  const std::string& electron_property_file_input,
  const Box& box)
{
  update_box_geometry(box);
  initialize_active_cells();

  T_electron.resize(ngrid_total, T_e_init);
  T_electron_old.resize(ngrid_total, T_e_init);
  electron_heat_capacity.resize(ngrid_total, Ce * rho_e);
  electron_thermal_conductivity.resize(ngrid_total, kappa_e);
  electron_gamma_p_nat.resize(ngrid_total, gamma_p_nat);
  electron_eta.resize(ngrid_total, 0.0);
  if (!use_electron_properties) {
    for (int i = 0; i < ngrid_total; ++i) {
      if (electron_cell_active[i]) {
        electron_eta[i] = 1.0;
      }
    }
  }
  load_electron_properties(electron_property_file_input);
  load_initial_electron_temperatures(electron_temperature_init_file_input);
}

void Ensemble_TTM::initialize_ttm_gpu_data()
{
  gpu_T_electron.resize(ngrid_total);
  gpu_T_electron.fill(0);
  gpu_electron_cell_active.resize(ngrid_total);
  gpu_electron_cell_active.fill(0);
  gpu_electron_gamma_p_nat.resize(ngrid_total);
  gpu_electron_gamma_p_nat.fill(0);
  gpu_net_energy.resize(ngrid_total);
  gpu_net_energy.fill(0);
  gpu_atom_grid_index.resize(N_metal);
  gpu_atom_grid_index.fill(0);
  gpu_ttm_force.resize(N_metal * 3);
  gpu_ttm_force.fill(0);

  gpu_T_electron.copy_from_host(T_electron.data());
  gpu_electron_cell_active.copy_from_host(electron_cell_active.data());
  gpu_electron_gamma_p_nat.copy_from_host(electron_gamma_p_nat.data());
  open_electron_temperature_file();
  write_electron_temperature_snapshot(0);
}

void Ensemble_TTM::initialize_ttm_common(
  int type_input,
  int ttm_group_size,
  int ttm_group_offset,
  const TTM_Parameters& ttm_parameters,
  const Box& box)
{
  type = type_input;

  ttm_grouping_method = ttm_parameters.grouping_method;
  ttm_group_id = ttm_parameters.group_id;
  N_metal = ttm_group_size;
  offset_metal = ttm_group_offset;
  initialize_ttm_random_states();

  energy_transferred[0] = 0.0;
  energy_transferred[1] = 0.0;

  // TTM physics parameters.
  // Input units are converted to internal GPUMD units:
  //   kappa_e : eV / (ps * K * A)
  //   gamma_* : mass / ps
  //   v_0     : A / ps
  Ce = ttm_parameters.Ce;
  rho_e = ttm_parameters.rho_e;
  kappa_e = ttm_parameters.kappa_e / 1000.0;
  gamma_p = ttm_parameters.gamma_p;
  gamma_s = ttm_parameters.gamma_s;
  gamma_p_nat = gamma_p * TIME_UNIT_CONVERSION / 1000.0;
  gamma_s_nat = gamma_s * TIME_UNIT_CONVERSION / 1000.0;
  double v_0_nat = ttm_parameters.v_0 * TIME_UNIT_CONVERSION / 1000.0;
  v_0_sq = v_0_nat * v_0_nat;
  electron_source = ttm_parameters.source / 1000.0;
  use_electron_properties = !ttm_parameters.properties_file.empty();

  nx = ttm_parameters.nx;
  ny = ttm_parameters.ny;
  nz = ttm_parameters.nz;
  ngrid_total = nx * ny * nz;
  active_x_min = ttm_parameters.active_x_min;
  active_x_max = ttm_parameters.active_x_max;
  active_y_min = ttm_parameters.active_y_min;
  active_y_max = ttm_parameters.active_y_max;
  active_z_min = ttm_parameters.active_z_min;
  active_z_max = ttm_parameters.active_z_max;
  electron_temperature_output_interval = ttm_parameters.out_interval;

  initialize_electron_grid(
    ttm_parameters.T_e_init,
    ttm_parameters.infile,
    ttm_parameters.properties_file,
    box);
  initialize_ttm_gpu_data();
}

Ensemble_TTM::Ensemble_TTM(
  int type_input,
  int source_input,
  int sink_input,
  int source_size,
  int sink_size,
  int source_offset,
  int sink_offset,
  int ttm_group_size,
  int ttm_group_offset,
  double T,
  double Tc,
  double dT,
  const TTM_Parameters& ttm_parameters,
  const Box& box)
{
  use_heat_lan = true;
  temperature = T;
  temperature_coupling = Tc;
  delta_temperature = dT;
  source = source_input;
  sink = sink_input;
  N_source = source_size;
  N_sink = sink_size;
  offset_source = source_offset;
  offset_sink = sink_offset;

  // source/sink Langevin coefficients (same as heat_lan)
  c1 = exp(-0.5 / temperature_coupling);
  c2_source = sqrt((1 - c1 * c1) * K_B * (T + dT));
  c2_sink = sqrt((1 - c1 * c1) * K_B * (T - dT));

  // initialize curand states for source, sink, and metal
  curand_states_source.resize(N_source);
  curand_states_sink.resize(N_sink);
  int grid_size_source = (N_source - 1) / 128 + 1;
  int grid_size_sink = (N_sink - 1) / 128 + 1;
  initialize_curand_states<<<grid_size_source, 128>>>(
    curand_states_source.data(), N_source, rand());
  GPU_CHECK_KERNEL
  initialize_curand_states<<<grid_size_sink, 128>>>(
    curand_states_sink.data(), N_sink, rand());
  GPU_CHECK_KERNEL
  initialize_ttm_common(
    type_input,
    ttm_group_size,
    ttm_group_offset,
    ttm_parameters,
    box);
}

Ensemble_TTM::Ensemble_TTM(
  int type_input,
  int ttm_group_size,
  int ttm_group_offset,
  const TTM_Parameters& ttm_parameters,
  const Box& box)
{
  use_heat_lan = false;
  temperature = 0.0;
  temperature_coupling = 0.0;
  delta_temperature = 0.0;
  source = -1;
  sink = -1;
  N_source = 0;
  N_sink = 0;
  offset_source = 0;
  offset_sink = 0;

  curand_states_source.resize(0);
  curand_states_sink.resize(0);
  initialize_ttm_common(
    type_input,
    ttm_group_size,
    ttm_group_offset,
    ttm_parameters,
    box);
}

Ensemble_TTM::~Ensemble_TTM(void)
{
  close_electron_temperature_file();
}

// Source/sink Langevin thermostat (identical to heat_lan)
void Ensemble_TTM::integrate_heat_lan_half(
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();
  int Ng = group[0].number;

  std::vector<double> ek2(Ng);
  GPU_Vector<double> ke(Ng);

  find_ke<<<Ng, 512>>>(
    group[0].size.data(),
    group[0].size_sum.data(),
    group[0].contents.data(),
    mass.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms,
    ke.data());
  GPU_CHECK_KERNEL

  ke.copy_to_host(ek2.data());
  energy_transferred[0] += ek2[source] * 0.5;
  energy_transferred[1] += ek2[sink] * 0.5;

  gpu_langevin<<<(N_source - 1) / 128 + 1, 128>>>(
    curand_states_source.data(),
    N_source,
    offset_source,
    group[0].contents.data(),
    c1,
    c2_source,
    mass.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL

  gpu_langevin<<<(N_sink - 1) / 128 + 1, 128>>>(
    curand_states_sink.data(),
    N_sink,
    offset_sink,
    group[0].contents.data(),
    c1,
    c2_sink,
    mass.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL

  find_ke<<<Ng, 512>>>(
    group[0].size.data(),
    group[0].size_sum.data(),
    group[0].contents.data(),
    mass.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms,
    ke.data());
  GPU_CHECK_KERNEL

  ke.copy_to_host(ek2.data());
  energy_transferred[0] -= ek2[source] * 0.5;
  energy_transferred[1] -= ek2[sink] * 0.5;
}

// Apply the stored TTM force to metal atoms for half a timestep.
void Ensemble_TTM::apply_ttm_force_half(
  const double half_time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();
  const int gm = ttm_grouping_method;
  gpu_apply_ttm_force_half<<<(N_metal - 1) / 128 + 1, 128>>>(
    N_metal,
    offset_metal,
    group[gm].contents.data(),
    mass.data(),
    gpu_ttm_force.data(),
    half_time_step,
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

// Compute the current TTM Langevin force from the electron grid.
void Ensemble_TTM::update_ttm_force(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = position_per_atom.size() / 3;
  const int gm = ttm_grouping_method;

  gpu_map_atoms_to_grid<<<(N_metal - 1) / 128 + 1, 128>>>(
    N_metal,
    offset_metal,
    group[gm].contents.data(),
    position_per_atom.data(),
    position_per_atom.data() + number_of_atoms,
    position_per_atom.data() + 2 * number_of_atoms,
    nx / box_length[0],
    ny / box_length[1],
    nz / box_length[2],
    nx,
    ny,
    nz,
    gpu_atom_grid_index.data());
  GPU_CHECK_KERNEL

  gpu_net_energy.fill(0);

  gpu_update_ttm_force<<<(N_metal - 1) / 128 + 1, 128>>>(
    curand_states_metal.data(),
    N_metal,
    offset_metal,
    group[gm].contents.data(),
    gpu_T_electron.data(),
    gpu_electron_cell_active.data(),
    gpu_electron_gamma_p_nat.data(),
    gpu_atom_grid_index.data(),
    gamma_s_nat,
    time_step,
    v_0_sq,
    gpu_ttm_force.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

void Ensemble_TTM::accumulate_ttm_power(
  const std::vector<Group>& group,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = velocity_per_atom.size() / 3;
  const int gm = ttm_grouping_method;
  gpu_accumulate_ttm_power<<<(N_metal - 1) / 128 + 1, 128>>>(
    N_metal,
    offset_metal,
    group[gm].contents.data(),
    gpu_electron_cell_active.data(),
    gpu_atom_grid_index.data(),
    gpu_ttm_force.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms,
    TIME_UNIT_CONVERSION,
    gpu_net_energy.data());
  GPU_CHECK_KERNEL
}

// Solve the electron heat diffusion equation with explicit finite differences
void Ensemble_TTM::update_electron_temperature(const double time_step)
{
  double dt_fs = time_step * TIME_UNIT_CONVERSION;
  double del_vol = dx * dy * dz;

  // copy net electron-to-atom power from GPU to CPU
  std::vector<double> net_energy_cpu(ngrid_total);
  gpu_net_energy.copy_to_host(net_energy_cpu.data());

  // stability criterion for explicit FD
  int num_inner_steps = 1;
  double inner_dt = dt_fs;

  const double voxel_coeff = 1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz);
  double fourier_max = 0.0;
  for (int i = 0; i < ngrid_total; ++i) {
    if (electron_heat_capacity[i] > 0.0 && electron_thermal_conductivity[i] > 0.0) {
      const double fourier =
        2.0 * electron_thermal_conductivity[i] * voxel_coeff / electron_heat_capacity[i];
      if (fourier > fourier_max) {
        fourier_max = fourier;
      }
    }
  }

  if (fourier_max > 0.0) {
    double stability = 1.0 - fourier_max * inner_dt;
    if (stability < 0.0) {
      inner_dt = 1.0 / fourier_max;
      num_inner_steps = static_cast<int>(dt_fs / inner_dt) + 1;
      inner_dt = dt_fs / static_cast<double>(num_inner_steps);
      if (num_inner_steps > 1000000) {
        printf("Warning: too many inner timesteps (%d) in TTM electron diffusion.\n",
               num_inner_steps);
      }
    }
  }

  // finite difference iterations
  for (int istep = 0; istep < num_inner_steps; istep++) {
    // save old temperatures
    for (int i = 0; i < ngrid_total; i++) {
      T_electron_old[i] = T_electron[i];
    }

    // update electron temperature
    for (int iz = 0; iz < nz; iz++) {
      for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
          int idx = iz * ny * nx + iy * nx + ix;
          const double T_center = T_electron_old[idx];
          if (electron_cell_active[idx] == 0 || T_center <= 0.0) {
            T_electron[idx] = 0.0;
            continue;
          }

          // periodic neighbors
          int xr = (ix + 1 < nx) ? ix + 1 : 0;
          int xl = (ix - 1 >= 0) ? ix - 1 : nx - 1;
          int yr = (iy + 1 < ny) ? iy + 1 : 0;
          int yl = (iy - 1 >= 0) ? iy - 1 : ny - 1;
          int zr = (iz + 1 < nz) ? iz + 1 : 0;
          int zl = (iz - 1 >= 0) ? iz - 1 : nz - 1;

          const int idx_xr = iz * ny * nx + iy * nx + xr;
          const int idx_xl = iz * ny * nx + iy * nx + xl;
          const int idx_yr = iz * ny * nx + yr * nx + ix;
          const int idx_yl = iz * ny * nx + yl * nx + ix;
          const int idx_zr = zr * ny * nx + iy * nx + ix;
          const int idx_zl = zl * ny * nx + iy * nx + ix;
          int left = 1;
          int right = 1;
          int in = 1;
          int out = 1;
          int up = 1;
          int down = 1;
          if (!(electron_cell_active[idx_xl] && T_electron_old[idx_xl] > 0.0)) left = 0;
          if (!(electron_cell_active[idx_xr] && T_electron_old[idx_xr] > 0.0)) right = 0;
          if (!(electron_cell_active[idx_yr] && T_electron_old[idx_yr] > 0.0)) in = 0;
          if (!(electron_cell_active[idx_yl] && T_electron_old[idx_yl] > 0.0)) out = 0;
          if (!(electron_cell_active[idx_zr] && T_electron_old[idx_zr] > 0.0)) up = 0;
          if (!(electron_cell_active[idx_zl] && T_electron_old[idx_zl] > 0.0)) down = 0;

          const double c_vol = electron_heat_capacity[idx];
          if (c_vol <= 0.0) {
            T_electron[idx] = 0.0;
            continue;
          }

          const double k_center = electron_thermal_conductivity[idx];
          const double diffusion =
            safe_effective_kappa(electron_thermal_conductivity[idx_xl], k_center) *
              (T_electron_old[idx_xl] - T_center) / (dx * dx) * left +
            safe_effective_kappa(electron_thermal_conductivity[idx_xr], k_center) *
              (T_electron_old[idx_xr] - T_center) / (dx * dx) * right +
            safe_effective_kappa(electron_thermal_conductivity[idx_yl], k_center) *
              (T_electron_old[idx_yl] - T_center) / (dy * dy) * out +
            safe_effective_kappa(electron_thermal_conductivity[idx_yr], k_center) *
              (T_electron_old[idx_yr] - T_center) / (dy * dy) * in +
            safe_effective_kappa(electron_thermal_conductivity[idx_zl], k_center) *
              (T_electron_old[idx_zl] - T_center) / (dz * dz) * down +
            safe_effective_kappa(electron_thermal_conductivity[idx_zr], k_center) *
              (T_electron_old[idx_zr] - T_center) / (dz * dz) * up;

          T_electron[idx] = T_center +
            inner_dt / c_vol *
              (diffusion - net_energy_cpu[idx] / del_vol + electron_source * electron_eta[idx]);
        }
      }
    }
  }

  for (int i = 0; i < ngrid_total; ++i) {
    if (electron_cell_active[i] == 0) {
      T_electron[i] = 0.0;
    } else if (T_electron[i] < 0.0) {
      PRINT_INPUT_ERROR("Electronic temperature dropped below zero.");
    }
  }

  // copy updated electron temperatures back to GPU
  gpu_T_electron.copy_from_host(T_electron.data());
  const int step = *current_step + 1;
  if (step % electron_temperature_output_interval == 0) {
    write_electron_temperature_snapshot(step);
  }
}

void Ensemble_TTM::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  update_box_geometry(box);

  if (use_heat_lan) {
    // 1. Apply source/sink Langevin thermostat (NEMD part)
    integrate_heat_lan_half(group, atom.mass, atom.velocity_per_atom);
  }

  // 2. Apply the stored TTM force for the first half-step.
  apply_ttm_force_half(0.5 * time_step, group, atom.mass, atom.velocity_per_atom);

  // 3. Velocity Verlet first half-step (all atoms)
  velocity_verlet(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
}

void Ensemble_TTM::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  update_box_geometry(box);

  // 1. Generate the current TTM force using the pre-final-integrate velocity.
  update_ttm_force(time_step, group, atom.position_per_atom, atom.velocity_per_atom);

  // 2. Velocity Verlet second half-step (all atoms)
  velocity_verlet(
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);

  if (use_heat_lan) {
    // 3. Apply source/sink Langevin thermostat (NEMD part)
    integrate_heat_lan_half(group, atom.mass, atom.velocity_per_atom);
  }

  // 4. Apply the current TTM force for the second half-step.
  apply_ttm_force_half(0.5 * time_step, group, atom.mass, atom.velocity_per_atom);

  // 5. Accumulate the electron-to-atom power with the final velocity.
  accumulate_ttm_power(group, atom.velocity_per_atom);

  // 6. Update electron temperature grid (FD diffusion)
  update_electron_temperature(time_step);
}
