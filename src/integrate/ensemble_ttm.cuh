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

References:
[1] P.B. Crozier, R.E. Jones, et al., LAMMPS fix_ttm implementation.
[2] G. Bussi and M. Parrinello, Phys. Rev. E 75, 056707 (2007).
------------------------------------------------------------------------------*/

#pragma once
#include "ensemble.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif
#include <cstdio>
#include <string>
#include <vector>

class Ensemble_TTM : public Ensemble
{
public:
  Ensemble_TTM(
    int type_input,
    int source_input,
    int sink_input,
    int source_size,
    int sink_size,
    int source_offset,
    int sink_offset,
    int ttm_grouping_method_input,
    int ttm_group_input,
    int ttm_group_size,
    int ttm_group_offset,
    double T,
    double Tc,
    double dT,
    double Ce_input,
    double rho_e_input,
    double kappa_e_input,
    double gamma_p_input,
    double gamma_s_input,
    double v_0_input,
    int nx_input,
    int ny_input,
    int nz_input,
    int active_x_min_input,
    int active_x_max_input,
    int active_y_min_input,
    int active_y_max_input,
    int active_z_min_input,
    int active_z_max_input,
    double T_e_init,
    int electron_temperature_output_interval_input,
    const std::string& electron_temperature_init_file_input,
    const std::string& electron_property_file_input,
    double electron_source_input,
    const Box& box);

  Ensemble_TTM(
    int type_input,
    int ttm_grouping_method_input,
    int ttm_group_input,
    int ttm_group_size,
    int ttm_group_offset,
    double Ce_input,
    double rho_e_input,
    double kappa_e_input,
    double gamma_p_input,
    double gamma_s_input,
    double v_0_input,
    int nx_input,
    int ny_input,
    int nz_input,
    int active_x_min_input,
    int active_x_max_input,
    int active_y_min_input,
    int active_y_max_input,
    int active_z_min_input,
    int active_z_max_input,
    double T_e_init,
    int electron_temperature_output_interval_input,
    const std::string& electron_temperature_init_file_input,
    const std::string& electron_property_file_input,
    double electron_source_input,
    const Box& box);

  virtual ~Ensemble_TTM(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

private:
  bool use_heat_lan;

  // source/sink Langevin parameters (same as heat_lan)
  int N_source, N_sink, offset_source, offset_sink;
  double c1, c2_source, c2_sink;
  GPU_Vector<gpurandState> curand_states_source;
  GPU_Vector<gpurandState> curand_states_sink;

  // TTM metal group
  int ttm_grouping_method;
  int ttm_group_id;
  int N_metal;
  int offset_metal;
  GPU_Vector<gpurandState> curand_states_metal;

  // electron-phonon coupling parameters
  double Ce;           // electronic specific heat, matching LAMMPS fix_ttm input
  double rho_e;        // electronic density [1/A^3] (Ce*rho_e = volumetric heat capacity)
  double kappa_e;      // internal conductivity [eV/(fs*K*A)]
  double gamma_p;      // raw LAMMPS-style input [mass/ps]
  double gamma_s;      // raw LAMMPS-style input [mass/ps]
  double gamma_p_nat;  // internal friction coefficient [mass/natural_time]
  double gamma_s_nat;  // internal stopping coefficient [mass/natural_time]
  double v_0_sq;       // velocity threshold squared in internal velocity units
  double electron_source;
  bool use_electron_properties;

  // electron temperature grid
  int nx, ny, nz;
  int ngrid_total;
  int active_x_min, active_x_max;
  int active_y_min, active_y_max;
  int active_z_min, active_z_max;
  double dx, dy, dz; // grid spacing in Angstrom
  std::vector<double> T_electron;
  std::vector<double> T_electron_old;
  std::vector<int> electron_cell_active;
  std::vector<double> electron_heat_capacity;
  std::vector<double> electron_thermal_conductivity;
  std::vector<double> electron_gamma_p_nat;
  std::vector<double> electron_eta;
  FILE* electron_temperature_file = nullptr;
  int electron_temperature_output_interval;

  // GPU arrays for TTM coupling
  GPU_Vector<double> gpu_T_electron;       // electron temperature at each grid point
  GPU_Vector<int> gpu_electron_cell_active;
  GPU_Vector<double> gpu_electron_gamma_p_nat;
  GPU_Vector<double> gpu_net_energy;       // net electron-to-atom power per grid cell
  GPU_Vector<int> gpu_atom_grid_index;     // grid cell index for each metal atom
  GPU_Vector<double> gpu_ttm_force;        // stored Langevin force for each metal atom

  // box dimensions (orthogonal periodic box)
  double box_length[3];

  // methods
  void initialize_ttm_common(
    int type_input,
    int ttm_grouping_method_input,
    int ttm_group_input,
    int ttm_group_size,
    int ttm_group_offset,
    double Ce_input,
    double rho_e_input,
    double kappa_e_input,
    double gamma_p_input,
    double gamma_s_input,
    double v_0_input,
    int nx_input,
    int ny_input,
    int nz_input,
    int active_x_min_input,
    int active_x_max_input,
    int active_y_min_input,
    int active_y_max_input,
    int active_z_min_input,
    int active_z_max_input,
    double T_e_init,
    int electron_temperature_output_interval_input,
    const std::string& electron_temperature_init_file_input,
    const std::string& electron_property_file_input,
    double electron_source_input,
    const Box& box);
  void initialize_ttm_random_states();
  void initialize_electron_grid(
    double T_e_init,
    const std::string& electron_temperature_init_file_input,
    const std::string& electron_property_file_input,
    const Box& box);
  void initialize_ttm_gpu_data();
  void update_box_geometry(const Box& box);
  void open_electron_temperature_file();
  void load_initial_electron_temperatures(const std::string& filename);
  void load_electron_properties(const std::string& filename);
  void write_electron_temperature_snapshot(const int step);
  void close_electron_temperature_file();
  void initialize_active_cells();
  void apply_active_cell_mask();

  void integrate_heat_lan_half(
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    GPU_Vector<double>& velocity_per_atom);

  void apply_ttm_force_half(
    const double half_time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    GPU_Vector<double>& velocity_per_atom);

  void update_ttm_force(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void accumulate_ttm_power(
    const std::vector<Group>& group,
    GPU_Vector<double>& velocity_per_atom);

  void update_electron_temperature(const double time_step);
};
