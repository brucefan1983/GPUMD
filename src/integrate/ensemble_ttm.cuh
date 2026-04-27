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
[1] D.M. Duffy and A.M. Rutherford, J. Phys.: Condens. Matter 19, 016207 (2007).
[2] A.M. Rutherford and D.M. Duffy, J. Phys.: Condens. Matter 19, 496201 (2007).
------------------------------------------------------------------------------*/

#pragma once
#include "ensemble.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif
#include <cstdio>
#include <string>
#include <vector>

class Atom;

struct TTM_Parameters
{
  int grouping_method = 0;
  int group_id = 0;
  double Ce = 0.0;
  double rho_e = 0.0;
  double kappa_e = 0.0;
  double gamma_p = 0.0;
  double gamma_s = 0.0;
  double v_0 = 0.0;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int active_x_min = 1;
  int active_x_max = 0;
  int active_y_min = 1;
  int active_y_max = 0;
  int active_z_min = 1;
  int active_z_max = 0;
  double T_e_init = 0.0;
  int out_interval = 1;
  std::string infile;
  std::string properties_file;
  double source = 0.0;
};

void parse_ttm_parameters(
  const int type,
  const char** param,
  const int num_param,
  const Atom& atom,
  const Box& box,
  const std::vector<Group>& group,
  const int source,
  const int sink,
  TTM_Parameters& ttm_parameters);

void print_ttm_settings(const TTM_Parameters& ttm_parameters);

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
    int ttm_group_size,
    int ttm_group_offset,
    double T,
    double Tc,
    double dT,
    const TTM_Parameters& ttm_parameters,
    const Box& box);

  Ensemble_TTM(
    int type_input,
    int ttm_group_size,
    int ttm_group_offset,
    const TTM_Parameters& ttm_parameters,
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

  int N_source, N_sink, offset_source, offset_sink;
  double c1, c2_source, c2_sink;
  GPU_Vector<gpurandState> curand_states_source;
  GPU_Vector<gpurandState> curand_states_sink;

  int ttm_grouping_method;
  int ttm_group_id;
  int N_metal;
  int offset_metal;
  GPU_Vector<gpurandState> curand_states_metal;

  double Ce;
  double rho_e;
  double kappa_e;      // internal conductivity [eV/(fs*K*A)]
  double gamma_p;
  double gamma_s;
  double gamma_p_nat;
  double gamma_s_nat;
  double v_0_sq;
  double electron_source;
  bool use_electron_properties;

  int nx, ny, nz;
  int ngrid_total;
  int active_x_min, active_x_max;
  int active_y_min, active_y_max;
  int active_z_min, active_z_max;
  double dx, dy, dz;
  std::vector<double> T_electron;
  std::vector<double> T_electron_old;
  std::vector<int> electron_cell_active;
  std::vector<double> electron_heat_capacity;
  std::vector<double> electron_thermal_conductivity;
  std::vector<double> electron_gamma_p_nat;
  std::vector<double> electron_eta;
  FILE* electron_temperature_file = nullptr;
  int electron_temperature_output_interval;

  GPU_Vector<double> gpu_T_electron;
  GPU_Vector<int> gpu_electron_cell_active;
  GPU_Vector<double> gpu_electron_gamma_p_nat;
  GPU_Vector<double> gpu_net_energy;
  GPU_Vector<int> gpu_atom_grid_index;
  GPU_Vector<double> gpu_ttm_force;

  double box_length[3];

  void initialize_ttm_common(
    int type_input,
    int ttm_group_size,
    int ttm_group_offset,
    const TTM_Parameters& ttm_parameters,
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
