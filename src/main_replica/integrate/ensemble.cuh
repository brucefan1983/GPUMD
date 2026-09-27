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

#pragma once
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Ensemble
{
public:
  Ensemble(void);
  virtual ~Ensemble(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) = 0;

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) = 0;

  void find_thermo(
    const bool use_target_temperature,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& thermo);

  void find_thermo(
    const bool use_target_temperature,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& thermo,
    const gpuStream_t stream);

  int type; // ensemble type in a specific run
  int source;
  int sink;
  int fixed_group = -1; // ID of the group in which the atoms will be fixed
  int move_group = -1;  // ID of the group in which the atoms will move with a constant velocity
  int fixed_grouping_method = 0;
  int move_grouping_method = 0;
  double move_velocity[3];
  int removed_degrees_of_freedom = 0; // e.g. three for PRD's zero COM momentum
  double temperature; // target temperature at a specific time
  double delta_temperature;
  double temperature_coupling;
  double energy_transferred[2]; // energy transferred from system to heat baths

protected:
  void velocity_verlet(
    const bool is_step1,
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void velocity_verlet(
    const bool is_step1,
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    const gpuStream_t stream);

#ifdef USE_NEPCG
  void velocity_verlet_cg(
    const bool is_step1,
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);
#endif

  void scale_velocity_global(const double factor, GPU_Vector<double>& velocity_per_atom);

  void scale_velocity_global(
    const double factor,
    GPU_Vector<double>& velocity_per_atom,
    const gpuStream_t stream);

  void find_vc_and_ke(
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& velocity_per_atom,
    double* vcx,
    double* vcy,
    double* vcz,
    double* ke);

  void scale_velocity_local(
    const double factor_1,
    const double factor_2,
    const double* vcx,
    const double* vcy,
    const double* vcz,
    const double* ke,
    const std::vector<Group>& group,
    GPU_Vector<double>& velocity_per_atom);
};
