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
#include "utilities/gpu_vector.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include <vector>

class Integrate;
class Atom;
class Force;


enum class Property_Name {
  none = 0,
  compute,
  dos,
  hac,
  shc,
  msd,
  sdc,
  rdf,
  adf,
  angular_rdf,
  viscosity,
  lsqt,
  hnemd_kappa,
  hnemdec_kappa,
  modal_analysis,
  plumed,
  dump_netcdf,
  dump_exyz,
  dump_force,
  dump_position,
  dump_restart,
  dump_thermo,
  dump_velocity,
  dump_shock_nemd,
  dump_dipole,
  dump_polarizability,
  dump_beads,
  dump_observer,
  active
};

class Property
{
public:

  Property_Name property_name = Property_Name::none;

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) = 0;

  virtual void process(
      const int number_of_steps,
      int step,
      const int fixed_group,
      const int move_group,
      const double global_time,
      const double temperature,
      Integrate& integrate,
      Box& box,
      std::vector<Group>& group,
      GPU_Vector<double>& thermo,
      Atom& atom,
      Force& force) = 0;

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) = 0;
};