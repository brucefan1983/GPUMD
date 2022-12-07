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

#pragma once
#include "compute.cuh"
#include "dos.cuh"
#include "dump_exyz.cuh"
#include "dump_force.cuh"
#include "dump_position.cuh"
#include "dump_restart.cuh"
#include "dump_thermo.cuh"
#include "dump_velocity.cuh"
#include "hac.cuh"
#include "hnemd_kappa.cuh"
#include "hnemdec_kappa.cuh"
#include "modal_analysis.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "msd.cuh"
#include "sdc.cuh"
#include "shc.cuh"
#include "utilities/gpu_vector.cuh"
#include "viscosity.cuh"
#include "force/force.cuh"
#ifdef USE_NETCDF
#include "dump_netcdf.cuh"
#endif
#ifdef USE_PLUMED
#include "plumed.cuh"
#endif

class Atom;
class Force;

class Measure
{
public:
  void initialize(
    const int number_of_steps,
    const double time_step,
    Box& box,
    std::vector<Group>& group,
    Atom& atom);

  void finalize(
    const int number_of_steps,
    const double time_step,
    const double temperature,
    const double volume);

  void process(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const double global_time,
    const double temperature,
    const double energy_transferred[],
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force);
  
  void dump_properties_for_all_potentials(
    int step,
    std::vector<Group>& group,
    Atom& atom,
    Force& force);
     
  DOS dos;
  SDC sdc;
  MSD msd;
  HAC hac;
  Viscosity viscosity;
  SHC shc;
  HNEMD hnemd;
  HNEMDEC hnemdec;
  Compute compute;
  MODAL_ANALYSIS modal_analysis;
  Dump_Position dump_position;
  Dump_Velocity dump_velocity;
  Dump_Thermo dump_thermo;
  Dump_Restart dump_restart;
  Dump_Force dump_force;
  Dump_EXYZ dump_exyz;
#ifdef USE_NETCDF
  DUMP_NETCDF dump_netcdf;
#endif
#ifdef USE_PLUMED
  PLUMED plmd;
#endif

  // functions to get inputs from run.in
  void parse_dump_position(const char**, int);
  void parse_compute_gkma(const char**, int, const int number_of_types);
  void parse_compute_hnema(const char**, int, const int number_of_types);
};
