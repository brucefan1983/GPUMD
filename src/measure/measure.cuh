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
#include "active.cuh"
#include "compute.cuh"
#include "dos.cuh"
#include "dump_beads.cuh"
#include "dump_exyz.cuh"
#include "dump_force.cuh"
#include "dump_observer.cuh"
#include "dump_piston.cuh"
#include "dump_position.cuh"
#include "dump_restart.cuh"
#include "dump_thermo.cuh"
#include "dump_velocity.cuh"
#include "force/force.cuh"
#include "hac.cuh"
#include "hnemd_kappa.cuh"
#include "hnemdec_kappa.cuh"
#include "integrate/integrate.cuh"
#include "lsqt.cuh"
#include "modal_analysis.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "msd.cuh"
#include "rdf.cuh"
#include "sdc.cuh"
#include "shc.cuh"
#include "utilities/gpu_vector.cuh"
#include "viscosity.cuh"
#ifdef USE_NETCDF
#include "dump_netcdf.cuh"
#endif
#ifdef USE_PLUMED
#include "plumed.cuh"
#endif

class Atom;
class Force;
class Ensemble;
class Measure
{
public:
  void initialize(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  void finalize(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature,
    const double volume,
    const double number_of_beads);

  void process(
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
    Force& force);

  LSQT lsqt;
  DOS dos;
  SDC sdc;
  MSD msd;
  HAC hac;
  RDF rdf;
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
  Dump_Beads dump_beads;
  Dump_Observer dump_observer;
  Dump_Piston dump_piston;
  Active active;
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
