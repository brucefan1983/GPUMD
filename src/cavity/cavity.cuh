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

#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include "cavity/potential_cavity.cuh"
#include "cavity/atom_cavity.cuh"
#include <string>
#include <vector>
class Box;
class Atom;
class Force;
class Integrate;

class Cavity
{
public:
  Cavity(void);
  void parse(
      const char** param, 
      int num_param,
      int number_of_atoms);
  void initialize(
    Box& box,
    Atom& atom,
    Force& force);
  void compute_dipole_and_jacobian(
    int step,
    Box& box,
    Atom& atom,
    Force& force);
  void compute_and_apply_cavity_force(Atom& atom);
  void update_cavity(const int step, const double global_time);
  void write(const int step, const double global_time);
  void finalize();

private:
  bool enabled_ = false;
  int dump_interval_ = 1;
  FILE* jacfile_;
  FILE* cavfile_;
  Atom atom_copy;
  AtomCavity atom_cavity;
  std::unique_ptr<PotentialCavity> potential;
  std::unique_ptr<PotentialCavity> potential_jacobian;
  int number_of_atoms_;
  int number_of_atoms_in_copied_system_;
  GPU_Vector<double> gpu_dipole_;
  std::vector<double> cpu_dipole_;
  GPU_Vector<double> gpu_dipole_jacobian_;
  std::vector<double> cpu_dipole_jacobian_;
  GPU_Vector<double> gpu_cavity_force_;
  std::vector<double> cpu_cavity_force_;
  std::vector<double> masses_;
  double mass_;
  double coupling_strength;
  double cavity_frequency; 
  int charge;
  double q0;           // Initial cavity coordinate, q_0
  double q;            // cavity canonical position coordinate, q(t)
  double p;            // cavity canonical momentum coordinate, p(t)
  double cos_integral; // sum for the cos integral
  double sin_integral; // sum for the sin integral
  double prevtime;     // previous time for use in cavity dynamics
  double cavity_pot;   // cavity potential energy
  double cavity_kin;   // cavity kinetic energy
  std::vector<double> prevdipole;
  void write_dipole(const int step);
  void write_cavity(const int step, const double time);
  void get_dipole(
    Box& box,
    Force& force,
    GPU_Vector<double>& dipole_);
  void get_dipole_jacobian(
    Box& box,
    Force& force,
    double displacement, 
    double charge);
  void _get_center_of_mass(GPU_Vector<double>& gpu_center_of_mass);
  void canonical_position(const double time);
  void canonical_momentum(const double time);
  void cavity_potential_energy();
  void cavity_kinetic_energy();
  void cavity_force();
  void step_cavity(const double time);
};
