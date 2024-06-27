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
#include <string>
#include <vector>
class Box;
class Atom;
class Force;
class Integrate;

class Cavity
{
public:
  void parse(const char** param, int num_param);
  void preprocess(
    const int number_of_atoms, 
    const int number_of_potentials, 
    Box& box,
    Atom& atom,
    Force& force);
  void process(
    int step,
    const double global_time,
    const int number_of_atoms_fixed,
    std::vector<Group>& group,
    Box& box,
    Atom& atom,
    Force& force);
  void postprocess();

private:
  bool enabled_ = false;
  int dump_interval_ = 1;
  FILE* file_;
  GPU_Vector<double> gpu_dipole_;
  std::vector<double> cpu_dipole_;
  GPU_Vector<double> gpu_dipole_jacobian_;
  std::vector<double> cpu_dipole_jacobian_;
  std::vector<double> masses_;
  double mass_;
  double coupling_strength;
  double cavity_frequency; 
  int charge;
  double q0; // Initial cavity coordinate, q_0
  double q; // cavity coordinate, q(t)
  double cos_integral;
  double sin_integral;
  double prevtime;
  std::vector<double> prev_dipole;
  void write_dipole(const int step);
  void cavity_force(const int step);
  void get_dipole(
    Box& box,
    Force& force,
    GPU_Vector<double>& dipole_);
  void get_dipole_jacobian(
    Box& box,
    Force& force,
    int number_of_atoms,
    double displacement, 
    double charge);
  void _get_center_of_mass(GPU_Vector<double>& gpu_center_of_mass);
  Atom atom_copy;
};
