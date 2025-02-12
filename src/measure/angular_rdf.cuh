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
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Group;
class Atom;

class AngularRDF
{
public:
  bool compute_ = false;

  double r_cut_ = 8.0;

  int rdf_r_bins_ = 100;

  int rdf_theta_bins_ = 100;

  double r_step_;

  double theta_step_; // angular step size

  int num_interval_ = 100; // sampling interval step

  int atom_id1_[6] = {-1, -1, -1, -1, -1, -1};
  int atom_id2_[6] = {-1, -1, -1, -1, -1, -1};

  void preprocess(
    const bool is_pimd,
    const int number_of_beads,
    const int num_atoms,
    std::vector<int>& cpu_type_size);

  void process(const bool is_pimd, const int number_of_steps, const int step, Box& box, Atom& atom);

  void postprocess(const bool is_pimd, const int number_of_beads);

  void parse(
    const char** param,
    const int num_param,
    Box& box,
    const int number_of_types,
    const int number_of_steps);

private:
  int num_atoms_;
  int rdf_atom_count = 1;
  int rdf_N_;
  int num_repeat_ = 0;

  std::vector<int> atom_id1_typesize;
  std::vector<int> atom_id2_typesize;

  std::vector<double> density1;
  std::vector<double> density2;

  std::vector<double> rdf_;

  GPU_Vector<double> rdf_g_;
  GPU_Vector<double> radial_;
  GPU_Vector<double> theta_;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;

  // Core function to calculate angular-dependent RDF
  void find_angular_rdf(
    const int bead,           // PIMD bead index
    const int rdf_atom_count, // Number of atom pairs
    const int rdf_atom_,      // Current atom pair index
    int* atom_id1_,           // Array of first atom type IDs
    int* atom_id2_,           // Array of second atom type IDs
    std::vector<int>& atom_id1_typesize,
    std::vector<int>& atom_id2_typesize,
    std::vector<double>& density1,
    std::vector<double>& density2,
    double rc,                                   // Cutoff radius
    Box& box,                                    // Simulation box
    const GPU_Vector<int>& type,                 // Atom types
    const GPU_Vector<double>& position_per_atom, // Atom positions
    GPU_Vector<int>& cell_count,
    GPU_Vector<int>& cell_count_sum,
    GPU_Vector<int>& cell_contents,
    int num_bins_0,
    int num_bins_1,
    int num_bins_2,
    const double rc_inv_cell_list, // Inverse of cell list cutoff
    GPU_Vector<double>& radial_,   // Radial distance array
    GPU_Vector<double>& theta_,    // Angular distance array
    GPU_Vector<double>& rdf_g_,    // RDF results
    const int rdf_r_bins_,         // Number of radial bins
    const int rdf_theta_bins_,     // Number of angular bins
    const double r_step_,          // Radial step size
    const double theta_step_       // Angular step size
  );
};