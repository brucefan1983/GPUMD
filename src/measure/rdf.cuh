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
#include "utilities/gpu_vector.cuh"
#include <vector>

class Group;
class Atom;

class RDF
{

public:
  bool compute_ = false;
  double r_cut_ = 8.0;
  int rdf_bins_ = 100;
  double r_step_;
  int num_last_steps_ = 5000;
  int num_every_ = 100;
  int num_repeat_ = 50;
  int atom_id1_[6] = {-1,-1,-1,-1,-1,-1};
  int atom_id2_[6] = {-1,-1,-1,-1,-1,-1};


  void preprocess(const bool is_pimd, const int number_of_beads, const int num_atoms, std::vector<int>& cpu_type_size, const double box); 
  void process(
    const bool is_pimd,
    const int number_of_steps, 
    const int step, 
    Box& box,
    Atom& atom);
  void postprocess(const bool is_pimd, const int number_of_beads);
  void parse(const char** param, const int num_param, const int number_of_types,  const int number_of_steps);
  
private:
  int num_atoms_;
  int rdf_atom_count = 1;
  int rdf_N_;
  std::vector<int> atom_id1_typesize;
  std::vector<int> atom_id2_typesize;
  std::vector<double> density1;
  std::vector<double> density2;
  std::vector<double> rdf_;
  std::vector<double> rdf_average_;

  GPU_Vector<double> rdf_g_;
  GPU_Vector<double> radial_;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;

  void find_rdf(
    const int bead,
    const int rdf_atom_count,
    const int rdf_atom_,
    int* atom_id1_,
    int* atom_id2_,
    std::vector<int>& atom_id1_typesize,
    std::vector<int>& atom_id2_typesize,
    std::vector<double>& density1,
    std::vector<double>& density2,
    double rc,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& cell_count,
    GPU_Vector<int>& cell_count_sum,
    GPU_Vector<int>& cell_contents,
    int num_bins_0,
    int num_bins_1,
    int num_bins_2,
    const double rc_inv_cell_list,
    GPU_Vector<double>& radial_,
    GPU_Vector<double>& rdf_g_, 
    const int rdf_bins_,
    const double r_step_);
};