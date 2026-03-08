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
#include "property.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Group;
class Atom;

class RDF : public Property
{

public:

  struct RDF_Para {
    int num_types;           // number of atom types in model.xyz
    int num_RDFs;            // 1 + (num_type * (num_types + 1)) / 2
    int num_bins;            // number of bins in the RDFs
    double volume;           // volume could change during NPT simulations
    double rc;               // cutoff for RDF calculation
    double rc_square;        // rc * rc
    double dr;               // rc / num_bins
    int type_index[89];      // map of atom type from model.xyz to nep.txt
    int num_atoms[89];       // number of atoms for each atom type
    double density_global;   // N/volume
    double density_type[89]; // num_atoms[89]/volume
  };

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

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
      Force& force);

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

  void parse(
    const char** param,
    const int num_param,
    Box& box,
    const std::vector<int>& cpu_type_size,
    const int number_of_steps);

  RDF(
    const char** param,
    const int num_param,
    Box& box,
    const std::vector<int>& cpu_type_size,
    const int number_of_steps);

private:
  int sampling_interval_ = 100;
  GPU_Vector<double> rdf_g_;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  RDF_Para rdf_para;

  void find_rdf(Box& box, const GPU_Vector<int>& type, const GPU_Vector<double>& position);
};