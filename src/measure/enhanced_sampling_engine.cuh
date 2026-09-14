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
#include "enhanced_sampling_bias.cuh"
#include "enhanced_sampling_cv.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

class Atom;
class Box;

class EnhancedSamplingEngine
{
public:
  EnhancedSamplingEngine();

  EnhancedSamplingEngine(const EnhancedSamplingEngine&) = delete;
  EnhancedSamplingEngine& operator=(const EnhancedSamplingEngine&) = delete;
  EnhancedSamplingEngine(EnhancedSamplingEngine&&) = delete;
  EnhancedSamplingEngine& operator=(EnhancedSamplingEngine&&) = delete;

  void add_cv(std::unique_ptr<EnhancedSamplingCV> cv);
  void add_bias(std::unique_ptr<EnhancedSamplingBias> bias);

  bool has_cv(const std::string& name) const;
  bool has_bias(const std::string& name) const;
  int get_cv_index(const std::string& name) const;

  int get_number_of_cvs() const;
  int get_number_of_biases() const;

  void prepare(const int number_of_atoms);
  void calculate(const Box& box, Atom& atom, const bool collect_output_energy);
  void check();

  void write_header(FILE* fid) const;
  void write_output(FILE* fid, const int step);

private:
  int number_of_atoms_;
  std::vector<std::unique_ptr<EnhancedSamplingCV> > cvs_;
  std::vector<std::unique_ptr<EnhancedSamplingBias> > biases_;
  std::map<std::string, int> cv_indices_;
  std::map<std::string, int> bias_indices_;

  GPU_Vector<double> cv_values_;
  GPU_Vector<double> cv_derivatives_;
  GPU_Vector<double> bias_energies_;
  GPU_Vector<double> bias_derivatives_;
  GPU_Vector<double> total_bias_energy_;
  GPU_Vector<double> total_bias_virial_;
  GPU_Vector<double> potential_before_after_;

  std::vector<double> cpu_cv_values_;
  std::vector<double> cpu_cv_derivatives_;
  std::vector<double> cpu_bias_energies_;
  std::vector<double> cpu_bias_derivatives_;
  double cpu_total_bias_energy_;
  double cpu_total_bias_virial_[9];
  double cpu_potential_before_after_[2];
};
