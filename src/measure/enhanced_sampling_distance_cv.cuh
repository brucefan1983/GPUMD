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
#include "enhanced_sampling_cv.cuh"

class EnhancedSamplingDistanceCV : public EnhancedSamplingCV
{
public:
  EnhancedSamplingDistanceCV(
    const std::string& name,
    const int atom_i,
    const int atom_j,
    const bool use_pbc);

  EnhancedSamplingDistanceCV(const EnhancedSamplingDistanceCV&) = delete;
  EnhancedSamplingDistanceCV& operator=(const EnhancedSamplingDistanceCV&) = delete;
  EnhancedSamplingDistanceCV(EnhancedSamplingDistanceCV&&) = delete;
  EnhancedSamplingDistanceCV& operator=(EnhancedSamplingDistanceCV&&) = delete;

  void prepare(const int number_of_atoms) override;

  void compute(
    const Box& box,
    const GPU_Vector<double>& position_per_atom,
    double* g_value) override;

  void add_force_virial(
    const double* g_derivative,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    double* g_bias_virial) override;

  void check() override;

  int get_atom_i() const { return atom_i_; }
  int get_atom_j() const { return atom_j_; }
  bool use_pbc() const { return use_pbc_; }

private:
  int atom_i_;
  int atom_j_;
  bool use_pbc_;
  int number_of_atoms_;
  GPU_Vector<double> distance_data_;
  GPU_Vector<int> invalid_distance_;
};
