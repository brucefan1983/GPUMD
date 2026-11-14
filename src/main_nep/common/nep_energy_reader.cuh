/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "main_nep/structure.cuh"
#include "nep_parameters.cuh"
class EnergyDifference;

class NEP_Energy_Reader : public NEP_Data_Reader
{
public:
  NEP_Energy_Reader(NEP_Energy_Parameters& para, EnergyDifference& energy_difference)
    : NEP_Data_Reader(para), energy_parameters(para), energy_difference(energy_difference) {}
  bool uses_grouped_batches() const override;
  void prepare_train(std::vector<Structure>& structures, int batch_size,
    int& num_batches, std::vector<int>& batch_sizes) override;
  void prepare_test(const std::vector<Structure>& structures, bool has_test_set) override;
  void prepare_dataset(Dataset& dataset) override;
  // Return the reason to reject a pair in ediff.in, or nullptr if it is allowed.
  virtual const char* energy_difference_conflict(const Structure& a, const Structure& b) const;

protected:
  void read_structure(const XYZ_Record& record, Structure& structure) override;

private:
  NEP_Energy_Parameters& energy_parameters;
  EnergyDifference& energy_difference;
};
