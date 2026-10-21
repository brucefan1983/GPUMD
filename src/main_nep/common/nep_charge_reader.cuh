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
#include "nep_energy_reader.cuh"
#include "nep_charge_dataset.cuh"
#include "nep_charge_parameters.cuh"

class NEP_Charge_Reader : public NEP_Energy_Reader
{
public:
  NEP_Charge_Reader(NEP_Charge_Parameters& para, EnergyDifference& energy_difference,
    NEP_Batch_Data<NEP_Charge_Dataset>& charge_data)
    : NEP_Energy_Reader(para, energy_difference), charge_parameters(para), charge_data(charge_data) {}
  void prepare_dataset(Dataset& dataset) override;
  const char* energy_difference_conflict(const Structure& a, const Structure& b) const override;

private:
  NEP_Charge_Parameters& charge_parameters;
  NEP_Batch_Data<NEP_Charge_Dataset>& charge_data;
  std::vector<NEP_Charge_Labels> labels;
  void begin_reading() override { labels.clear(); }
  void read_structure(const XYZ_Record& record, Structure& structure) override;
  void print_summary(const std::vector<Structure>& structures) const override;
};
