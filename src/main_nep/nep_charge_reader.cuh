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
#include "structure.cuh"

class NEP_Charge_Reader : public NEP_Data_Reader
{
public:
  explicit NEP_Charge_Reader(Parameters& para) : NEP_Data_Reader(para) {}
  const char* energy_difference_conflict(const Structure& a, const Structure& b) const override;

private:
  void read_structure(const XYZ_Record& record, Structure& structure) const override;
  void print_summary(const std::vector<Structure>& structures) const override;
};
