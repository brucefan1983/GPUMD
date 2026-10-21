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
#include <string>
#include <vector>

struct Structure;

// Fit energy_total / num_atom to the element fractions, with squared-error weight
// energy_weight * weight^2. Returns the minimum-norm reference energies in eV/atom
// and the numerical rank. Zero-weight structures do not contribute to the fit.
// On failure, reference is empty, rank is zero, and error describes the cause.
bool fit_nep_energy_reference(
  const std::vector<Structure>& structures,
  int num_types,
  std::vector<double>& reference,
  int& rank,
  std::string& error);
