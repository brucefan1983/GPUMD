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

#include "nep_energy_reader.cuh"
#include "energy_difference.cuh"
#include "nep_parameters.cuh"
#include "main_nep/xyz_reader.cuh"
#include <cmath>

void NEP_Energy_Reader::read_structure(const XYZ_Record& record, Structure& structure)
{
  NEP_Data_Reader::read_structure(record, structure);
  if (uses_grouped_batches()) {
    structure.name = EnergyDifference::read_structure_name(
      record.comment, record.filename, record.comment_line_number);
  }
}

bool NEP_Energy_Reader::uses_grouped_batches() const
{
  return para.prediction == 0 && energy_parameters.lambda_d > 0.0f;
}

void NEP_Energy_Reader::prepare_train(std::vector<Structure>& structures, int batch_size,
  int& num_batches, std::vector<int>& batch_sizes)
{
  energy_difference.read_train(energy_parameters, structures, batch_size, num_batches, batch_sizes, *this);
}

void NEP_Energy_Reader::prepare_test(
  const std::vector<Structure>& structures, bool has_test_set)
{
  energy_difference.read_test(energy_parameters, structures, has_test_set, *this);
}

void NEP_Energy_Reader::prepare_dataset(Dataset& dataset)
{
  int num_virial_configurations = 0;
  for (const auto& structure : dataset.structures) {
    num_virial_configurations += structure.has_virial;
  }
  printf("Number of configurations having virial = %d.\n", num_virial_configurations);
  if (energy_parameters.has_ediff_combinations) {
    energy_difference.prepare_dataset(dataset);
  }
}

const char* NEP_Energy_Reader::energy_difference_conflict(
  const Structure& structure_a, const Structure& structure_b) const
{
  const float tolerance = 1.0e-5f;
  if (
    structure_a.num_atom != structure_b.num_atom || structure_a.type != structure_b.type ||
    structure_a.pbc != structure_b.pbc) {
    return nullptr;
  }
  for (int d = 0; structure_a.pbc && d < 9; ++d) {
    if (std::fabs(structure_a.box_original[d] - structure_b.box_original[d]) > tolerance) {
      return nullptr;
    }
  }
  for (int n = 0; n < structure_a.num_atom; ++n) {
    if (
      std::fabs(structure_a.x[n] - structure_b.x[n]) > tolerance ||
      std::fabs(structure_a.y[n] - structure_b.y[n]) > tolerance ||
      std::fabs(structure_a.z[n] - structure_b.z[n]) > tolerance) {
      return nullptr;
    }
  }
  return "have the same geometry, for which the model predicts the same energy.";
}
