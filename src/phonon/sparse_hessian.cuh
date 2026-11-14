/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/
#pragma once

#include <array>
#include <cmath>
#include <algorithm>
#include "utilities/error.cuh"
#include <vector>

namespace sparse_hessian {

struct BlockPattern {
  int atoms = 0;
  std::vector<int> row_offsets;
  std::vector<int> columns;

  size_t block_count() const { return columns.size(); }
  size_t value_count() const { return columns.size() * 9; }

  int find_block(const int row, const int column) const
  {
    const auto first = columns.begin() + row_offsets[row];
    const auto last = columns.begin() + row_offsets[row + 1];
    const auto found = std::lower_bound(first, last, column);
    return found != last && *found == column
      ? static_cast<int>(found - columns.begin()) : -1;
  }

  void symmetrize(std::vector<double>& values) const
  {
    if (values.size() != value_count())
      PRINT_INPUT_ERROR("invalid sparse Hessian value count");
    for (int atom = 0; atom < atoms; ++atom)
      for (int block = row_offsets[atom]; block < row_offsets[atom + 1]; ++block) {
        const int other = columns[block];
        const int transpose = find_block(other, atom);
        if (transpose < 0)
          PRINT_INPUT_ERROR("sparse Hessian pattern is not symmetric");
        if (block > transpose) continue;
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            const size_t index = static_cast<size_t>(block) * 9 + a * 3 + b;
            const size_t transpose_index = static_cast<size_t>(transpose) * 9 + b * 3 + a;
            const double average = 0.5 * (values[index] + values[transpose_index]);
            values[index] = average;
            values[transpose_index] = average;
          }
      }
  }
};

inline BlockPattern build_pattern_from_neighbors(
  const int atoms,
  const std::vector<int>& radial_counts,
  const std::vector<int>& radial_list,
  const size_t radial_capacity,
  const std::vector<int>& angular_counts,
  const std::vector<int>& angular_list,
  const size_t angular_capacity)
{
  if (atoms <= 0 || radial_counts.size() != static_cast<size_t>(atoms) ||
      angular_counts.size() != static_cast<size_t>(atoms) ||
      radial_capacity > radial_list.size() / static_cast<size_t>(atoms) ||
      angular_capacity > angular_list.size() / static_cast<size_t>(atoms))
    PRINT_INPUT_ERROR("invalid sparse Hessian neighbor lists");
  // Store each unique environment once and map atoms back to its centers.
  // Row assembly below uses reusable marks rather than retaining repeated
  // copies of every environment in every row.
  std::vector<std::vector<int>> environments(atoms), centers_for_atom(atoms);
  for (int center = 0; center < atoms; ++center) {
    auto& environment = environments[center];
    environment.push_back(center);
    const int radial_count = radial_counts[center];
    const int angular_count = angular_counts[center];
    if (radial_count < 0 || static_cast<size_t>(radial_count) > radial_capacity ||
        angular_count < 0 || static_cast<size_t>(angular_count) > angular_capacity)
      PRINT_INPUT_ERROR("invalid sparse Hessian neighbor count");
    for (int slot = 0; slot < radial_count; ++slot)
      environment.push_back(radial_list[center + static_cast<size_t>(atoms) * slot]);
    for (int slot = 0; slot < angular_count; ++slot)
      environment.push_back(angular_list[center + static_cast<size_t>(atoms) * slot]);
    for (const int atom : environment)
      if (atom < 0 || atom >= atoms)
        PRINT_INPUT_ERROR("sparse Hessian neighbor index is out of range");
    std::sort(environment.begin(), environment.end());
    environment.erase(std::unique(environment.begin(), environment.end()), environment.end());
    for (const int atom : environment)
      centers_for_atom[atom].push_back(center);
  }
  BlockPattern result;
  result.atoms = atoms;
  result.row_offsets.reserve(static_cast<size_t>(atoms) + 1);
  result.row_offsets.push_back(0);
  std::vector<int> marks(atoms, -1), row;
  for (int atom = 0; atom < atoms; ++atom) {
    row.clear();
    for (const int center : centers_for_atom[atom])
      for (const int other : environments[center])
        if (marks[other] != atom) {
          marks[other] = atom;
          row.push_back(other);
        }
    std::sort(row.begin(), row.end());
    if (row.empty())
      PRINT_INPUT_ERROR("invalid sparse Hessian CSR row");
    result.columns.insert(result.columns.end(), row.begin(), row.end());
    if (result.columns.size() > static_cast<size_t>(0x7fffffff))
      PRINT_INPUT_ERROR("sparse Hessian pattern exceeds supported size");
    result.row_offsets.push_back(static_cast<int>(result.columns.size()));
  }
  return result;
}

__device__ __forceinline__ int find_block(
  const int row_atom,
  const int column_atom,
  const int* row_offsets,
  const int* columns)
{
  int first = row_offsets[row_atom];
  int last = row_offsets[row_atom + 1];
  while (first < last) {
    const int middle = first + (last - first) / 2;
    const int value = columns[middle];
    if (value < column_atom) first = middle + 1;
    else last = middle;
  }
  return first < row_offsets[row_atom + 1] && columns[first] == column_atom
    ? first : -1;
}

__device__ __forceinline__ void atomic_add(
  double* values,
  const int* row_offsets,
  const int* columns,
  const int atoms,
  const int row,
  const int column,
  const double value)
{
  const int row_atom = row % atoms;
  const int column_atom = column % atoms;
  const int block = find_block(row_atom, column_atom, row_offsets, columns);
  if (block >= 0) {
    const int row_axis = row / atoms;
    const int column_axis = column / atoms;
    atomicAdd(
      &values[static_cast<size_t>(block) * 9 + row_axis * 3 + column_axis],
      value);
  }
}

__device__ __forceinline__ void atomic_add_any(
  double* values,
  const int* row_offsets,
  const int* columns,
  const int atoms,
  const int row,
  const int column,
  const double value)
{
  if (row_offsets == nullptr || columns == nullptr) {
    atomicAdd(&values[row * (3 * atoms) + column], value);
    return;
  }
  atomic_add(values, row_offsets, columns, atoms, row, column, value);
}

} // namespace sparse_hessian
