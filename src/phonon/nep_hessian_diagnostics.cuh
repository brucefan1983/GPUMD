/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/
#pragma once
#include "sparse_hessian.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace nep_hessian {

struct Diagnostics {
  bool finite = true;
  double max_abs = 0.0;
  double frobenius = 0.0;
  double asymmetry_max = 0.0;
  double asymmetry_frobenius = 0.0;
  double translation_max = 0.0;
  double asymmetry_max_relative = 0.0;
  double asymmetry_frobenius_relative = 0.0;
  double translation_relative = 0.0;
};

// Visit each stored scalar once. Translation sums are accumulated once per
// atom and Cartesian component, also for sparse rows with many blocks.
inline Diagnostics inspect(
  const std::vector<double>& values, int atoms,
  const sparse_hessian::BlockPattern* pattern = nullptr)
{
  Diagnostics result;
  const size_t dimension = static_cast<size_t>(3) * atoms;
  if (values.size() != (pattern ? pattern->value_count() : dimension * dimension)) {
    result.finite = false;
    return result;
  }
  auto accumulate = [&](double value, double transposed) {
    result.finite = result.finite && std::isfinite(value) && std::isfinite(transposed);
    result.max_abs = std::max(result.max_abs, std::abs(value));
    result.frobenius += value * value;
    const double difference = value - transposed;
    result.asymmetry_max = std::max(result.asymmetry_max, std::abs(difference));
    result.asymmetry_frobenius += difference * difference;
  };
  if (pattern) {
    for (int atom = 0; atom < atoms; ++atom) {
      double sums[3][3] = {};
      for (int block = pattern->row_offsets[atom]; block < pattern->row_offsets[atom + 1]; ++block) {
        const int transpose = pattern->find_block(pattern->columns[block], atom);
        if (transpose < 0) {
          result.finite = false;
          return result;
        }
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            const double value = values[static_cast<size_t>(block) * 9 + a * 3 + b];
            accumulate(value, values[static_cast<size_t>(transpose) * 9 + b * 3 + a]);
            sums[a][b] += value;
          }
      }
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          result.translation_max = std::max(result.translation_max, std::abs(sums[a][b]));
    }
  } else {
    for (size_t row = 0; row < dimension; ++row) {
      double sums[3] = {};
      for (size_t column = 0; column < dimension; ++column) {
        const double value = values[row * dimension + column];
        accumulate(value, values[column * dimension + row]);
        sums[column / atoms] += value;
      }
      for (int axis = 0; axis < 3; ++axis)
        result.translation_max = std::max(result.translation_max, std::abs(sums[axis]));
    }
  }
  result.frobenius = std::sqrt(result.frobenius);
  result.asymmetry_frobenius = std::sqrt(result.asymmetry_frobenius);
  const double tiny = std::numeric_limits<double>::min();
  const double scale = std::max(result.max_abs, tiny);
  result.asymmetry_max_relative = result.asymmetry_max / scale;
  result.asymmetry_frobenius_relative = result.asymmetry_frobenius / std::max(result.frobenius, tiny);
  result.translation_relative = result.translation_max / scale;
  return result;
}

} // namespace nep_hessian
