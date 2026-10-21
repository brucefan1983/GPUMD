#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace fitness_spin {

// A local displacement for differentiating a sampled spin path. The sphere log
// keeps fixed-length rotations tangent, while the separate radial difference
// retains genuine changes in the converged moment magnitude. Neighbouring
// samples must resolve the path (angular separation strictly below pi).
inline bool rotation_displacement(
  const std::array<double, 3>& center,
  const std::array<double, 3>& neighbour,
  std::array<double, 3>& displacement)
{
  double r2 = 0.0, q2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    r2 += center[d] * center[d];
    q2 += neighbour[d] * neighbour[d];
  }
  const double r = std::sqrt(r2), q = std::sqrt(q2);
  // At zero moment there is no sphere tangent; retain the Cartesian limit.
  if (r <= 1.0e-12 || q <= 1.0e-12) {
    for (int d = 0; d < 3; ++d) displacement[d] = neighbour[d] - center[d];
    return true;
  }
  double cosine = 0.0;
  for (int d = 0; d < 3; ++d) cosine += (center[d] / r) * (neighbour[d] / q);
  cosine = std::max(-1.0, std::min(1.0, cosine));
  std::array<double, 3> tangent;
  double sine2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    tangent[d] = neighbour[d] / q - cosine * center[d] / r;
    sine2 += tangent[d] * tangent[d];
  }
  const double sine = std::sqrt(sine2);
  if (cosine < 0.0 && sine < 1.0e-10) return false;
  const double factor = sine > 1.0e-12 ? std::atan2(sine, cosine) / sine : 1.0;
  for (int d = 0; d < 3; ++d) {
    displacement[d] = r * factor * tangent[d] + (q - r) * center[d] / r;
  }
  return true;
}

} // namespace fitness_spin
