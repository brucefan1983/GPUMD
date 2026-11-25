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
/*
    Contributors
    OpenAI Codex: Initial implementation
*/

#include "force/neighbor.cuh"
#include "meam.cuh"
#include "meam_functions.cuh"
#include "meam_parameters.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace
{

constexpr int block_size = 128;
constexpr int cached_neighbors = 64;
constexpr int screening_atoms_per_block = 4;
constexpr int screening_threads_per_atom = 32;
// Distances are in angstroms. The skin is extra search space for list reuse.
constexpr double neighbor_skin = 0.35;
// Changes in the three triangle lengths are bounded independently of the
// Verlet list. Within this bound, the cached masks omit only saturated terms.
constexpr double screen_certificate_skin = 0.19;

struct Vec {
  double x, y, z;

  __device__ Vec operator+(Vec v) const { return {x + v.x, y + v.y, z + v.z}; }
  __device__ Vec operator-(Vec v) const { return {x - v.x, y - v.y, z - v.z}; }
  __device__ Vec operator*(double s) const { return {x * s, y * s, z * s}; }
  __device__ double square() const { return x * x + y * y + z * z; }
  __device__ double dot(Vec v) const { return x * v.x + y * v.y + z * v.z; }
};

// With a normal float seed, two Newton steps recover double precision.
__device__ double reciprocal(double x)
{
  const float reduced = static_cast<float>(x);
  if (!(fabsf(reduced) >= 0x1p-120f && fabsf(reduced) <= 0x1p120f)) {
    return 1.0 / x;
  }
  double result = 1.0f / reduced;
  result = fma(fma(-x, result, 1.0), result, result);
  return fma(fma(-x, result, 1.0), result, result);
}

struct Positions {
  const double* xyz;
  int size;

  __device__ Vec at(int i) const { return {xyz[i], xyz[i + size], xyz[i + 2 * size]}; }

  __device__ Vec displacement(const Box& box, int i, int j) const
  {
    Vec v = at(j) - at(i);
    apply_mic(box, v.x, v.y, v.z);
    return v;
  }
};

struct ScreeningLimits {
  MeamScreen precise;
  float minimum;
  float maximum;
};

struct Model {
  MeamSettings settings;
  const MeamElement* element;
  const ScreeningLimits* screens;
  const double* bounds;
  const MeamSpline* tables;
  const double* density_table;
  const double* radial_exponents;
  const double* inverse_re;
  const int* tabulate_density;
  double inverse_smoothing_width;

  __device__ MeamValueSlope pair(int index, double r) const
  {
    const double coordinate = r * settings.inverse_table_step;
    const int knot = static_cast<int>(coordinate);
    const double x = coordinate - knot;
    const auto c = tables[index * MEAM_TABLE_POINTS + knot];
    return {
      c.value + x * (c.slope + x * (c.quadratic + x * c.cubic)),
      (c.slope + x * (2.0 * c.quadratic + 3.0 * x * c.cubic)) * settings.inverse_table_step};
  }
};

struct CellList {
  int nx, ny, nz;
  const int* count;
  const int* offset;
  const int* atoms;
};

struct Neighbors {
  const size_t* offset;
  const int* atom;
  const size_t* reverse;
  const int* near_counts;
  double guaranteed_far_distance_squared;
};

struct Geometry {
  Vec displacement;
  double squared_distance;
  int atom;
  int type;
  float3 displacement32;
  float squared_distance32;
};

__device__ Geometry
geometry_for(const Box& box, Positions positions, int center, int atom, const int* type)
{
  const Vec v = positions.displacement(box, center, atom);
  const double squared = v.square();
  return {
    v,
    squared,
    atom,
    type[atom],
    {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)},
    static_cast<float>(squared)};
}

__device__ Geometry get_geometry(
  const Box& box,
  Positions positions,
  Neighbors neighbors,
  int center,
  const int* type,
  const Geometry* cache,
  size_t index,
  bool cached)
{
  if (cached) {
    return cache[index - neighbors.offset[center]];
  }
  return geometry_for(box, positions, center, neighbors.atom[index], type);
}

struct MotionCheck {
  double affine[9];
  double tolerance_squared;
};

template <bool CheckSpecies = false>
__global__ void check_neighbor_motion(
  Box box,
  Positions positions,
  const double* reference,
  MotionCheck check,
  int* rebuild,
  const int* species = nullptr,
  const int* reference_species = nullptr)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < positions.size) {
    if (CheckSpecies) {
      if (species[atom] != reference_species[atom]) {
        atomicExch(rebuild, 1);
        return;
      }
    }
    const Positions old{reference, positions.size};
    const Vec x = old.at(atom);
    const auto a = check.affine;
    const Vec predicted{
      a[0] * x.x + a[1] * x.y + a[2] * x.z,
      a[3] * x.x + a[4] * x.y + a[5] * x.z,
      a[6] * x.x + a[7] * x.y + a[8] * x.z};
    Vec delta = positions.at(atom) - predicted;
    apply_mic(box, delta.x, delta.y, delta.z);
    if (delta.square() > check.tolerance_squared) {
      atomicExch(rebuild, 1);
    }
  }
}

double neighbor_motion_limit(const Box& current, const Box& old, MotionCheck& check)
{
  double distortion_squared = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k) {
        value += current.cpu_h[row * 3 + k] * old.cpu_h[9 + k * 3 + col];
      }
      check.affine[row * 3 + col] = value;
      const double delta = value - (row == col ? 1.0 : 0.0);
      distortion_squared += delta * delta;
    }
  }
  // The Frobenius norm bounds the contraction of any periodic image vector.
  return sqrt(distortion_squared);
}

__global__ void reverse_neighbors(
  int atoms, const size_t* offset, const int* neighbors, size_t* reverse, int* error)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < atoms) {
    for (size_t e = offset[i]; e < offset[i + 1]; ++e) {
      const int j = neighbors[e];
      size_t r = offset[j];
      while (r < offset[j + 1] && neighbors[r] != i) {
        ++r;
      }
      if (r == offset[j + 1]) {
        atomicMax(error, 2);
        r = e;
      }
      reverse[e] = r;
    }
  }
}

struct Density {
  double radial;
  double weight[3], weight2[3];
  double dipole[3], quadrupole[6], octupole[10];
  double quadrupole_trace, octupole_trace[3];
};

struct BondGradient {
  Vec displacement;
  double screening;
};

__global__ void make_neighbors(
  Box box,
  Positions positions,
  CellList cells,
  double cutoff,
  double near2,
  int* near_counts,
  size_t* counts,
  const size_t* offsets,
  int* neighbors,
  int* overlap)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= positions.size) {
    return;
  }
  const auto pos = positions.at(i);
  int cx, cy, cz, cell;
  find_cell_id(
    box, pos.x, pos.y, pos.z, 1.0 / cutoff, cells.nx, cells.ny, cells.nz, cx, cy, cz, cell);
  int bx, ex, by, ey, bz, ez;
  const int sx = min(3, cells.nx);
  const int sy = min(3, cells.ny);
  const int sz = min(3, cells.nz);
  bx = -sx / 2;
  ex = bx + sx;
  by = -sy / 2;
  ey = by + sy;
  bz = -sz / 2;
  ez = bz + sz;
  size_t count = 0;
  int near_index = 0;
  int far_index = neighbors ? near_counts[i] : 0;
  for (int iz = bz; iz < ez; ++iz) {
    const int z = (cz + iz + cells.nz) % cells.nz;
    for (int iy = by; iy < ey; ++iy) {
      const int y = (cy + iy + cells.ny) % cells.ny;
      for (int ix = bx; ix < ex; ++ix) {
        const int x = (cx + ix + cells.nx) % cells.nx;
        const int c = (z * cells.ny + y) * cells.nx + x;
        const int end = cells.offset[c] + cells.count[c];
        for (int k = cells.offset[c]; k < end; ++k) {
          const int j = cells.atoms[k];
          if (i == j) {
            continue;
          }
          const double r2 = positions.displacement(box, i, j).square();
          if (r2 < cutoff * cutoff) {
            const bool close = r2 < near2;
            if (neighbors) {
              neighbors[offsets[i] + (close ? near_index++ : far_index++)] = j;
            } else {
              near_index += close;
              if (r2 == 0.0) {
                atomicMax(overlap, 1);
              }
            }
            ++count;
          }
        }
      }
    }
  }
  if (counts) {
    counts[i] = count;
    near_counts[i] = near_index;
  }
}

__device__ Geometry
relative_geometry(const Box& box, Positions positions, Vec center, int atom, const int* type)
{
  Vec v{
    positions.xyz[atom] - center.x,
    positions.xyz[atom + positions.size] - center.y,
    positions.xyz[atom + 2 * positions.size] - center.z};
  apply_mic(box, v.x, v.y, v.z);
  const double squared = v.square();
  return {
    v,
    squared,
    atom,
    type[atom],
    {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)},
    static_cast<float>(squared)};
}

__device__ bool owns_bond(Vec v)
{
  if (v.z != 0.0) {
    return v.z > 0.0;
  }
  if (v.y != 0.0) {
    return v.y > 0.0;
  }
  return v.x > 0.0;
}

// The third atom and both bond endpoints use the same periodic triangle.
__device__ MeamValueSlope screening_term(
  const Geometry& bond,
  const Geometry& third,
  double inverse_r2,
  double bound,
  ScreeningLimits limits,
  Vec* dv,
  Vec* dw)
{
  const double ik = third.squared_distance;
  if (ik > bound) {
    return {1.0, 0.0};
  }
  const Vec v = bond.displacement;
  const Vec w = third.displacement;
  const Vec wj = w - v;
  const double jk = wj.square();
  if (jk > bound) {
    return {1.0, 0.0};
  }
  const double x = ik * inverse_r2;
  const double y = jk * inverse_r2;
  const double diff = x - y;
  const double a = 1.0 - diff * diff;
  if (a <= 0.0) {
    return {1.0, 0.0};
  }
  const double sum = x + y - 1.0;
  const double interval =
    (2.0 * sum + (1.0 - limits.precise.minimum) * a) * limits.precise.inverse_width;
  // Pairs outside the smoothing interval do not need a division.
  if (interval <= 0.0) {
    return {0.0, 0.0};
  }
  if (interval >= a) {
    return {1.0, 0.0};
  }
  const double inverse_a = reciprocal(a);
  const auto s = meam_cutoff(interval * inverse_a);
  if (dv && s.slope != 0.0) {
    const double skew = 4.0 * sum * diff * inverse_a * inverse_a;
    const double cx = 2.0 * inverse_a + skew;
    const double cy = 2.0 * inverse_a - skew;
    const double scale = 2.0 * limits.precise.inverse_width * inverse_r2;
    *dv = (v * (-cx * x - cy * y) - wj * cy) * scale;
    *dw = (w * cx + wj * cy) * scale;
  }
  return s;
}

struct ScreeningTest {
  float3 displacement;
  float squared_distance;
  float bound;
  float r4;
  float guard;
  bool safe_scale;
};

__device__ int
saturated_screening(const ScreeningTest& test, const Geometry& third, ScreeningLimits limits)
{
  // The float filter only accepts a saturated result well outside roundoff.
  // Extreme distances or bounds are classified using the double-precision path.
  if (!test.safe_scale) {
    return -1;
  }
  const float ik = third.squared_distance32;
  if (ik > test.bound + test.guard) {
    return 1;
  }
  const auto w = third.displacement32;
  const float dx = w.x - test.displacement.x;
  const float dy = w.y - test.displacement.y;
  const float dz = w.z - test.displacement.z;
  const float jk = dx * dx + dy * dy + dz * dz;
  const float difference = ik - jk;
  if (jk > test.bound + test.guard || fabsf(difference) > test.squared_distance + test.guard) {
    return 1;
  }
  // Multiplying the normalized screening formula by r_ij^4 avoids forming
  // ratios for triangles whose screening is exactly zero or one.
  const float a = test.r4 - difference * difference;
  const float margin = test.guard * test.squared_distance;
  if (a <= margin || !isfinite(a)) {
    return -1;
  }
  const float numerator = 2.0f * (ik + jk - test.squared_distance) * test.squared_distance;
  const float low = limits.minimum;
  const float high = limits.maximum;
  if (fabsf(high) <= 32.0f && numerator - high * a > margin * (3.0f + fabsf(high))) {
    return 1;
  }
  if (fabsf(low) <= 32.0f && numerator - low * a < -margin * (3.0f + fabsf(low))) {
    return 0;
  }
  return -1;
}

struct DistanceInterval {
  float low, high;
};

__device__ DistanceInterval bounded_square(float distance_squared, float shift)
{
  const float r = sqrtf(distance_squared);
  const float error = shift * (2.0f * r + shift);
  return {fmaxf(0.0f, distance_squared - error), distance_squared + error};
}

__device__ int certify_screen(
  DistanceInterval ij,
  DistanceInterval ik,
  DistanceInterval jk,
  float bound_factor,
  ScreeningLimits limits)
{
  const float guard = 0.0004f * (ij.high + ik.high + jk.high);
  const float reach = bound_factor * ij.high;
  const float d_min = fmaxf(0.0f, fmaxf(ik.low - jk.high, jk.low - ik.high));
  if (ik.low > reach + guard || jk.low > reach + guard || d_min > ij.high + guard) {
    return 1;
  }
  if (!(ij.low > 1.e-12f && ij.high < 1.e12f && fabsf(limits.minimum) <= 32.0f &&
        fabsf(limits.maximum) <= 32.0f)) {
    return -1;
  }
  // Bound the normalized geometry over all lengths in the three intervals.
  const float inv_hi = 1.f / ij.high;
  const float xpy_min = (ik.low + jk.low) * inv_hi;
  const float q_min = d_min * inv_hi;
  const float a_max = 1.f - q_min * q_min;
  const float margin = guard / ij.low;
  const float cmax = fmaxf(limits.maximum, 0.0f);
  if (2.f * (xpy_min - 1.f) - cmax * a_max > margin * (3.f + cmax)) {
    return 1;
  }
  // Certify a blocker only if screening is defined and in range for the entire interval.
  if (ik.high + guard < bound_factor * ij.low && jk.high + guard < bound_factor * ij.low) {
    const float d_max = fmaxf(ik.high - jk.low, jk.high - ik.low);
    if (d_max + guard < ij.low) {
      const float inv_lo = 1.f / ij.low;
      const float q_max = d_max * inv_lo;
      const float a_min = 1.f - q_max * q_max;
      const float cmin = limits.minimum;
      const float a = cmin >= 0.f ? a_min : a_max;
      if (2.f * ((ik.high + jk.high) * inv_lo - 1.f) - cmin * a < -margin * (3.f + fabsf(cmin))) {
        return 0;
      }
    }
  }
  return -1;
}

struct CertificateGeometry {
  float3 displacement;
  DistanceInterval distance;
  int type;
};

__global__ void build_screen_certificates(
  Box box,
  Positions positions,
  Neighbors neighbors,
  const int* type,
  Model model,
  unsigned long long* certificates)
{
  constexpr int atoms_per_block = screening_atoms_per_block;
  constexpr int threads_per_atom = screening_threads_per_atom;
  const int group = threadIdx.x / threads_per_atom;
  const int lane = threadIdx.x % threads_per_atom;
  const int i = blockIdx.x * atoms_per_block + group;
  if (i >= positions.size) {
    return;
  }
  const auto start = neighbors.offset[i];
  const int count = neighbors.offset[i + 1] - start;
  if (count > cached_neighbors) {
    return;
  }
  constexpr float shift = static_cast<float>(screen_certificate_skin);
  __shared__ CertificateGeometry shared[atoms_per_block][cached_neighbors];
  auto* cache = shared[group];
  for (int q = lane; q < count; q += threads_per_atom) {
    const auto candidate = geometry_for(box, positions, i, neighbors.atom[start + q], type);
    cache[q] = {
      candidate.displacement32,
      bounded_square(candidate.squared_distance32, shift),
      candidate.type};
  }
  __syncwarp();
  const int n = model.settings.type_count;
  const float cutoff = static_cast<float>(model.settings.cutoff);
  for (int q = lane; q < count; q += threads_per_atom) {
    const auto bond = cache[q];
    unsigned long long mask = 0;
    if (bond.distance.low < cutoff * cutoff) {
      const int pair = type[i] * n + bond.type;
      const float bound_factor = static_cast<float>(model.bounds[pair]);
      for (int k = 0; k < count; ++k) {
        if (k == q) {
          continue;
        }
        const auto third = cache[k];
        if (
          third.distance.low > bound_factor * bond.distance.high +
                                 0.0004f * (third.distance.high + bond.distance.high)) {
          continue;
        }
        const float x = third.displacement.x - bond.displacement.x;
        const float y = third.displacement.y - bond.displacement.y;
        const float z = third.displacement.z - bond.displacement.z;
        const auto jk = bounded_square(x * x + y * y + z * z, shift);
        const int code = certify_screen(
          bond.distance, third.distance, jk, bound_factor, model.screens[pair * n + third.type]);
        if (code == 0) {
          // Use the otherwise unused j slot to identify a permanently blocked bond.
          mask = 1ull << q;
          break;
        }
        if (code < 0) {
          mask |= 1ull << k;
        }
      }
    }
    certificates[start + q] = mask;
  }
}

__global__ void screen_bonds(
  Box box,
  Positions positions,
  Neighbors neighbors,
  const int* type,
  Model model,
  double* screening,
  unsigned long long* partial,
  unsigned char* previous_blocker,
  const unsigned long long* certificates,
  int* overlap,
  int* screened_atoms,
  int* screened_edges,
  int* active_count)
{
  constexpr int atoms_per_block = screening_atoms_per_block;
  constexpr int threads_per_atom = screening_threads_per_atom;
  const int group = threadIdx.x / threads_per_atom;
  const int lane = threadIdx.x % threads_per_atom;
  const int i = blockIdx.x * atoms_per_block + group;
  const int atoms = positions.size;
  if (i >= atoms) {
    return;
  }
  const int n = model.settings.type_count;
  const size_t start = neighbors.offset[i];
  const size_t end = neighbors.offset[i + 1];
  const bool cached = end - start <= cached_neighbors;
  __shared__ Geometry geometry_by_group[atoms_per_block][cached_neighbors];
  __shared__ int edges_by_group[atoms_per_block][cached_neighbors];
  auto* cache = geometry_by_group[group];
  auto* active_edges = edges_by_group[group];
  int work = end - start;
  const double cutoff2 = model.settings.cutoff * model.settings.cutoff;
  if (cached) {
    work = 0;
    for (int offset = 0; offset < end - start; offset += threads_per_atom) {
      const size_t edge = start + offset + lane;
      bool selected = false;
      if (edge < end) {
        const auto geometry = geometry_for(box, positions, i, neighbors.atom[edge], type);
        cache[offset + lane] = geometry;
        const double r2 = geometry.squared_distance;
        if (r2 >= cutoff2 || r2 == 0.0) {
          screening[edge] = 0.0;
          if (r2 == 0.0) {
            atomicMax(overlap, 1);
          }
        } else {
          selected = owns_bond(geometry.displacement);
        }
      }
      const unsigned int selected_mask = __ballot_sync(0xffffffffu, selected);
      if (selected) {
        active_edges[work + __popc(selected_mask & ((1u << lane) - 1u))] = offset + lane;
      }
      work += __popc(selected_mask);
    }
    __syncwarp();
  }
  for (int q = lane; q < work; q += threads_per_atom) {
    const size_t edge = start + (cached ? active_edges[q] : q);
    const auto bond = get_geometry(box, positions, neighbors, i, type, cache, edge, cached);
    const double r2 = bond.squared_distance;
    if (!cached) {
      if (r2 >= cutoff2 || r2 == 0.0) {
        screening[edge] = 0.0;
        if (r2 == 0.0) {
          atomicMax(overlap, 1);
        }
        continue;
      }
      if (!owns_bond(bond.displacement)) {
        continue;
      }
    }
    const bool use_certificate = cached && certificates;
    unsigned long long candidates = use_certificate ? certificates[edge] : 0;
    const bool blocked = use_certificate && (candidates & (1ull << (edge - start)));
    if (use_certificate && (blocked || candidates == 0)) {
      const double value = blocked ? 0.0 : 1.0;
      partial[edge] = 0;
      screening[edge] = value;
      screening[neighbors.reverse[edge]] = value;
      if (!blocked) {
        const int index = atomicAdd(active_count, 1);
        screened_atoms[index] = i;
        screened_edges[index] = edge;
      }
      continue;
    }
    const int pair = type[i] * n + bond.type;
    const double bound = model.bounds[pair] * r2;
    const double inverse_r2 = reciprocal(r2);
    const float r2f = bond.squared_distance32;
    const float bound32 = static_cast<float>(bound);
    const float guard = 0.0002f * (r2f + bound32);
    const float r4 = r2f * r2f;
    const bool safe_scale = r4 >= 1e-28f && r4 <= 1e28f && bound32 <= 64.0f * r2f;
    const ScreeningTest test{bond.displacement32, r2f, bound32, r4, guard, safe_scale};
    // Recheck the last atom that completely screened this bond. An old
    // result is never used without evaluating the current geometry.
    const unsigned int witness = previous_blocker[edge];
    if (witness != 0) {
      const size_t k = start + witness - 1;
      const auto third = get_geometry(box, positions, neighbors, i, type, cache, k, cached);
      const auto limits = model.screens[pair * n + third.type];
      const int saturated = saturated_screening(test, third, limits);
      if (
        saturated == 0 ||
        (saturated == -1 &&
         screening_term(bond, third, inverse_r2, bound, limits, nullptr, nullptr).value == 0.0)) {
        screening[edge] = 0.0;
        screening[neighbors.reverse[edge]] = 0.0;
        continue;
      }
    }
    previous_blocker[edge] = 0;
    double product = 1.0;
    unsigned long long active = 0;
    // Far-list atoms cannot screen this bond if the screening ellipse lies
    // inside the guaranteed radius of the near list.
    const size_t screen_end =
      bound < neighbors.guaranteed_far_distance_squared ? start + neighbors.near_counts[i] : end;
    size_t next = start;
    while (use_certificate ? candidates != 0 : next < screen_end) {
      size_t k;
      if (use_certificate) {
        k = start + __ffsll(candidates) - 1;
        candidates &= candidates - 1;
      } else {
        k = next++;
      }
      if (k != edge) {
        const auto third = get_geometry(box, positions, neighbors, i, type, cache, k, cached);
        const auto limits = model.screens[pair * n + third.type];
        const int saturated = saturated_screening(test, third, limits);
        if (saturated == 1) {
          continue;
        }
        if (saturated == 0) {
          if (k - start < 255) {
            previous_blocker[edge] = k - start + 1;
          }
          product = 0.0;
          break;
        }
        const auto contribution =
          screening_term(bond, third, inverse_r2, bound, limits, nullptr, nullptr);
        product *= contribution.value;
        if (k - start < 64 && contribution.slope != 0.0) {
          active |= 1ull << (k - start);
        }
        if (product == 0.0) {
          if (contribution.value == 0.0 && k - start < 255) {
            previous_blocker[edge] = k - start + 1;
          }
          break;
        }
      }
    }
    partial[edge] = active;
    screening[edge] = product;
    screening[neighbors.reverse[edge]] = product;
    if (product != 0.0) {
      const int index = atomicAdd(active_count, 1);
      screened_atoms[index] = i;
      screened_edges[index] = edge;
    }
  }
}

__device__ void radial_density(const Model& model, int species, double r, double (&density)[4])
{
  if (model.tabulate_density[species]) {
    const double coordinate = r * model.settings.inverse_table_step;
    const int knot = static_cast<int>(coordinate);
    const double fraction = coordinate - knot;
    const double* base = model.density_table + (species * MEAM_TABLE_POINTS + knot) * 4;
    const double* exponent = model.radial_exponents + 4 * species;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const double x = exponent[k] * fraction;
      const float z = static_cast<float>(x);
      const double tail = (1.0f / 24.0f) + z * ((1.0f / 120.0f) + z / 720.0f);
      const double x2 = x * x;
      // The largest remainder from the rounded fourth-order tail is below 5e-14.
      const double series = fma(x2 * x2, tail, fma(x2, 0.5 + x * (1.0 / 6.0), 1.0 + x));
      density[k] = base[k] * series;
    }
    return;
  }
  const auto& element = model.element[species];
  const double stretch = r * model.inverse_re[species] - 1.0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    density[k] = element.density_scale * exp(-element.beta[k] * stretch);
    if (model.settings.alloy_rule == 1 && k > 0) {
      density[k] *= element.t[k - 1];
    }
  }
}

template <bool FixedWeights>
__device__ void accumulate_density(
  Density& density, const MeamElement& element, Vec u, const double* radial, double screen)
{
  const double a[3] = {u.x, u.y, u.z};
  const double q0 = screen * radial[0];
  const double q1 = screen * radial[1];
  const double q2 = screen * radial[2];
  const double q3 = screen * radial[3];
  density.radial += q0;
  density.quadrupole_trace += q2;
  int ij = 0;
  int ijk = 0;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    if (!FixedWeights) {
      density.weight[i] += q0 * element.t[i];
      density.weight2[i] += q0 * element.t[i] * element.t[i];
    }
    density.dipole[i] += q1 * a[i];
    density.octupole_trace[i] += q3 * a[i];
#pragma unroll
    for (int j = i; j < 3; ++j) {
      const double second = a[i] * a[j];
      density.quadrupole[ij++] += q2 * second;
#pragma unroll
      for (int k = j; k < 3; ++k) {
        density.octupole[ijk++] += q3 * second * a[k];
      }
    }
  }
}

__device__ int multiplicity(int i, int j, int k)
{
  if (i == k) {
    return 1;
  }
  return i == j || j == k ? 3 : 6;
}

// Transform moment storage in place into the derivatives of the site embedding energy.
__device__ double
embedding_gradient(Density& moments, const MeamElement& element, const MeamSettings& settings)
{
  const double rho = moments.radial;
  if (rho <= 0.0) {
    moments = {};
    return 0.0;
  }
  double invariant[3]{0.0, -moments.quadrupole_trace * moments.quadrupole_trace / 3.0, 0.0};
  int ij = 0;
  int ijk = 0;
  for (int i = 0; i < 3; ++i) {
    invariant[0] += moments.dipole[i] * moments.dipole[i];
    invariant[2] -= 0.6 * moments.octupole_trace[i] * moments.octupole_trace[i];
    for (int j = i; j < 3; ++j) {
      invariant[1] += (i == j ? 1.0 : 2.0) * moments.quadrupole[ij] * moments.quadrupole[ij];
      ++ij;
      for (int k = j; k < 3; ++k) {
        invariant[2] += multiplicity(i, j, k) * moments.octupole[ijk] * moments.octupole[ijk];
        ++ijk;
      }
    }
  }
  double t[3];
  double gamma = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (settings.alloy_rule == 2) {
      t[i] = element.t[i];
    } else {
      const double denominator = settings.alloy_rule == 1 ? moments.weight2[i] : rho;
      t[i] = denominator > 0.0 ? moments.weight[i] / denominator : 0.0;
    }
    gamma += t[i] * invariant[i];
  }
  gamma /= rho * rho;
  const auto angular =
    meam_angular_scaling(gamma, element.density_function, settings.gamma_smoothing);
  double background = element.reference_density;
  double background_derivative = 0.0;
  if (settings.mixed_reference || settings.dynamic_background) {
    background = element.density_scale * element.coordination;
    if (settings.mixed_reference && element.density_function > 0) {
      const double z2 = element.coordination * element.coordination;
      const auto reference = meam_angular_scaling(
        (t[0] * element.reference_shape[0] + t[1] * element.reference_shape[1] +
         t[2] * element.reference_shape[2]) /
          z2,
        element.density_function,
        settings.gamma_smoothing);
      background *= reference.value;
      background_derivative = reference.slope / (reference.value * z2);
    }
  }

  const double normalized = rho * angular.value / background;
  const auto embedding =
    meam_embedding(normalized, element.embedding_scale, settings.linear_negative_embedding);
  const double scale = embedding.slope / background;
  const double dgamma = scale * angular.slope / rho;
  moments.radial = scale * (angular.value - 2.0 * angular.slope * gamma);
  for (int i = 0; i < 3; ++i) {
    const double dw = dgamma * invariant[i] - embedding.slope * normalized * background_derivative *
                                                element.reference_shape[i];
    const double weight2 = moments.weight2[i];
    moments.weight[i] = 0.0;
    moments.weight2[i] = 0.0;
    if (settings.alloy_rule == 0) {
      moments.weight[i] = dw / rho;
      moments.radial -= dw * t[i] / rho;
    } else if (settings.alloy_rule == 1 && weight2 > 0.0) {
      moments.weight[i] = dw / weight2;
      moments.weight2[i] = -dw * t[i] / weight2;
    }
  }
  const double d1 = 2.0 * dgamma * t[0];
  const double d2 = 2.0 * dgamma * t[1];
  const double d3 = 2.0 * dgamma * t[2];
  moments.quadrupole_trace *= -d2 / 3.0;
  ij = 0;
  ijk = 0;
  for (int i = 0; i < 3; ++i) {
    moments.dipole[i] *= d1;
    moments.octupole_trace[i] *= -0.6 * d3;
    for (int j = i; j < 3; ++j) {
      moments.quadrupole[ij++] *= d2 * (i == j ? 1.0 : 2.0);
      for (int k = j; k < 3; ++k) {
        moments.octupole[ijk++] *= d3 * multiplicity(i, j, k);
      }
    }
  }
  return embedding.value;
}

template <bool FixedWeights>
__global__ void site_energies(
  Box box,
  Positions positions,
  Neighbors neighbors,
  const int* type,
  Model model,
  const double* screens,
  Density* gradients,
  double* energy)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= positions.size) {
    return;
  }
  const auto center = positions.at(i);
  const double cutoff = model.settings.cutoff;
  const bool interior = box.is_orthogonal &&
                        (!box.pbc_x || (center.x > cutoff && center.x < box.cpu_h[0] - cutoff)) &&
                        (!box.pbc_y || (center.y > cutoff && center.y < box.cpu_h[4] - cutoff)) &&
                        (!box.pbc_z || (center.z > cutoff && center.z < box.cpu_h[8] - cutoff));
  Density density{};
  double pair_energy = 0.0;
  for (size_t edge = neighbors.offset[i]; edge < neighbors.offset[i + 1]; ++edge) {
    const double geometry = screens[edge];
    if (geometry == 0.0) {
      continue;
    }
    const int j = neighbors.atom[edge];
    Vec v = positions.at(j) - center;
    double r2 = v.square();
    if (!interior || r2 > cutoff * cutoff) {
      apply_mic(box, v.x, v.y, v.z);
      r2 = v.square();
    }
    const double inverse_r = rsqrt(r2);
    const double r = r2 * inverse_r;
    const double screen =
      geometry * meam_cutoff((model.settings.cutoff - r) * model.inverse_smoothing_width).value;
    double radial[4];
    const auto& elem = model.element[type[j]];
    radial_density(model, type[j], r, radial);
    accumulate_density<FixedWeights>(density, elem, v * inverse_r, radial, screen);
    const int pair = type[i] * model.settings.type_count + type[j];
    pair_energy += 0.5 * screen * model.pair(pair, r).value;
  }
  energy[i] += pair_energy + embedding_gradient(density, model.element[type[i]], model.settings);
  gradients[i] = density;
}

template <bool FixedWeights>
__device__ BondGradient site_bond_gradient(
  const Density& gradient,
  const Model& model,
  int species,
  Vec u,
  double inverse_r,
  double r,
  double screen,
  MeamValueSlope pair)
{
  double radial[4];
  radial_density(model, species, r, radial);
  const auto& element = model.element[species];
  const double x = u.x;
  const double y = u.y;
  const double z = u.z;
  const double xx = x * x;
  const double xy = x * y;
  const double xz = x * z;
  const double yy = y * y;
  const double yz = y * z;
  const double zz = z * z;
  const auto& q = gradient.quadrupole;
  const auto& t = gradient.octupole;
  const Vec q_grad{
    2.0 * q[0] * x + q[1] * y + q[2] * z,
    q[1] * x + 2.0 * q[3] * y + q[4] * z,
    q[2] * x + q[4] * y + 2.0 * q[5] * z};
  const Vec t_grad{
    3.0 * t[0] * xx + 2.0 * (t[1] * xy + t[2] * xz) + t[3] * yy + t[4] * yz + t[5] * zz,
    t[1] * xx + 2.0 * (t[3] * xy + t[7] * yz) + t[4] * xz + 3.0 * t[6] * yy + t[8] * zz,
    t[2] * xx + t[4] * xy + 2.0 * (t[5] * xz + t[8] * yz) + t[7] * yy + 3.0 * t[9] * zz};
  const Vec dipole{gradient.dipole[0], gradient.dipole[1], gradient.dipole[2]};
  const Vec octupole_trace{
    gradient.octupole_trace[0], gradient.octupole_trace[1], gradient.octupole_trace[2]};
  // Euler's identity expresses the quadratic and cubic contractions using their gradients.
  const double angular[3]{
    dipole.dot(u),
    0.5 * q_grad.dot(u) + gradient.quadrupole_trace,
    (1.0 / 3.0) * t_grad.dot(u) + octupole_trace.dot(u)};
  double scalar = gradient.radial;
  if (!FixedWeights) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      scalar += element.t[i] * (gradient.weight[i] + element.t[i] * gradient.weight2[i]);
    }
  }
  double dscreen = 0.5 * pair.value + scalar * radial[0];
  double radial_slope = scalar * element.beta[0] * radial[0];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    dscreen += angular[i] * radial[i + 1];
    radial_slope += angular[i] * element.beta[i + 1] * radial[i + 1];
  }
  const double distance = 0.5 * pair.slope - radial_slope * model.inverse_re[species];
  const Vec direction =
    dipole * radial[1] + q_grad * radial[2] + (t_grad + octupole_trace) * radial[3];
  return {(u * distance + (direction - u * direction.dot(u)) * inverse_r) * screen, dscreen};
}

__device__ void add_force(int atom, Vec f, int n, double* force)
{
  atomicAdd(force + atom, f.x);
  atomicAdd(force + atom + n, f.y);
  atomicAdd(force + atom + 2 * n, f.z);
}

__device__ void add_virial(int atom, Vec r, Vec g, int n, double* virial)
{
  atomicAdd(virial + atom, -r.x * g.x);
  atomicAdd(virial + atom + n, -r.y * g.y);
  atomicAdd(virial + atom + 2 * n, -r.z * g.z);
  atomicAdd(virial + atom + 3 * n, -r.x * g.y);
  atomicAdd(virial + atom + 4 * n, -r.x * g.z);
  atomicAdd(virial + atom + 5 * n, -r.y * g.z);
  atomicAdd(virial + atom + 6 * n, -r.y * g.x);
  atomicAdd(virial + atom + 7 * n, -r.z * g.x);
  atomicAdd(virial + atom + 8 * n, -r.z * g.y);
}

template <bool FixedWeights>
__global__ void forces_base(
  Box box,
  Positions positions,
  Neighbors neighbors,
  const int* type,
  Model model,
  const double* screens,
  const Density* gradients,
  int active_count,
  const int* active_atoms,
  const int* active_edges,
  double* bond_ci,
  double* bond_cj,
  int* k_bonds,
  int* k_count,
  double* force,
  double* virial)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= active_count) {
    return;
  }
  const int i = active_atoms[index];
  const size_t edge = active_edges[index];
  const int j = neighbors.atom[edge];
  const int atoms = positions.size;
  const int n = model.settings.type_count;
  const Vec v = positions.displacement(box, i, j);
  const double r2 = v.square();
  const double inverse_r = rsqrt(r2);
  const double r = r2 * inverse_r;
  const Vec direction = v * inverse_r;
  const double geometric_screen = screens[edge];
  const auto cutoff = meam_cutoff((model.settings.cutoff - r) * model.inverse_smoothing_width);
  const double total_screen = geometric_screen * cutoff.value;
  const int p = type[i] * n + type[j];
  const auto pair = model.pair(p, r);
  const auto di = site_bond_gradient<FixedWeights>(
    gradients[i], model, type[j], direction, inverse_r, r, total_screen, pair);
  const auto dj = site_bond_gradient<FixedWeights>(
    gradients[j], model, type[i], direction * -1.0, inverse_r, r, total_screen, pair);
  const double radial_screen = -geometric_screen * cutoff.slope * model.inverse_smoothing_width;
  const Vec on_j = di.displacement + direction * (di.screening * radial_screen);
  const Vec on_i = dj.displacement - direction * (dj.screening * radial_screen);
  bond_ci[index] = di.screening;
  bond_cj[index] = dj.screening;
  if (geometric_screen < 1.0) {
    const int slot = atomicAdd(k_count, 1);
    k_bonds[slot] = index;
  }
  add_force(i, on_j - on_i, atoms, force);
  add_force(j, on_i - on_j, atoms, force);
  add_virial(j, v, on_j, atoms, virial);
  add_virial(i, v * -1.0, on_i, atoms, virial);
}

__global__ void forces_screen(
  Box box,
  Positions positions,
  Neighbors neighbors,
  const int* type,
  Model model,
  const double* screens,
  const unsigned long long* partial,
  const int* active_atoms,
  const int* active_edges,
  const double* bond_ci,
  const double* bond_cj,
  int k_count,
  const int* k_bonds,
  double* force,
  double* virial)
{
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= k_count) {
    return;
  }
  const int index = k_bonds[slot];
  const int i = active_atoms[index];
  const size_t edge = active_edges[index];
  const int atoms = positions.size;
  const int n = model.settings.type_count;
  const int j = neighbors.atom[edge];
  const Vec center = positions.at(i);
  const auto bond = relative_geometry(box, positions, center, j, type);
  const double r2 = bond.squared_distance;
  const double inverse_r = rsqrt(r2);
  const double r = r2 * inverse_r;
  const double geometric_screen = screens[edge];
  const auto cutoff = meam_cutoff((model.settings.cutoff - r) * model.inverse_smoothing_width);
  const double total_screen = geometric_screen * cutoff.value;
  const int p = type[i] * n + type[j];
  const double bound = model.bounds[p] * r2;
  const double inverse_r2 = reciprocal(r2);
  const double ci = bond_ci[index];
  const double cj = bond_cj[index];
  const Vec v = bond.displacement;
  const size_t start = neighbors.offset[i];
  const size_t end = neighbors.offset[i + 1];
  const bool use_mask = end - start <= 64;
  unsigned long long pending = partial[edge];
  size_t next = start;
  while (use_mask ? pending != 0 : next < end) {
    size_t k;
    if (use_mask) {
      k = start + __ffsll(pending) - 1;
      pending &= pending - 1;
    } else {
      k = next++;
    }
    if (k == edge) {
      continue;
    }
    const auto third = relative_geometry(box, positions, center, neighbors.atom[k], type);
    Vec dv{}, dw{};
    const auto s =
      screening_term(bond, third, inverse_r2, bound, model.screens[p * n + third.type], &dv, &dw);
    if (s.slope == 0.0) {
      continue;
    }
    const double ds = total_screen * reciprocal(s.value) * s.slope;
    const double si = ci * ds;
    const double sj = cj * ds;
    const double total = si + sj;
    add_force(i, (dv + dw) * total, atoms, force);
    add_force(j, dv * -total, atoms, force);
    add_force(third.atom, dw * -total, atoms, force);
    add_virial(i, v, (dv + dw) * sj, atoms, virial);
    add_virial(j, v, dv * si, atoms, virial);
    add_virial(third.atom, third.displacement * total - v * sj, dw, atoms, virial);
  }
}
} // namespace

struct MEAM::Data {
  MeamSettings settings;
  double candidate_radius;
  GPU_Vector<MeamElement> elements;
  GPU_Vector<ScreeningLimits> screens;
  GPU_Vector<double> bounds;
  GPU_Vector<MeamSpline> tables;
  GPU_Vector<double> density_table, radial_exponents;
  GPU_Vector<int> cell_count, cell_offset, cell_atoms;
  GPU_Vector<int> neighbors, overlap, rebuild, near_counts;
  GPU_Vector<size_t> counts, offsets, reverse;
  GPU_Vector<int> active_atoms, active_edges, k_bonds, active_count, k_count;
  GPU_Vector<double> bond_ci, bond_cj;
  size_t list_capacity = 0;
  GPU_Vector<double> screening, reference_position, inverse_re;
  GPU_Vector<int> tabulate_density;
  GPU_Vector<unsigned long long> partial_screeners, certificates;
  GPU_Vector<double> screen_reference_position;
  GPU_Vector<int> screen_reference_species;
  Box screen_reference_box;
  bool screen_certificates_ready = false;
  size_t neighbor_entries = 0;
  GPU_Vector<unsigned char> previous_blocker;
  Box reference_box;
  bool list_ready = false;
  GPU_Vector<Density> gradients;
};

MEAM::MEAM(const char* filename, int number_of_types) : data_(std::make_unique<Data>())
{
  const auto input = read_meam_parameters(filename, number_of_types);
  auto& d = *data_;
  d.settings = input.settings;
  rc = d.settings.cutoff;
  d.candidate_radius =
    rc * sqrt(*std::max_element(input.screening_bounds.begin(), input.screening_bounds.end()));
  d.elements.resize(input.elements.size());
  d.elements.copy_from_host(input.elements.data());
  std::vector<ScreeningLimits> screen_table;
  screen_table.reserve(input.screening.size());
  for (const auto& screen : input.screening) {
    screen_table.push_back(
      {screen,
       static_cast<float>(screen.minimum - 1.0),
       static_cast<float>(screen.minimum + 1.0 / screen.inverse_width - 1.0)});
  }
  d.screens.resize(screen_table.size());
  d.screens.copy_from_host(screen_table.data());
  d.bounds.resize(input.screening_bounds.size());
  d.bounds.copy_from_host(input.screening_bounds.data());
  d.tables.resize(input.pair_tables.size());
  d.tables.copy_from_host(input.pair_tables.data());
  const auto in_table_range = [](double value) {
    return std::isfinite(value) && fabs(value) >= 2.0 * std::numeric_limits<double>::min() &&
           fabs(value) <= 0.5 * std::numeric_limits<double>::max();
  };
  std::vector<double> radial(input.elements.size() * MEAM_TABLE_POINTS * 4);
  std::vector<int> tabulate_density(input.elements.size(), 1);
  for (size_t species = 0; species < input.elements.size(); ++species) {
    const auto& e = input.elements[species];
    for (int knot = 0; knot < MEAM_TABLE_POINTS; ++knot) {
      const double stretch = (knot / d.settings.inverse_table_step) / e.equilibrium_distance - 1.0;
      for (int rank = 0; rank < 4; ++rank) {
        const double exponential = exp(-e.beta[rank] * stretch);
        double value = e.density_scale * exponential;
        if (!in_table_range(exponential) || !in_table_range(value)) {
          tabulate_density[species] = 0;
        }
        if (d.settings.alloy_rule == 1 && rank > 0) {
          value *= e.t[rank - 1];
        }
        if (
          !in_table_range(value) &&
          !(d.settings.alloy_rule == 1 && rank > 0 && e.t[rank - 1] == 0.0)) {
          tabulate_density[species] = 0;
        }
        radial[(species * MEAM_TABLE_POINTS + knot) * 4 + rank] = value;
      }
    }
  }
  d.density_table.resize(radial.size());
  d.density_table.copy_from_host(radial.data());
  d.tabulate_density.resize(tabulate_density.size());
  std::vector<double> inverse_re;
  inverse_re.reserve(input.elements.size());
  for (const auto& element : input.elements) {
    inverse_re.push_back(1.0 / element.equilibrium_distance);
  }
  d.inverse_re.resize(inverse_re.size());
  d.inverse_re.copy_from_host(inverse_re.data());
  std::vector<double> exponents(4 * input.elements.size());
  for (size_t species = 0; species < input.elements.size(); ++species) {
    for (int k = 0; k < 4; ++k) {
      const double x =
        -input.elements[species].beta[k] * inverse_re[species] / d.settings.inverse_table_step;
      exponents[4 * species + k] = x;
      tabulate_density[species] &= fabs(x) <= 0.04;
    }
  }
  d.radial_exponents.resize(exponents.size());
  d.radial_exponents.copy_from_host(exponents.data());
  d.tabulate_density.copy_from_host(tabulate_density.data());
  d.overlap.resize(1);
  d.rebuild.resize(1);
  d.active_count.resize(1);
  d.k_count.resize(1);
  printf("Use MEAM potential with %d element(s) and cutoff %.6g Angstrom.\n", number_of_types, rc);
}

MEAM::~MEAM() = default;

void MEAM::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& energy,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  auto& d = *data_;
  const int n = type.size();
  if (n == 0) {
    return;
  }
  int bins[3];
  const double candidate_radius = d.candidate_radius;
  const double list_radius = candidate_radius + neighbor_skin;
  const double near_radius = 0.75 * rc;
  const double near2 = near_radius * near_radius;
  // Motion and box deformation can shrink any reference separation by at
  // most the skin within the range used for screening.
  const double far_bound = std::max(0.0, near_radius - neighbor_skin);
  box.get_num_bins(list_radius, bins);
  for (int d = 0; d < 3; ++d) {
    bins[d] = std::max(1, bins[d]);
  }
  if (
    (box.pbc_x && box.thickness_x <= 2.0 * candidate_radius) ||
    (box.pbc_y && box.thickness_y <= 2.0 * candidate_radius) ||
    (box.pbc_z && box.thickness_z <= 2.0 * candidate_radius)) {
    PRINT_INPUT_ERROR(
      "MEAM requires periodic box thickness greater than twice its screening radius.");
  }
  if (d.cell_atoms.size() != static_cast<size_t>(n)) {
    d.cell_atoms.resize(n);
    d.gradients.resize(n);
    d.counts.resize(n + 1);
    d.near_counts.resize(n);
    d.offsets.resize(n + 1);
    d.reference_position.resize(position.size());
    d.list_ready = false;
  }
  const Positions positions{position.data(), n};
  const int blocks = (n - 1) / block_size + 1;
  int rebuild = 1;
  if (
    d.list_ready && box.pbc_x == d.reference_box.pbc_x && box.pbc_y == d.reference_box.pbc_y &&
    box.pbc_z == d.reference_box.pbc_z) {
    MotionCheck check{};
    const double contraction = neighbor_motion_limit(box, d.reference_box, check);
    const double budget = 0.5 * (neighbor_skin - list_radius * contraction);
    if (budget > 0.0) {
      check.tolerance_squared = budget * budget;
      d.rebuild.fill(0);
      check_neighbor_motion<<<blocks, block_size>>>(
        box, positions, d.reference_position.data(), check, d.rebuild.data());
      GPU_CHECK_KERNEL
      d.rebuild.copy_to_host(&rebuild);
    }
  }
  d.overlap.fill(0);
  if (rebuild) {
    d.screen_certificates_ready = false;
    find_cell_list(list_radius, bins, box, position, d.cell_count, d.cell_offset, d.cell_atoms);
    const CellList cells{
      bins[0], bins[1], bins[2], d.cell_count.data(), d.cell_offset.data(), d.cell_atoms.data()};
    CHECK(gpuMemset(d.counts.data() + n, 0, sizeof(size_t)));
    make_neighbors<<<blocks, block_size>>>(
      box,
      positions,
      cells,
      list_radius,
      near2,
      d.near_counts.data(),
      d.counts.data(),
      nullptr,
      nullptr,
      d.overlap.data());
    GPU_CHECK_KERNEL
    thrust::exclusive_scan(
      thrust::device, d.counts.data(), d.counts.data() + n + 1, d.offsets.data());
    size_t needed;
    d.offsets.copy_to_host(&needed, 1, n);
    d.neighbor_entries = needed;
    if (needed > d.neighbors.size()) {
      const size_t capacity = needed + needed / 8;
      d.neighbors.resize(capacity);
      d.reverse.resize(capacity);
      d.screening.resize(capacity);
      d.partial_screeners.resize(capacity);
      d.previous_blocker.resize(capacity);
    }
    const size_t list_capacity = needed / 2 + 1024;
    if (list_capacity > d.list_capacity) {
      d.list_capacity = list_capacity;
      d.active_atoms.resize(list_capacity);
      d.active_edges.resize(list_capacity);
      d.k_bonds.resize(list_capacity);
      d.bond_ci.resize(list_capacity);
      d.bond_cj.resize(list_capacity);
    }
    if (needed) {
      // A new list can change the local indices of all candidate screeners.
      CHECK(gpuMemset(d.previous_blocker.data(), 0, needed * sizeof(unsigned char)));
      make_neighbors<<<blocks, block_size>>>(
        box,
        positions,
        cells,
        list_radius,
        near2,
        d.near_counts.data(),
        nullptr,
        d.offsets.data(),
        d.neighbors.data(),
        nullptr);
      GPU_CHECK_KERNEL
      reverse_neighbors<<<blocks, block_size>>>(
        n, d.offsets.data(), d.neighbors.data(), d.reverse.data(), d.overlap.data());
      GPU_CHECK_KERNEL
    }
    CHECK(gpuMemcpy(
      d.reference_position.data(),
      position.data(),
      position.size() * sizeof(double),
      gpuMemcpyDeviceToDevice));
    d.reference_box = box;
    d.list_ready = true;
  }
  const Neighbors neighbors{
    d.offsets.data(),
    d.neighbors.data(),
    d.reverse.data(),
    d.near_counts.data(),
    far_bound * far_bound};
  const Model model{
    d.settings,
    d.elements.data(),
    d.screens.data(),
    d.bounds.data(),
    d.tables.data(),
    d.density_table.data(),
    d.radial_exponents.data(),
    d.inverse_re.data(),
    d.tabulate_density.data(),
    1.0 / d.settings.smoothing_width};
  // A single image must remain valid for each short triangle for the entire interval.
  const double certified_box_width = 4.0 * (list_radius + screen_certificate_skin);
  const bool large_orthogonal_box = box.is_orthogonal &&
                                    (!box.pbc_x || box.thickness_x > certified_box_width) &&
                                    (!box.pbc_y || box.thickness_y > certified_box_width) &&
                                    (!box.pbc_z || box.thickness_z > certified_box_width);
  // Leave the ordinary MEAM work buffers enough headroom for resizes and
  // measurements. The optional interval cache can be bypassed under memory pressure.
  if (
    large_orthogonal_box && rebuild && d.neighbor_entries > 0 &&
    (d.certificates.size() < d.neighbor_entries ||
     d.screen_reference_position.size() != position.size())) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t bytes = d.neighbors.size() * sizeof(unsigned long long) +
                         position.size() * sizeof(double) + type.size() * sizeof(int);
    if (bytes <= free_bytes / 4) {
      d.certificates.resize(d.neighbors.size());
      d.screen_reference_position.resize(position.size());
      d.screen_reference_species.resize(type.size());
    }
  }
  const bool certify = large_orthogonal_box && d.neighbor_entries > 0 &&
                       d.certificates.size() >= d.neighbor_entries &&
                       d.screen_reference_position.size() == position.size();
  if (certify) {
    int refresh = 1;
    if (d.screen_certificates_ready) {
      MotionCheck check{};
      const double distortion = neighbor_motion_limit(box, d.screen_reference_box, check);
      const double budget = 0.5 * (screen_certificate_skin -
                                   2.0 * (list_radius + screen_certificate_skin) * distortion);
      if (budget > 0.0) {
        d.rebuild.fill(0);
        check.tolerance_squared = budget * budget;
        check_neighbor_motion<true><<<blocks, block_size>>>(
          box,
          positions,
          d.screen_reference_position.data(),
          check,
          d.rebuild.data(),
          type.data(),
          d.screen_reference_species.data());
        GPU_CHECK_KERNEL
        d.rebuild.copy_to_host(&refresh);
      }
    }
    if (refresh) {
      build_screen_certificates<<<
        (n - 1) / screening_atoms_per_block + 1,
        screening_atoms_per_block * screening_threads_per_atom>>>(
        box, positions, neighbors, type.data(), model, d.certificates.data());
      GPU_CHECK_KERNEL
      CHECK(gpuMemcpy(
        d.screen_reference_position.data(),
        position.data(),
        position.size() * sizeof(double),
        gpuMemcpyDeviceToDevice));
      CHECK(gpuMemcpy(
        d.screen_reference_species.data(),
        type.data(),
        type.size() * sizeof(int),
        gpuMemcpyDeviceToDevice));
      d.screen_reference_box = box;
      d.screen_certificates_ready = true;
    }
  } else {
    d.screen_certificates_ready = false;
  }
  CHECK(gpuMemset(d.active_count.data(), 0, sizeof(int)));
  screen_bonds<<<
    (n - 1) / screening_atoms_per_block + 1,
    screening_atoms_per_block * screening_threads_per_atom>>>(
    box,
    positions,
    neighbors,
    type.data(),
    model,
    d.screening.data(),
    d.partial_screeners.data(),
    d.previous_blocker.data(),
    certify ? d.certificates.data() : nullptr,
    d.overlap.data(),
    d.active_atoms.data(),
    d.active_edges.data(),
    d.active_count.data());
  GPU_CHECK_KERNEL
  int active_count = 0;
  d.active_count.copy_to_host(&active_count);
  if (active_count > static_cast<int>(d.list_capacity)) {
    PRINT_INPUT_ERROR("MEAM active-bond list overflow.");
  }
  CHECK(gpuMemset(d.k_count.data(), 0, sizeof(int)));
  if (d.settings.alloy_rule == 2) {
    site_energies<true><<<blocks, block_size>>>(
      box,
      positions,
      neighbors,
      type.data(),
      model,
      d.screening.data(),
      d.gradients.data(),
      energy.data());
    GPU_CHECK_KERNEL
    forces_base<true><<<(active_count - 1) / 128 + 1, 128>>>(
      box,
      positions,
      neighbors,
      type.data(),
      model,
      d.screening.data(),
      d.gradients.data(),
      active_count,
      d.active_atoms.data(),
      d.active_edges.data(),
      d.bond_ci.data(),
      d.bond_cj.data(),
      d.k_bonds.data(),
      d.k_count.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
    int k_count = 0;
    d.k_count.copy_to_host(&k_count);
    forces_screen<<<(k_count - 1) / 128 + 1, 128>>>(
      box,
      positions,
      neighbors,
      type.data(),
      model,
      d.screening.data(),
      d.partial_screeners.data(),
      d.active_atoms.data(),
      d.active_edges.data(),
      d.bond_ci.data(),
      d.bond_cj.data(),
      k_count,
      d.k_bonds.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
  } else {
    site_energies<false><<<blocks, block_size>>>(
      box,
      positions,
      neighbors,
      type.data(),
      model,
      d.screening.data(),
      d.gradients.data(),
      energy.data());
    GPU_CHECK_KERNEL
    forces_base<false><<<(active_count - 1) / 128 + 1, 128>>>(
      box,
      positions,
      neighbors,
      type.data(),
      model,
      d.screening.data(),
      d.gradients.data(),
      active_count,
      d.active_atoms.data(),
      d.active_edges.data(),
      d.bond_ci.data(),
      d.bond_cj.data(),
      d.k_bonds.data(),
      d.k_count.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
    int k_count = 0;
    d.k_count.copy_to_host(&k_count);
    forces_screen<<<(k_count - 1) / 128 + 1, 128>>>(
      box,
      positions,
      neighbors,
      type.data(),
      model,
      d.screening.data(),
      d.partial_screeners.data(),
      d.active_atoms.data(),
      d.active_edges.data(),
      d.bond_ci.data(),
      d.bond_cj.data(),
      k_count,
      d.k_bonds.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
  }
  int overlaps;
  d.overlap.copy_to_host(&overlaps);
  if (overlaps == 2) {
    PRINT_INPUT_ERROR("MEAM neighbor list is missing a reverse bond.");
  }
  if (overlaps) {
    PRINT_INPUT_ERROR("MEAM encountered overlapping atoms.");
  }
}
