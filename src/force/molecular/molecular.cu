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

/*----------------------------------------------------------------------------80
Lennard-Jones, harmonic bond, and harmonic angle forces for fixed molecular topologies.

Major contributors:
  Zheyong Fan: Initial implementation.
------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "input.cuh"
#include "model/atom.cuh"
#include "molecular.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cfloat>
#include <cmath>

template <typename T>
static void copy_to_gpu(const std::vector<T>& source, GPU_Vector<T>& target)
{
  target.resize(source.size());
  target.copy_from_host(source.data());
}

Molecular::Molecular(const char* forcefield, const char* topology, const Atom& atom)
{
  const MolecularInput input =
    read_molecular_input(forcefield, topology, atom.number_of_atoms, atom.cpu_atom_symbol);
  for (const float charge : atom.cpu_charge) {
    if (charge != 0.0f) {
      PRINT_INPUT_ERROR("The molecular LJ potential requires zero charges in model.xyz.");
    }
  }
  rc = input.cutoff_;
  cell_contents_.resize(atom.number_of_atoms);
  num_types_ = input.num_types_;
  copy_to_gpu(input.types_, types_);
  copy_to_gpu(input.c6_, c6_);
  copy_to_gpu(input.c12_, c12_);
  if (!input.partners_.empty()) {
    copy_to_gpu(input.offsets_, offsets_);
    copy_to_gpu(input.partners_, partners_);
    copy_to_gpu(input.lj_weights_, lj_weights_);
  }
  if (!input.bond_partners_.empty()) {
    copy_to_gpu(input.bond_offsets_, bond_offsets_);
    copy_to_gpu(input.bond_partners_, bond_partners_);
    copy_to_gpu(input.bond_types_, bond_types_);
    copy_to_gpu(input.bond_k_, bond_k_);
    copy_to_gpu(input.bond_r0_, bond_r0_);
  }
  if (!input.angles_.empty()) {
    copy_to_gpu(input.angles_, angles_);
    copy_to_gpu(input.angle_offsets_, angle_offsets_);
    copy_to_gpu(input.angle_indices_, angle_indices_);
    copy_to_gpu(input.angle_k_, angle_k_);
    copy_to_gpu(input.angle_theta0_, angle_theta0_);
    copy_to_gpu(input.angle_pi_minus_theta0_, angle_pi_minus_theta0_);
    angle_error_reported_.resize(1, 0);
  }
}

namespace
{
struct PairSum {
  float energy_, fx_, fy_, fz_;
  float xx_, yy_, zz_, xy_, xz_, yz_;

  __device__ void add_force(const float dx, const float dy, const float dz, const float f)
  {
    fx_ += dx * f;
    fy_ += dy * f;
    fz_ += dz * f;
    // W_i = -r_ij tensor F_ij / 2 for a directed pair.
    const float v = -0.5f * f;
    xx_ += dx * dx * v;
    yy_ += dy * dy * v;
    zz_ += dz * dz * v;
    xy_ += dx * dy * v;
    xz_ += dx * dz * v;
    yz_ += dy * dz * v;
  }

  __device__ void
  write(const int i, const size_t N, double* potential, double* force, double* virial) const
  {
    potential[i] += energy_;
    force[i] += fx_;
    force[i + N] += fy_;
    force[i + 2 * N] += fz_;
    virial[i] += xx_;
    virial[i + N] += yy_;
    virial[i + 2 * N] += zz_;
    virial[i + 3 * N] += xy_;
    virial[i + 4 * N] += xz_;
    virial[i + 5 * N] += yz_;
    virial[i + 6 * N] += xy_;
    virial[i + 7 * N] += xz_;
    virial[i + 8 * N] += yz_;
  }
};

__device__ void add_pair(
  const float c6,
  const float c12,
  const float dx,
  const float dy,
  const float dz,
  const float r2,
  const float weight,
  PairSum& sum)
{
  if (c6 == 0.0f && c12 == 0.0f) {
    return;
  }
  const float inv2 = 1.0f / r2;
  const float inv6 = inv2 * inv2 * inv2;
  const float attraction = c6 * inv6;
  const float repulsion = c12 * inv6 * inv6;
  const float f = weight * 6.0f * (attraction - 2.0f * repulsion) * inv2;
  sum.energy_ += weight * 0.5f * (repulsion - attraction);
  sum.add_force(dx, dy, dz, f);
}

// Check the rounding uncertainty of the fractional-coordinate dot product before choosing an image.
__device__ bool find_image(float a, float b, float c, float& image)
{
  const float s = a + b + c;
  image = nearbyintf(s);
  const float error = 8.0f * FLT_EPSILON * (fabsf(a) + fabsf(b) + fabsf(c));
  return fabsf(s - image) + error < 0.5f;
}

__device__ bool subtract_image(float& d, float a, float b, float c)
{
  // If the subtraction loses more than four bits, reduce the original double coordinates instead.
  const float scale = fmaxf(fmaxf(fabsf(d), fabsf(a)), fmaxf(fabsf(b), fabsf(c)));
  d -= a + b + c;
  return scale <= 16.0f * fabsf(d);
}

__device__ bool wrap_orthogonal(float& d, float length, float inverse_length)
{
  const float s = d * inverse_length;
  // Leave a rounding margin near half a period for the image check below.
  if (fabsf(s) < 0.5f - 4.0f * FLT_EPSILON) {
    return true;
  }
  float image;
  return find_image(s, 0.0f, 0.0f, image) && subtract_image(d, image * length, 0.0f, 0.0f);
}

__device__ bool apply_pair_mic(const Box& box, float& x, float& y, float& z)
{
  const float* h = box.float_h;
  if (box.is_orthogonal) {
    return wrap_orthogonal(x, h[0], h[9]) && wrap_orthogonal(y, h[4], h[13]) &&
           wrap_orthogonal(z, h[8], h[17]);
  }

  float ix, iy, iz;
  if (
    !find_image(x * h[9], y * h[10], z * h[11], ix) ||
    !find_image(x * h[12], y * h[13], z * h[14], iy) ||
    !find_image(x * h[15], y * h[16], z * h[17], iz)) {
    return false;
  }
  if (ix == 0.0f && iy == 0.0f && iz == 0.0f) {
    return true;
  }
  // Subtract the lattice translation directly to retain short vectors in highly skewed cells.
  return subtract_image(x, h[0] * ix, h[1] * iy, h[2] * iz) &&
         subtract_image(y, h[3] * ix, h[4] * iy, h[5] * iz) &&
         subtract_image(z, h[6] * ix, h[7] * iy, h[8] * iz);
}

__device__ void apply_pair_mic(const Box& box, double& dx, double& dy, double& dz)
{
  if (box.is_orthogonal) {
    dx -= box.cpu_h[0] * nearbyint(dx * box.cpu_h[9]);
    dy -= box.cpu_h[4] * nearbyint(dy * box.cpu_h[13]);
    dz -= box.cpu_h[8] * nearbyint(dz * box.cpu_h[17]);
  } else {
    apply_mic(box, dx, dy, dz);
  }
}

__device__ void
get_pair_displacement(const Box& box, double dx, double dy, double dz, float& x, float& y, float& z)
{
  x = dx;
  y = dy;
  z = dz;
  if (!apply_pair_mic(box, x, y, z)) {
    apply_pair_mic(box, dx, dy, dz);
    x = dx;
    y = dy;
    z = dz;
  }
}

__device__ bool
inside_lj_cutoff(const Box& box, double dx, double dy, double dz, const double cutoff2)
{
  apply_pair_mic(box, dx, dy, dz);
  return dx * dx + dy * dy + dz * dz < cutoff2;
}

__device__ int wrap_bin(int index, const int count)
{
  if (index < 0) {
    return index + count;
  }
  if (index >= count) {
    return index - count;
  }
  return index;
}

__device__ bool is_special(const int partner, const int* partners, int begin, const int end)
{
  while (begin < end && partners[begin] < partner) {
    ++begin;
  }
  return begin < end && partners[begin] == partner;
}

__global__ void find_force(
  const Box box,
  const size_t N,
  const int nx,
  const int ny,
  const int nz,
  const int stencil_size,
  const double cutoff2,
  const int* cell_count,
  const int* cell_sum,
  const int* cell_contents,
  const int num_types,
  const int* types,
  const float* c6,
  const float* c12,
  const int* offsets,
  const int* partners,
  const float* weights,
  const double* position,
  double* potential,
  double* force,
  double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) {
    return;
  }

  const double xi = position[i], yi = position[i + N], zi = position[i + 2 * N];
  const size_t row = static_cast<size_t>(types[i]) * num_types;
  const int begin = offsets ? offsets[i] : 0;
  const int end = offsets ? offsets[i + 1] : 0;
  int ix, iy, iz, cell;
  find_cell_id(box, xi, yi, zi, 1.0, nx, ny, nz, ix, iy, iz, cell);
  int x0, x1, y0, y1, z0, z1;
  get_neighbor_cell_offset_range(box.pbc_x, nx < stencil_size ? nx : stencil_size, x0, x1);
  get_neighbor_cell_offset_range(box.pbc_y, ny < stencil_size ? ny : stencil_size, y0, y1);
  get_neighbor_cell_offset_range(box.pbc_z, nz < stencil_size ? nz : stencil_size, z0, z1);
  // The unshifted LJ energy jumps at the cutoff. Classify only its float roundoff
  // neighborhood using the original coordinates.
  const float cutoff2_float = cutoff2;
  const float inner_cutoff2 = cutoff2_float * (1.0f - 64.0f * FLT_EPSILON);
  const float outer_cutoff2 = cutoff2_float * (1.0f + 64.0f * FLT_EPSILON);
  PairSum sum = {};
  for (int oz = z0; oz < z1; ++oz) {
    const int plane = wrap_bin(iz + oz, nz) * ny;
    for (int oy = y0; oy < y1; ++oy) {
      const int line = (plane + wrap_bin(iy + oy, ny)) * nx;
      for (int ox = x0; ox < x1; ++ox) {
        const int bin = line + wrap_bin(ix + ox, nx);
        const int first = cell_sum[bin];
        const int last = first + cell_count[bin];
        for (int p = first; p < last; ++p) {
          const int j = cell_contents[p];
          if (j == i) {
            continue;
          }
          const double raw_x = position[j] - xi, raw_y = position[j + N] - yi,
                       raw_z = position[j + 2 * N] - zi;
          float dx, dy, dz;
          get_pair_displacement(box, raw_x, raw_y, raw_z, dx, dy, dz);
          const float r2 = dx * dx + dy * dy + dz * dz;
          if (
            r2 <= outer_cutoff2 && !is_special(j, partners, begin, end) &&
            (r2 < inner_cutoff2 || inside_lj_cutoff(box, raw_x, raw_y, raw_z, cutoff2))) {
            const size_t pair = row + types[j];
            add_pair(c6[pair], c12[pair], dx, dy, dz, r2, 1.0f, sum);
          }
        }
      }
    }
  }
  // Explicit scaled pairs use the main periodic image even outside the ordinary cutoff.
  for (int p = begin; p < end; ++p) {
    if (weights[p] == 0.0f) {
      continue;
    }
    const int j = partners[p];
    float dx, dy, dz;
    get_pair_displacement(
      box, position[j] - xi, position[j + N] - yi, position[j + 2 * N] - zi, dx, dy, dz);
    const size_t pair = row + types[j];
    add_pair(c6[pair], c12[pair], dx, dy, dz, dx * dx + dy * dy + dz * dz, weights[p], sum);
  }

  sum.write(i, N, potential, force, virial);
}

__global__ void find_bond_force(
  const Box box,
  const size_t N,
  const int* offsets,
  const int* partners,
  const int* types,
  const float* spring_constants,
  const float* equilibrium_lengths,
  const double* position,
  double* potential,
  double* force,
  double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || offsets[i] == offsets[i + 1]) {
    return;
  }
  const double xi = position[i], yi = position[i + N], zi = position[i + 2 * N];
  PairSum sum = {};
  for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
    const int j = partners[p];
    const int type = types[p];
    float dx, dy, dz;
    get_pair_displacement(
      box, position[j] - xi, position[j + N] - yi, position[j + 2 * N] - zi, dx, dy, dz);
    const float r = sqrtf(dx * dx + dy * dy + dz * dz);
    const float extension = r - equilibrium_lengths[type];
    const float derivative = spring_constants[type] * extension;
    sum.energy_ += 0.25f * derivative * extension;
    // At coincident endpoints no force direction is defined.
    if (r > 0.0f) {
      sum.add_force(dx, dy, dz, derivative / r);
    }
  }
  sum.write(i, N, potential, force, virial);
}

__device__ float dot(const float a[3], const float b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Correct for rounding in c*d so two exactly equal products cancel even with FMA enabled.
template <typename T>
__device__ T difference_of_products(T a, T b, T c, T d)
{
  const T cd = c * d;
  return std::fma(a, b, -cd) + std::fma(-c, d, cd);
}

__device__ void get_pair_displacement(
  const Box& box, double x, double y, double z, double& dx, double& dy, double& dz)
{
  apply_pair_mic(box, x, y, z);
  dx = x;
  dy = y;
  dz = z;
}

template <typename T>
__device__ void find_angle_arm(
  const Box& box, const size_t N, const double* position, int endpoint, int center, T arm[3])
{
  get_pair_displacement(
    box,
    position[endpoint] - position[center],
    position[endpoint + N] - position[center + N],
    position[endpoint + 2 * N] - position[center + 2 * N],
    arm[0],
    arm[1],
    arm[2]);
}

template <typename T>
__device__ void angle_normal(const T a[3], const T b[3], T normal[3])
{
  normal[0] = difference_of_products(a[1], b[2], a[2], b[1]);
  normal[1] = difference_of_products(a[2], b[0], a[0], b[2]);
  normal[2] = difference_of_products(a[0], b[1], a[1], b[0]);
}

// Recover the bending plane from the input coordinates when rounding the two arms to float
// can create or erase a small bend. Scaling before narrowing preserves its direction.
__device__ float recover_angle_normal(
  const Box& box,
  const size_t N,
  const double* position,
  const MolecularAngle& angle,
  float a[3],
  float b[3],
  float normal[3])
{
  double raw_a[3], raw_b[3], raw_normal[3];
  find_angle_arm(box, N, position, angle.first_, angle.center_, raw_a);
  find_angle_arm(box, N, position, angle.last_, angle.center_, raw_b);
  angle_normal(raw_a, raw_b, raw_normal);
  const double scale = fmax(fabs(raw_normal[0]), fmax(fabs(raw_normal[1]), fabs(raw_normal[2])));
  for (int d = 0; d < 3; ++d) {
    a[d] = raw_a[d];
    b[d] = raw_b[d];
    normal[d] = scale == 0.0 ? 0.0f : static_cast<float>(raw_normal[d] / scale);
  }
  return static_cast<float>(scale);
}

// Call the device assertion handler directly so invalid geometry is fatal in all builds.
#if defined(__CUDA_ARCH__) || defined(__CUDA__) || defined(USE_HIP)
#ifdef _WIN32
#define ANGLE_GEOMETRY_ERROR(message) _wassert(L"" message, L"" __FILE__, __LINE__)
#else
#define ANGLE_GEOMETRY_ERROR(message) __assert_fail(message, __FILE__, __LINE__, __func__)
#endif
#else
#define ANGLE_GEOMETRY_ERROR(message) PRINT_INPUT_ERROR(message)
#endif

__global__ void find_angle_force(
  const Box box,
  const size_t N,
  const MolecularAngle* angles,
  const int* offsets,
  const int* indices,
  const float* spring_constants,
  const float* equilibrium_angles,
  const float* pi_minus_equilibrium_angles,
  int* error_reported,
  const double* position,
  double* potential,
  double* force,
  double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N || offsets[i] == offsets[i + 1]) {
    return;
  }
  float energy = 0.0f, force_sum[3] = {}, w[3][3] = {};
  for (int p = offsets[i]; p < offsets[i + 1]; ++p) {
    const MolecularAngle angle = angles[indices[p]];
    const float k = spring_constants[angle.type_];
    if (k == 0.0f) {
      continue;
    }
    const float theta0 = equilibrium_angles[angle.type_];
    const float pi_minus_theta0 = pi_minus_equilibrium_angles[angle.type_];
    float a[3], b[3], normal[3];
    find_angle_arm(box, N, position, angle.first_, angle.center_, a);
    find_angle_arm(box, N, position, angle.last_, angle.center_, b);
    angle_normal(a, b, normal);
    float cross_norm = hypotf(hypotf(normal[0], normal[1]), normal[2]);
    float ab = dot(a, b);
    float scale = 1.0f;
    // Recompute the input plane when |tan(theta)| <= 1/128: cancellation can cost at least
    // seven bits of the normal direction. This also covers collinear and zero-length arms.
    if (!(cross_norm > (1.0f / 128.0f) * fabsf(ab))) {
      scale = recover_angle_normal(box, N, position, angle, a, b, normal);
      cross_norm = hypotf(hypotf(normal[0], normal[1]), normal[2]);
      ab = dot(a, b);
    }
    const float a2 = dot(a, a), b2 = dot(b, b);
    const bool zero_arm = !(a2 > 0.0f && b2 > 0.0f);
    if (zero_arm || cross_norm == 0.0f) {
      if (!zero_arm && (ab >= 0.0f ? theta0 : pi_minus_theta0) == 0.0f) {
        continue;
      }
      if (atomicCAS(error_reported, 0, 1) == 0) {
        if (zero_arm) {
          ANGLE_GEOMETRY_ERROR("A harmonic angle has a zero-length arm");
        } else {
          ANGLE_GEOMETRY_ERROR("A collinear harmonic angle has no unique force direction");
        }
      }
      return;
    }
    for (int d = 0; d < 3; ++d) {
      normal[d] /= cross_norm;
    }
    const float delta = ab >= 0.0f ? atan2f(cross_norm * scale, ab) - theta0
                                   : pi_minus_theta0 - atan2f(cross_norm * scale, -ab);
    const float derivative = k * delta;
    const bool is_center = (i == angle.center_);
    if (is_center) {
      energy += 0.5f * derivative * delta;
    }
    float local_force[3];
    for (int d = 0; d < 3; ++d) {
      const int m = (d + 1) % 3, n = (d + 2) % 3;
      const float f_first = derivative * (normal[m] * a[n] - normal[n] * a[m]) / a2;
      const float f_last = derivative * (b[m] * normal[n] - b[n] * normal[m]) / b2;
      local_force[d] = is_center ? -(f_first + f_last) : (i == angle.first_ ? f_first : f_last);
      force_sum[d] += local_force[d];
    }
    // With the angle energy at the vertex, each endpoint gets the outer product of its arm and
    // force.
    if (!is_center) {
      const float* arm = (i == angle.first_) ? a : b;
      for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
          w[row][col] += arm[row] * local_force[col];
        }
      }
    }
  }
  potential[i] += energy;
  for (int d = 0; d < 3; ++d) {
    force[i + d * N] += force_sum[d];
  }
  virial[i] += w[0][0];
  virial[i + N] += w[1][1];
  virial[i + 2 * N] += w[2][2];
  virial[i + 3 * N] += w[0][1];
  virial[i + 4 * N] += w[0][2];
  virial[i + 5 * N] += w[1][2];
  virial[i + 6 * N] += w[1][0];
  virial[i + 7 * N] += w[2][0];
  virial[i + 8 * N] += w[2][1];
}

#undef ANGLE_GEOMETRY_ERROR

int get_cell_stencil(const Box& box, const double cutoff, const int num_atoms, int bins[3])
{
  const double volume = box.get_volume();
  const double heights[] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};
  if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
    PRINT_INPUT_ERROR("The molecular potential requires periodic boundaries in three dimensions.");
  }
  for (const double h : heights) {
    if (!std::isfinite(h) || h <= 2.0 * cutoff) {
      PRINT_INPUT_ERROR(
        "The molecular potential requires box face heights greater than twice the cutoff.");
    }
  }

  // Keep cell storage proportional to N even for dilute systems or elongated boxes.
  double width = 0.5 * cutoff;
  const int max_cells = std::max(num_atoms, 64);
  for (;;) {
    double counts[3];
    for (int d = 0; d < 3; ++d) {
      counts[d] = std::max(1.0, std::floor(heights[d] / width));
    }
    if (counts[0] * counts[1] * counts[2] <= max_cells) {
      for (int d = 0; d < 3; ++d) {
        bins[d] = static_cast<int>(counts[d]);
      }
      return width >= cutoff ? 3 : 5;
    }
    width *= 2.0;
  }
}
} // namespace

void Molecular::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  if (type.size() != cell_contents_.size() || position.size() != 3 * cell_contents_.size()) {
    PRINT_INPUT_ERROR("The atom count must match the fixed molecular topology.");
  }
  const int N = type.size();
  int bins[3];
  const int stencil = get_cell_stencil(box, rc, N, bins);
  // Unit width and these counts make find_cell_id use floor(fraction * bins).
  // Every lattice translation then shifts the index by exactly a full period.
  Box grid = box;
  grid.thickness_x = bins[0];
  grid.thickness_y = bins[1];
  grid.thickness_z = bins[2];
  find_cell_list(1.0, bins, grid, position, cell_scan_, cell_count_, cell_sum_, cell_contents_);
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  find_force<<<grid_size, block_size>>>(
    grid,
    N,
    bins[0],
    bins[1],
    bins[2],
    stencil,
    rc * rc,
    cell_count_.data(),
    cell_sum_.data(),
    cell_contents_.data(),
    num_types_,
    types_.data(),
    c6_.data(),
    c12_.data(),
    offsets_.size() ? offsets_.data() : nullptr,
    partners_.size() ? partners_.data() : nullptr,
    lj_weights_.size() ? lj_weights_.data() : nullptr,
    position.data(),
    potential.data(),
    force.data(),
    virial.data());
  GPU_CHECK_KERNEL
  if (bond_partners_.size() > 0) {
    find_bond_force<<<grid_size, block_size>>>(
      box,
      N,
      bond_offsets_.data(),
      bond_partners_.data(),
      bond_types_.data(),
      bond_k_.data(),
      bond_r0_.data(),
      position.data(),
      potential.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
  }
  if (angles_.size() > 0) {
    find_angle_force<<<grid_size, block_size>>>(
      box,
      N,
      angles_.data(),
      angle_offsets_.data(),
      angle_indices_.data(),
      angle_k_.data(),
      angle_theta0_.data(),
      angle_pi_minus_theta0_.data(),
      angle_error_reported_.data(),
      position.data(),
      potential.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
  }
}
