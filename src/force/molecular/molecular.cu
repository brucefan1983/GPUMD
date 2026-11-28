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
Lennard-Jones and harmonic bond forces for fixed molecular topologies.

Major contributors:
  Zheyong Fan: Initial implementation.
------------------------------------------------------------------------------*/

#include "force/neighbor.cuh"
#include "input.cuh"
#include "model/atom.cuh"
#include "molecular.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>

Molecular::Molecular(const char* forcefield, const char* topology, const Atom& atom)
{
  const MolecularInput input =
    read_molecular_input(forcefield, topology, atom.number_of_atoms, atom.cpu_atom_symbol);
  for (const float charge : atom.cpu_charge) {
    if (charge != 0.0f) {
      PRINT_INPUT_ERROR("The molecular LJ potential requires zero charges in model.xyz.");
    }
  }
  rc = input.cutoff;
  cell_contents.resize(atom.number_of_atoms);
  num_types = input.num_types;
  types.resize(input.types.size());
  types.copy_from_host(input.types.data());
  c6.resize(input.c6.size());
  c6.copy_from_host(input.c6.data());
  c12.resize(input.c12.size());
  c12.copy_from_host(input.c12.data());
  if (!input.partners.empty()) {
    offsets.resize(input.offsets.size());
    offsets.copy_from_host(input.offsets.data());
    partners.resize(input.partners.size());
    partners.copy_from_host(input.partners.data());
    lj_weights.resize(input.lj_weights.size());
    lj_weights.copy_from_host(input.lj_weights.data());
  }
  if (!input.bond_partners.empty()) {
    bond_offsets.resize(input.bond_offsets.size());
    bond_offsets.copy_from_host(input.bond_offsets.data());
    bond_partners.resize(input.bond_partners.size());
    bond_partners.copy_from_host(input.bond_partners.data());
    bond_types.resize(input.bond_types.size());
    bond_types.copy_from_host(input.bond_types.data());
    bond_k.resize(input.bond_k.size());
    bond_k.copy_from_host(input.bond_k.data());
    bond_r0.resize(input.bond_r0.size());
    bond_r0.copy_from_host(input.bond_r0.data());
  }
}

namespace
{
struct PairSum {
  double energy, fx, fy, fz;
  double xx, yy, zz, xy, xz, yz;

  __device__ void add_force(const double dx, const double dy, const double dz, const double f)
  {
    fx += dx * f;
    fy += dy * f;
    fz += dz * f;
    // W_i = -r_ij tensor F_ij / 2 for a directed pair.
    const double v = -0.5 * f;
    xx += dx * dx * v;
    yy += dy * dy * v;
    zz += dz * dz * v;
    xy += dx * dy * v;
    xz += dx * dz * v;
    yz += dy * dz * v;
  }

  __device__ void
  write(const int i, const size_t N, double* potential, double* force, double* virial) const
  {
    potential[i] += energy;
    force[i] += fx;
    force[i + N] += fy;
    force[i + 2 * N] += fz;
    virial[i] += xx;
    virial[i + N] += yy;
    virial[i + 2 * N] += zz;
    virial[i + 3 * N] += xy;
    virial[i + 4 * N] += xz;
    virial[i + 5 * N] += yz;
    virial[i + 6 * N] += xy;
    virial[i + 7 * N] += xz;
    virial[i + 8 * N] += yz;
  }
};

__device__ void add_pair(
  const double c6,
  const double c12,
  const double dx,
  const double dy,
  const double dz,
  const double r2,
  const double weight,
  PairSum& sum)
{
  if (c6 == 0.0 && c12 == 0.0) {
    return;
  }
  const double inv2 = 1.0 / r2;
  const double inv6 = inv2 * inv2 * inv2;
  const double attraction = c6 * inv6;
  const double repulsion = c12 * inv6 * inv6;
  const double f = weight * 6.0 * (attraction - 2.0 * repulsion) * inv2;
  sum.energy += weight * 0.5 * (repulsion - attraction);
  sum.add_force(dx, dy, dz, f);
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
  const double* c6,
  const double* c12,
  const int* offsets,
  const int* partners,
  const double* weights,
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
          double dx = position[j] - xi, dy = position[j + N] - yi, dz = position[j + 2 * N] - zi;
          apply_pair_mic(box, dx, dy, dz);
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < cutoff2 && !is_special(j, partners, begin, end)) {
            const size_t pair = row + types[j];
            add_pair(c6[pair], c12[pair], dx, dy, dz, r2, 1.0, sum);
          }
        }
      }
    }
  }
  // Explicit scaled pairs use the main periodic image even outside the ordinary cutoff.
  for (int p = begin; p < end; ++p) {
    if (weights[p] == 0.0) {
      continue;
    }
    const int j = partners[p];
    double dx = position[j] - xi, dy = position[j + N] - yi, dz = position[j + 2 * N] - zi;
    apply_pair_mic(box, dx, dy, dz);
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
  const double* spring_constants,
  const double* equilibrium_lengths,
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
    double dx = position[j] - xi, dy = position[j + N] - yi, dz = position[j + 2 * N] - zi;
    apply_pair_mic(box, dx, dy, dz);
    const double r = sqrt(dx * dx + dy * dy + dz * dz);
    const double extension = r - equilibrium_lengths[type];
    const double derivative = spring_constants[type] * extension;
    sum.energy += 0.25 * derivative * extension;
    // At coincident endpoints no force direction is defined.
    if (r > 0.0) {
      sum.add_force(dx, dy, dz, derivative / r);
    }
  }
  sum.write(i, N, potential, force, virial);
}

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
  if (type.size() != cell_contents.size() || position.size() != 3 * cell_contents.size()) {
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
  find_cell_list(1.0, bins, grid, position, cell_scan_, cell_count, cell_sum, cell_contents);
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
    cell_count.data(),
    cell_sum.data(),
    cell_contents.data(),
    num_types,
    types.data(),
    c6.data(),
    c12.data(),
    offsets.size() ? offsets.data() : nullptr,
    partners.size() ? partners.data() : nullptr,
    lj_weights.size() ? lj_weights.data() : nullptr,
    position.data(),
    potential.data(),
    force.data(),
    virial.data());
  GPU_CHECK_KERNEL
  if (bond_partners.size() > 0) {
    find_bond_force<<<grid_size, block_size>>>(
      box,
      N,
      bond_offsets.data(),
      bond_partners.data(),
      bond_types.data(),
      bond_k.data(),
      bond_r0.data(),
      position.data(),
      potential.data(),
      force.data(),
      virial.data());
    GPU_CHECK_KERNEL
  }
}
