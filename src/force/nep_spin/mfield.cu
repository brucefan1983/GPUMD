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

#include "mfield.cuh"
#include "utilities/error.cuh"
#include "utilities/fractional_region.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

// spin is in muB; this converts a tesla-valued B to magnetic force in eV/muB.
void Spin_Magnetic_Field::parse(const char** param, int num_param, const std::vector<Group>& group)
{
  mfield_use_region_ = num_param > 1 && strcmp(param[1], "region") == 0;
  if (num_param != (mfield_use_region_ ? 11 : 6)) {
    PRINT_INPUT_ERROR(
      "add_mfield requires group_method group_id Bx By Bz or "
      "region amin amax bmin bmax cmin cmax Bx By Bz.");
  }
  if (mfield_use_region_) {
    for (int k = 0; k < 6; ++k) {
      if (!is_valid_real(param[k + 2], &mfield_region_[k]) ||
          !std::isfinite(mfield_region_[k]) || mfield_region_[k] < 0.0 ||
          mfield_region_[k] > 1.0) {
        PRINT_INPUT_ERROR("add_mfield region bounds must be finite numbers in [0, 1].");
      }
    }
    for (int k = 0; k < 6; k += 2) {
      if (mfield_region_[k] >= mfield_region_[k + 1]) {
        PRINT_INPUT_ERROR("add_mfield region minimum must be smaller than maximum.");
      }
    }
  } else {
    if (!is_valid_int(param[1], &mfield_group_method_) ||
        mfield_group_method_ < 0 || mfield_group_method_ >= group.size()) {
      PRINT_INPUT_ERROR("add_mfield grouping method is invalid.");
    }
    if (!is_valid_int(param[2], &mfield_group_id_) || mfield_group_id_ < 0 ||
        mfield_group_id_ >= group[mfield_group_method_].number) {
      PRINT_INPUT_ERROR("add_mfield group id is invalid.");
    }
  }
  const int field_start = mfield_use_region_ ? 8 : 3;
  constexpr double muB_eV_per_T = 5.7883818060e-5;
  double field[3];
  mfield_enabled_ = false;
  for (int k = 0; k < 3; ++k) {
    if (!is_valid_real(param[k + field_start], &field[k]) || !std::isfinite(field[k])) {
      PRINT_INPUT_ERROR("add_mfield components must be finite numbers in tesla.");
    }
    mfield_[k] = field[k] * muB_eV_per_T;
    mfield_enabled_ = mfield_enabled_ || field[k] != 0.0;
  }
  printf("External magnetic field: (%g, %g, %g) T.\n", field[0], field[1], field[2]);
  if (mfield_use_region_) {
    printf("    fractional region [%g, %g) [%g, %g) [%g, %g).\n",
      mfield_region_[0], mfield_region_[1], mfield_region_[2],
      mfield_region_[3], mfield_region_[4], mfield_region_[5]);
  } else {
    printf("    grouping method %d, group %d.\n", mfield_group_method_, mfield_group_id_);
  }
}

template <bool UseRegion>
static __global__ void add_mfield(
  const int n, const int count, const int offset, const int* members,
  const double hx, const double hy, const double hz,
  const Box box, const double* position,
  const double amin, const double amax, const double bmin, const double bmax,
  const double cmin, const double cmax,
  const double* spin, double* energy, double* mforce)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  const int i = UseRegion ? tid : members[offset + tid];
  if (UseRegion) {
    double sa, sb, sc;
    get_fractional_position(box, position[i], position[n + i], position[2 * n + i], sa, sb, sc);
    if (!is_in_region(sa, sb, sc, amin, amax, bmin, bmax, cmin, cmax)) return;
  }
  energy[i] -= spin[i] * hx + spin[n + i] * hy + spin[2 * n + i] * hz;
  mforce[i] += hx;
  mforce[n + i] += hy;
  mforce[2 * n + i] += hz;
}

void Spin_Magnetic_Field::compute(
  const Box& box, const GPU_Vector<double>& position_per_atom,
  const std::vector<Group>& group, const GPU_Vector<double>& spin_per_atom,
  GPU_Vector<double>& potential_per_atom, GPU_Vector<double>& mforce_per_atom) const
{
  const int number_of_atoms = spin_per_atom.size() / 3;
  if (mfield_enabled_) {
    if (mfield_use_region_) {
      add_mfield<true><<<(number_of_atoms + 127) / 128, 128>>>(
        number_of_atoms, number_of_atoms, 0, nullptr, mfield_[0], mfield_[1], mfield_[2],
        box, position_per_atom.data(), mfield_region_[0], mfield_region_[1],
        mfield_region_[2], mfield_region_[3], mfield_region_[4], mfield_region_[5],
        spin_per_atom.data(), potential_per_atom.data(), mforce_per_atom.data());
      GPU_CHECK_KERNEL
    } else {
      const Group& selected = group[mfield_group_method_];
      const int count = selected.cpu_size[mfield_group_id_];
      if (count > 0) {
        add_mfield<false><<<(count + 127) / 128, 128>>>(
          number_of_atoms, count, selected.cpu_size_sum[mfield_group_id_], selected.contents.data(),
          mfield_[0], mfield_[1], mfield_[2], box, nullptr, 0, 0, 0, 0, 0, 0,
          spin_per_atom.data(), potential_per_atom.data(), mforce_per_atom.data());
        GPU_CHECK_KERNEL
      }
    }
  }
}
