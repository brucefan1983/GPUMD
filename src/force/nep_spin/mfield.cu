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

#include "force/force.cuh"
#include "mfield.cuh"
#include "model/box.cuh"
#include "nep_spin.cuh"
#include "nep_spin_multigpu.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

Spin_Magnetic_Field::Spin_Magnetic_Field(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  action_name = "add_mfield";
  std::vector<const char*> fields;
  for (const auto& token : tokens) {
    fields.push_back(token.c_str());
  }
  parse(fields.data(), fields.size(), group);
}

void Spin_Magnetic_Field::pre_run(
  int, double, Integrate&, std::vector<Group>&, Atom& atom, Box&, Force& force)
{
  const std::vector<int>* types = nullptr;
  if (force.get_number_of_potentials() == 1) {
    auto& potential = force.get_potential(0);
    if (const auto* spin = dynamic_cast<const NEP_Spin*>(&potential)) {
      types = &spin->spin_types();
    } else if (const auto* spin = dynamic_cast<const NEP_Spin_MultiGPU*>(&potential)) {
      types = &spin->spin_types();
    }
  }
  if (!types || !atom.has_spin) {
    PRINT_INPUT_ERROR("add_mfield requires a Spin potential and spin:R:3.");
  }
  active_types_.resize(types->size());
  active_types_.copy_from_host(types->data());
}

void Spin_Magnetic_Field::setup_force(
  double, Integrate&, std::vector<Group>& group, Atom& atom, Box& box, Force&)
{
  compute(box, group, atom);
}

void Spin_Magnetic_Field::post_force(
  int, double, Integrate&, std::vector<Group>& group, Atom& atom, Box& box, Force&)
{
  compute(box, group, atom);
}

static __device__ void get_fractional_position(
  const Box& box,
  const double x,
  const double y,
  const double z,
  double& sa,
  double& sb,
  double& sc)
{
  sa = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  sb = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  sc = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
}

static __device__ bool is_in_region(
  const double sa,
  const double sb,
  const double sc,
  const double amin,
  const double amax,
  const double bmin,
  const double bmax,
  const double cmin,
  const double cmax)
{
  return sa >= amin && sa < amax && sb >= bmin && sb < bmax && sc >= cmin && sc < cmax;
}

// spin is in muB; this converts a tesla-valued B to magnetic force in eV/muB.
void Spin_Magnetic_Field::parse(const char** param, int num_param, const std::vector<Group>& group)
{
  mfield_use_region_ = num_param > 1 && strcmp(param[1], "region") == 0;
  if (num_param != (mfield_use_region_ ? 11 : 6)) {
    PRINT_INPUT_ERROR("add_mfield requires group_method group_id Bx By Bz or "
                      "region amin amax bmin bmax cmin cmax Bx By Bz.");
  }
  if (mfield_use_region_) {
    for (int k = 0; k < 6; ++k) {
      if (
        !is_valid_real(param[k + 2], &mfield_region_[k]) || !std::isfinite(mfield_region_[k]) ||
        mfield_region_[k] < 0.0 || mfield_region_[k] > 1.0) {
        PRINT_INPUT_ERROR("add_mfield region bounds must be finite numbers in [0, 1].");
      }
    }
    for (int k = 0; k < 6; k += 2) {
      if (mfield_region_[k] >= mfield_region_[k + 1]) {
        PRINT_INPUT_ERROR("add_mfield region minimum must be smaller than maximum.");
      }
    }
  } else {
    if (
      !is_valid_int(param[1], &mfield_group_method_) || mfield_group_method_ < 0 ||
      mfield_group_method_ >= group.size()) {
      PRINT_INPUT_ERROR("add_mfield grouping method is invalid.");
    }
    if (
      !is_valid_int(param[2], &mfield_group_id_) || mfield_group_id_ < 0 ||
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
    printf(
      "    fractional region [%g, %g) [%g, %g) [%g, %g).\n",
      mfield_region_[0],
      mfield_region_[1],
      mfield_region_[2],
      mfield_region_[3],
      mfield_region_[4],
      mfield_region_[5]);
  } else {
    printf("    grouping method %d, group %d.\n", mfield_group_method_, mfield_group_id_);
  }
}

template <bool UseRegion>
static __global__ void add_mfield(
  const int n,
  const int count,
  const int offset,
  const int* members,
  const double hx,
  const double hy,
  const double hz,
  const Box box,
  const double* position,
  const double amin,
  const double amax,
  const double bmin,
  const double bmax,
  const double cmin,
  const double cmax,
  const double* spin,
  const int* type,
  const int* active,
  double* energy,
  double* mforce)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count)
    return;
  const int i = UseRegion ? tid : members[offset + tid];
  if (!active[type[i]]) {
    return;
  }
  if (UseRegion) {
    double sa, sb, sc;
    get_fractional_position(box, position[i], position[n + i], position[2 * n + i], sa, sb, sc);
    if (!is_in_region(sa, sb, sc, amin, amax, bmin, bmax, cmin, cmax))
      return;
  }
  energy[i] -= spin[i] * hx + spin[n + i] * hy + spin[2 * n + i] * hz;
  mforce[i] += hx;
  mforce[n + i] += hy;
  mforce[2 * n + i] += hz;
}

void Spin_Magnetic_Field::compute(Box& box, const std::vector<Group>& group, Atom& atom) const
{
  const int number_of_atoms = atom.number_of_atoms;
  if (mfield_enabled_) {
    if (mfield_use_region_) {
      add_mfield<true><<<(number_of_atoms + 127) / 128, 128>>>(
        number_of_atoms,
        number_of_atoms,
        0,
        nullptr,
        mfield_[0],
        mfield_[1],
        mfield_[2],
        box,
        atom.position_per_atom.data(),
        mfield_region_[0],
        mfield_region_[1],
        mfield_region_[2],
        mfield_region_[3],
        mfield_region_[4],
        mfield_region_[5],
        atom.spin_per_atom.data(),
        atom.type.data(),
        active_types_.data(),
        atom.potential_per_atom.data(),
        atom.mforce_per_atom.data());
      GPU_CHECK_KERNEL
    } else {
      const Group& selected = group[mfield_group_method_];
      const int count = selected.cpu_size[mfield_group_id_];
      if (count > 0) {
        add_mfield<false><<<(count + 127) / 128, 128>>>(
          number_of_atoms,
          count,
          selected.cpu_size_sum[mfield_group_id_],
          selected.contents.data(),
          mfield_[0],
          mfield_[1],
          mfield_[2],
          box,
          nullptr,
          0,
          0,
          0,
          0,
          0,
          0,
          atom.spin_per_atom.data(),
          atom.type.data(),
          active_types_.data(),
          atom.potential_per_atom.data(),
          atom.mforce_per_atom.data());
        GPU_CHECK_KERNEL
      }
    }
  }
}
