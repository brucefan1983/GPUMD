/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <vector>
class Group;

void parse_group(
  const char** param,
  const int num_param,
  const bool allow_all_groups,
  const std::vector<Group>& groups,
  int& k,
  int& grouping_method,
  int& group_id);

void parse_precision(const char** param, const int num_param, int& k, int& precision);

// The per-atom quantities that dump_xyz and dump_netcdf can write. Shared so that the two
// keywords accept the same names and cannot drift apart.
struct DumpQuantities {
  bool has_velocity_ = false;
  bool has_force_ = false;
  bool has_potential_ = false;
  bool has_unwrapped_position_ = false;
  bool has_mass_ = false;
  bool has_charge_ = false;
  bool has_bec_ = false;
  bool has_virial_ = false;
  bool has_group_ = false;
};

// Returns false when the token is not a quantity at all, leaving the caller free to try its own
// options before rejecting it. `keyword` only names the keyword in error messages.
bool parse_dump_quantity(
  const char* token,
  DumpQuantities& quantities,
  const bool is_nep_charge,
  const std::vector<Group>& groups,
  const char* keyword);
