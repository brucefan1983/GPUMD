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
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>

// NR1a: host-only, read-only diagnostics for the default-stream Neighbor path.
// Disabled unless GPUMD_NEIGHBOR_AUDIT is "counts" or "full". No list ownership.
class NeighborAudit
{
public:
  bool enabled() const { return mode != 0; }
  bool force_rebuild() const { return force_rebuild_enabled; }
  void initialize(double rc, int num_atoms, int capacity);
  void begin_global(bool first, bool forced);
  void rebuilt(const Box& box, double list_cutoff);
  void end_global(
    double rc, double skin, const Box& box, const GPU_Vector<int>& type,
    const GPU_Vector<double>& position, const GPU_Vector<double>& x0,
    const GPU_Vector<double>& y0, const GPU_Vector<double>& z0,
    const GPU_Vector<int>& NN, const GPU_Vector<int>& NL);
  void local_filter(double rc, const GPU_Vector<int>& NN, const GPU_Vector<int>& NL);
  void nep_filter(
    int N, int N1, int N2, int num_types, const float* rc_radial, const float* rc_angular,
    const GPU_Vector<int>& NN_radial, const GPU_Vector<int>& NL_radial,
    const GPU_Vector<int>& NN_angular, const GPU_Vector<int>& NL_angular);

private:
  int mode = 0;
  unsigned long long calls = 0;
  unsigned long long checks = 0;
  unsigned long long rebuilds = 0;
  unsigned long long local_filters = 0;
  unsigned long long nep_filters = 0;
  bool first_call = false;
  bool forced_call = false;
  bool force_rebuild_enabled = false;
  bool was_rebuilt = false;
  bool dump_this_call = false;
  int snapshots = 0;
  double build_cutoff = 0.0;
  Box build_box;

  FILE* open_event(const char* kind) const;
  void close_event(FILE* output) const;
  void write_list(
    FILE* output, const char* name, int N, int N1, int N2,
    const GPU_Vector<int>& NN, const GPU_Vector<int>& NL) const;
};
