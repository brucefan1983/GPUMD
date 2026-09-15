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

#include "neighbor_audit.cuh"
#include "utilities/error.cuh"
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{
void write_box(FILE* output, const char* name, const Box& box)
{
  fprintf(output, ",\"%s\":{\"pbc\":[%d,%d,%d],\"orthogonal\":%s,\"h\":[",
    name, box.pbc_x, box.pbc_y, box.pbc_z, box.is_orthogonal ? "true" : "false");
  for (int i = 0; i < 18; ++i) {
    fprintf(output, "%s%.17g", i ? "," : "", box.cpu_h[i]);
  }
  fprintf(output, "],\"float_h\":[");
  for (int i = 0; i < 18; ++i) {
    fprintf(output, "%s%.17g", i ? "," : "", static_cast<double>(box.float_h[i]));
  }
  fprintf(output, "]}");
}

void write_position(FILE* output, const char* name, int N, const double* position)
{
  fprintf(output, ",\"%s\":[", name);
  for (int i = 0; i < N; ++i) {
    fprintf(output, "%s[%.17g,%.17g,%.17g]", i ? "," : "",
      position[i], position[i + N], position[i + 2 * N]);
  }
  fprintf(output, "]");
}
}

FILE* NeighborAudit::open_event(const char* kind) const
{
  FILE* output = fopen("neighbor_audit.jsonl", "ab");
  if (output == nullptr) {
    PRINT_INPUT_ERROR("Cannot open neighbor_audit.jsonl for diagnostics.");
  }
  // An address identifies a live object without a process-global counter.
  // The reader canonicalizes it in initialization order, never compares addresses.
  fprintf(output, "{\"schema\":1,\"kind\":\"%s\",\"object\":\"%p\",\"call\":%llu",
    kind, static_cast<const void*>(this), calls);
  return output;
}

void NeighborAudit::close_event(FILE* output) const
{
  fprintf(output, "}\n");
  const bool failed = ferror(output) != 0;
  const int close_status = fclose(output);
  if (failed || close_status != 0) {
    PRINT_INPUT_ERROR("Cannot write neighbor_audit.jsonl diagnostics.");
  }
}

void NeighborAudit::initialize(double rc, int num_atoms, int capacity)
{
  // This resets diagnostic state only. No production cache state is touched.
  mode = 0;
  calls = checks = rebuilds = local_filters = nep_filters = 0;
  snapshots = 0;
  first_call = was_rebuilt = dump_this_call = false;
  const char* value = std::getenv("GPUMD_NEIGHBOR_AUDIT");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "off") == 0 ||
      std::strcmp(value, "0") == 0) {
    return;
  }
  if (std::strcmp(value, "counts") == 0) {
    mode = 1;
  } else if (std::strcmp(value, "full") == 0) {
    mode = 2;
  } else {
    PRINT_INPUT_ERROR("GPUMD_NEIGHBOR_AUDIT must be off, counts, or full.");
  }
  FILE* output = open_event("initialize");
  fprintf(output, ",\"mode\":\"%s\",\"n\":%d,\"capacity\":%d,\"rc\":%.17g",
    mode == 1 ? "counts" : "full", num_atoms, capacity, rc);
  close_event(output);
}

void NeighborAudit::begin_global(bool first)
{
  ++calls;
  first_call = first;
  was_rebuilt = false;
  dump_this_call = false;
  if (!first) {
    ++checks;
  }
}

void NeighborAudit::rebuilt(const Box& box, double list_cutoff)
{
  ++rebuilds;
  was_rebuilt = true;
  build_box = box;
  build_cutoff = list_cutoff;
}

void NeighborAudit::write_list(
  FILE* output, const char* name, int N, int N1, int N2,
  const GPU_Vector<int>& NN, const GPU_Vector<int>& NL) const
{
  if (N <= 0 || N1 < 0 || N2 < N1 || N2 > N || NN.size() < static_cast<size_t>(N) ||
      NL.size() % N != 0) {
    PRINT_INPUT_ERROR("Invalid neighbor dimensions in read-only diagnostic export.");
  }
  const size_t capacity = NL.size() / N;
  fprintf(output,
    ",\"%s\":{\"n1\":%d,\"n2\":%d,\"stride\":%d,\"capacity\":%zu,"
    "\"nn_size\":%zu,\"nl_size\":%zu,\"snapshot\":%s",
    name, N1, N2, N, capacity, NN.size(), NL.size(), dump_this_call ? "true" : "false");
  if (dump_this_call) {
    std::vector<int> counts(N2 - N1);
    if (!counts.empty()) {
      CHECK(gpuMemcpy(
        counts.data(), NN.data() + N1, counts.size() * sizeof(int), gpuMemcpyDeviceToHost));
    }
    // Post-build diagnostic only, NOT prevention of a preceding kernel overflow.
    // Check counts before reading NL; never copy padding or unused NN entries.
    for (int i = 0; i < N2 - N1; ++i) {
      if (counts[i] < 0 || static_cast<size_t>(counts[i]) > capacity) {
        PRINT_INPUT_ERROR("NN exceeds allocated capacity in read-only diagnostic export.");
      }
    }
    fprintf(output, ",\"nn\":[");
    for (int i = 0; i < N2 - N1; ++i) {
      fprintf(output, "%s%d", i ? "," : "", counts[i]);
    }
    fprintf(output, "],\"neighbors\":[");
    std::vector<int> row;
    for (int i = 0; i < N2 - N1; ++i) {
      row.resize(counts[i]);
      if (!row.empty()) {
        // One strided copy per atom; only live slots NL[i + N*k] are read.
        // These consumers use the default stream. Full export is intentionally
        // synchronous and must not be used for performance measurements.
#ifdef USE_HIP
        CHECK(hipMemcpy2D(row.data(), sizeof(int), NL.data() + N1 + i,
          static_cast<size_t>(N) * sizeof(int), sizeof(int), row.size(), gpuMemcpyDeviceToHost));
#else
        CHECK(cudaMemcpy2D(row.data(), sizeof(int), NL.data() + N1 + i,
          static_cast<size_t>(N) * sizeof(int), sizeof(int), row.size(), gpuMemcpyDeviceToHost));
#endif
      }
      fprintf(output, "%s[", i ? "," : "");
      for (int j = 0; j < counts[i]; ++j) {
        fprintf(output, "%s%d", j ? "," : "", row[j]);
      }
      fprintf(output, "]");
    }
    fprintf(output, "]");
  }
  fprintf(output, "}");
}

void NeighborAudit::end_global(
  double rc, double skin, const Box& box, const GPU_Vector<int>& type,
  const GPU_Vector<double>& position, const GPU_Vector<double>& x0,
  const GPU_Vector<double>& y0, const GPU_Vector<double>& z0,
  const GPU_Vector<int>& NN, const GPU_Vector<int>& NL)
{
  // Bounded snapshots: first two calls, then actual rebuilds, at most four.
  // Counts and decisions are always written, including calls without a snapshot.
  dump_this_call = mode == 2 && snapshots < 4 && (calls <= 2 || was_rebuilt);
  if (dump_this_call) {
    ++snapshots;
  }
  const int N = type.size();
  FILE* output = open_event("global");
  fprintf(output,
    ",\"checks\":%llu,\"rebuilds\":%llu,\"rebuilt\":%s,\"reason\":\"%s\","
    "\"n\":%d,\"skin\":%.17g,\"rc\":%.17g,\"build_cutoff\":%.17g",
    checks, rebuilds, was_rebuilt ? "true" : "false",
    was_rebuilt ? (first_call ? "first" : "displacement") : "reuse", N, skin, rc, build_cutoff);
  write_list(output, "candidate", N, 0, N, NN, NL);
  if (dump_this_call) {
    if (position.size() != static_cast<size_t>(3) * N ||
        x0.size() != N || y0.size() != N || z0.size() != N) {
      PRINT_INPUT_ERROR("Invalid position dimensions in read-only diagnostic export.");
    }
    std::vector<int> types(N);
    std::vector<double> positions(static_cast<size_t>(3) * N);
    std::vector<double> reference(static_cast<size_t>(3) * N);
    CHECK(gpuMemcpy(types.data(), type.data(), N * sizeof(int), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(
      positions.data(), position.data(), positions.size() * sizeof(double), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(reference.data(), x0.data(), N * sizeof(double), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(reference.data() + N, y0.data(), N * sizeof(double), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(reference.data() + 2 * N, z0.data(), N * sizeof(double), gpuMemcpyDeviceToHost));
    fprintf(output, ",\"type\":[");
    for (int i = 0; i < N; ++i) {
      fprintf(output, "%s%d", i ? "," : "", types[i]);
    }
    fprintf(output, "]");
    write_position(output, "position", N, positions.data());
    write_position(output, "reference_position", N, reference.data());
    write_box(output, "box", box);
    write_box(output, "reference_box", build_box);
  }
  close_event(output);
}

void NeighborAudit::local_filter(double rc, const GPU_Vector<int>& NN, const GPU_Vector<int>& NL)
{
  ++local_filters;
  FILE* output = open_event("local_filter");
  fprintf(output, ",\"filters\":%llu,\"rc\":%.17g", local_filters, rc);
  write_list(output, "local", NN.size(), 0, NN.size(), NN, NL);
  close_event(output);
}

void NeighborAudit::nep_filter(
  int N, int N1, int N2, int num_types, const float* rc_radial, const float* rc_angular,
  const GPU_Vector<int>& NN_radial, const GPU_Vector<int>& NL_radial,
  const GPU_Vector<int>& NN_angular, const GPU_Vector<int>& NL_angular)
{
  ++nep_filters;
  FILE* output = open_event("nep_filter");
  fprintf(output, ",\"filters\":%llu,\"num_types\":%d,\"rc_radial\":[", nep_filters, num_types);
  for (int i = 0; i < num_types; ++i) {
    fprintf(output, "%s%.17g", i ? "," : "", static_cast<double>(rc_radial[i]));
  }
  fprintf(output, "],\"rc_angular\":[");
  for (int i = 0; i < num_types; ++i) {
    fprintf(output, "%s%.17g", i ? "," : "", static_cast<double>(rc_angular[i]));
  }
  fprintf(output, "]");
  write_list(output, "radial", N, N1, N2, NN_radial, NL_radial);
  write_list(output, "angular", N, N1, N2, NN_angular, NL_angular);
  close_event(output);
}
