/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
DFT-DN dispersion Hessian via central-differencing of the analytic
dispersion force.  See nep_hessian_dftd3.cuh for the high-level contract.
------------------------------------------------------------------------------*/

#include "nep_hessian_dftd3.cuh"

#include "force/force.cuh"
#include "force/nep.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "sparse_hessian.cuh"
#include "nep_hessian_accessors.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace nep_hessian {

namespace {

// Device buffer view of one of Atom's GPU_Vector<double> members.  Mirrors
// AtomStateBackup in nep_analytic_hessian.cu so the FD perturbations below
// can be undone before returning.
struct DeviceBuffer {
  double* ptr;
  size_t size;
  DeviceBuffer(double* p, size_t s) : ptr(p), size(s) {}
};

class AtomStateBackupDFTD3
{
public:
  explicit AtomStateBackupDFTD3(Atom& atom)
    : position_(atom.position_per_atom.data(), atom.position_per_atom.size()),
      force_(atom.force_per_atom.data(), atom.force_per_atom.size()),
      potential_(atom.potential_per_atom.data(), atom.potential_per_atom.size()),
      virial_(atom.virial_per_atom.data(), atom.virial_per_atom.size())
  {
  }

  bool save()
  {
    if (!copy_d2h(position_.ptr, position_host_, position_.size)) return false;
    if (!copy_d2h(force_.ptr, force_host_, force_.size)) return false;
    if (!copy_d2h(potential_.ptr, potential_host_, potential_.size)) return false;
    if (!copy_d2h(virial_.ptr, virial_host_, virial_.size)) return false;
    return true;
  }

  bool restore()
  {
    if (!copy_h2d(force_host_.data(), force_.ptr, force_.size)) return false;
    if (!copy_h2d(potential_host_.data(), potential_.ptr, potential_.size)) return false;
    if (!copy_h2d(virial_host_.data(), virial_.ptr, virial_.size)) return false;
    if (!copy_h2d(position_host_.data(), position_.ptr, position_.size)) return false;
    return true;
  }

private:
  DeviceBuffer position_;
  DeviceBuffer force_;
  DeviceBuffer potential_;
  DeviceBuffer virial_;
  std::vector<double> position_host_;
  std::vector<double> force_host_;
  std::vector<double> potential_host_;
  std::vector<double> virial_host_;

  static bool copy_d2h(
    const double* src, std::vector<double>& dst, size_t count)
  {
    if (count == 0) {
      dst.clear();
      return true;
    }
    dst.resize(count);
    const gpuError_t err = gpuMemcpy(
      dst.data(), src, count * sizeof(double), gpuMemcpyDeviceToHost);
    if (err != gpuSuccess) {
      fprintf(stderr, "DFTD3 backup: D2H failed: %s\n", gpuGetErrorString(err));
      return false;
    }
    return true;
  }

  static bool copy_h2d(
    const double* src, double* dst, size_t count)
  {
    if (count == 0) return true;
    const gpuError_t err = gpuMemcpy(
      dst, src, count * sizeof(double), gpuMemcpyHostToDevice);
    if (err != gpuSuccess) {
      fprintf(stderr, "DFTD3 backup: H2D failed: %s\n", gpuGetErrorString(err));
      return false;
    }
    return true;
  }
};

// Default central-difference displacement.  Matches compute_hessian fd's
// default of 1.0e-3 Angstrom so that the dftd3-fallback hessian is
// numerically equivalent to a stand-alone FD run on the same geometry.
constexpr double kDFTD3DefaultDisplacement = 1.0e-3;

} // namespace

bool compute_nep_hessian_with_dftd3(
  Force& force,
  Box& box,
  Atom& atom,
  const std::vector<double>& position_soa,
  int N,
  std::vector<double>& hessian_out,
  std::vector<double>& force_out,
  double& pe_out,
  sparse_hessian::BlockPattern* sparse_pattern)
{
  hessian_out.clear();
  force_out.clear();
  pe_out = 0.0;
  if (sparse_pattern) *sparse_pattern = sparse_hessian::BlockPattern{};

  if (N <= 0 || N > std::numeric_limits<int>::max() / 3 ||
      atom.number_of_atoms != N) {
    return false;
  }
  const int N3 = 3 * N;
  if (position_soa.size() != static_cast<size_t>(N3)) {
    printf("NEP analytic Hessian (dftd3): explicit position array size is invalid.\n");
    return false;
  }
  if (force.get_number_of_potentials() != 1) return false;
  auto* nep = dynamic_cast<NEP*>(&force.get_potential(0));
  if (nep == nullptr) return false;

  // Recover NEP neighbor lists for the sparse pattern (mirrors the analytic
  // path's choice between expanded-box and large-box storage).
  const bool use_expanded_box = nep_hessian::nep_requires_expanded_box(*nep, box);
  GPU_Vector<int>& radial_neighbor_counts_gpu = use_expanded_box
    ? nep->small_box_data.NN_radial
    : nep->nep_data.NN_radial;
  GPU_Vector<int>& radial_neighbor_list_gpu = use_expanded_box
    ? nep->small_box_data.NL_radial
    : nep->nep_data.NL_radial;
  GPU_Vector<int>& angular_neighbor_counts_gpu = use_expanded_box
    ? nep->small_box_data.NN_angular
    : nep->nep_data.NN_angular;
  GPU_Vector<int>& angular_neighbor_list_gpu = use_expanded_box
    ? nep->small_box_data.NL_angular
    : nep->nep_data.NL_angular;

  AtomStateBackupDFTD3 backup(atom);
  if (!backup.save()) {
    printf("NEP analytic Hessian (dftd3): failed to save Atom state.\n");
    return false;
  }

  auto finish = [&](bool success) {
    if (!backup.restore()) success = false;
    if (!success) {
      hessian_out.clear();
      force_out.clear();
      pe_out = 0.0;
    }
    return success;
  };

  // Upload the caller-supplied geometry.
  const gpuError_t upload_err = gpuMemcpy(
    atom.position_per_atom.data(), position_soa.data(),
    N3 * sizeof(double), gpuMemcpyHostToDevice);
  if (upload_err != gpuSuccess) {
    fprintf(stderr,
      "NEP analytic Hessian (dftd3): failed to upload explicit position: %s.\n",
      gpuGetErrorString(upload_err));
    return finish(false);
  }

  // Baseline force / energy via the public force.compute path.  Because
  // dftd3 was enabled in run.in this dispatch goes through NEP::compute,
  // which calls dftd3.compute on top of the regular NEP terms.
  std::vector<Group> empty_groups;
  force.compute(
    box, atom.position_per_atom, atom.type, empty_groups,
    atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
  CHECK(gpuDeviceSynchronize());

  std::vector<double> force_baseline(N3);
  atom.force_per_atom.copy_to_host(force_baseline.data());
  std::vector<double> potential_baseline(N);
  atom.potential_per_atom.copy_to_host(potential_baseline.data());
  pe_out = 0.0;
  for (int i = 0; i < N; ++i) pe_out += potential_baseline[i];
  force_out = std::move(force_baseline);

  // Build the sparse block pattern (optional).  We use the same neighbor
  // lists as the analytic path so the user-facing sparse layout is the
  // same regardless of whether dftd3 is enabled.
  if (sparse_pattern != nullptr) {
    if (radial_neighbor_counts_gpu.size() != static_cast<size_t>(N) ||
        angular_neighbor_counts_gpu.size() != static_cast<size_t>(N) ||
        radial_neighbor_list_gpu.size() % static_cast<size_t>(N) != 0 ||
        angular_neighbor_list_gpu.size() % static_cast<size_t>(N) != 0) {
      printf(
        "NEP analytic Hessian (dftd3): neighbor-list dimensions are invalid.\n");
      return finish(false);
    }
    const size_t angular_neighbor_capacity =
      angular_neighbor_list_gpu.size() / N;
    const size_t radial_neighbor_capacity =
      radial_neighbor_list_gpu.size() / N;

    std::vector<int> radial_counts(N), angular_counts(N);
    radial_neighbor_counts_gpu.copy_to_host(radial_counts.data());
    angular_neighbor_counts_gpu.copy_to_host(angular_counts.data());

    std::vector<int> radial_list(radial_neighbor_list_gpu.size());
    std::vector<int> angular_list(angular_neighbor_list_gpu.size());
    radial_neighbor_list_gpu.copy_to_host(radial_list.data());
    angular_neighbor_list_gpu.copy_to_host(angular_list.data());

    try {
      *sparse_pattern = sparse_hessian::build_pattern_from_neighbors(
        N, radial_counts, radial_list, radial_neighbor_capacity,
        angular_counts, angular_list, angular_neighbor_capacity);
    } catch (const std::exception& error) {
      printf("NEP analytic Hessian (dftd3): sparse pattern failed: %s\n",
             error.what());
      return finish(false);
    }
  }

  // Central-difference loop over all 3N Cartesian directions.  Each step
  // calls force.compute (which dispatches to NEP::compute and so includes
  // the dftd3 contribution) and computes
  //   H[row, column] = -(F_plus[row] - F_minus[row]) / (2 * displacement)
  // matching the convention of the analytic NEP hessian and the FD mode.
  double displacement = kDFTD3DefaultDisplacement;
  if (const char* text = std::getenv("GPUMD_DFTD3_FD_DISPLACEMENT")) {
    const double value = std::atof(text);
    if (std::isfinite(value) && value > 0.0) displacement = value;
  }

  std::vector<double> hessian(static_cast<size_t>(N3) * N3, 0.0);
  std::vector<double> displacement_buffer(position_soa.begin(), position_soa.end());
  std::vector<double> force_plus(N3), force_minus(N3);

  for (int column = 0; column < N3; ++column) {
    displacement_buffer[column] = position_soa[column] + displacement;
    atom.position_per_atom.copy_from_host(displacement_buffer.data());
    force.compute(
      box, atom.position_per_atom, atom.type, empty_groups,
      atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
    CHECK(gpuDeviceSynchronize());
    atom.force_per_atom.copy_to_host(force_plus.data());

    displacement_buffer[column] = position_soa[column] - displacement;
    atom.position_per_atom.copy_from_host(displacement_buffer.data());
    force.compute(
      box, atom.position_per_atom, atom.type, empty_groups,
      atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
    CHECK(gpuDeviceSynchronize());
    atom.force_per_atom.copy_to_host(force_minus.data());

    displacement_buffer[column] = position_soa[column];

    const double inv_two_delta = 1.0 / (2.0 * displacement);
    for (int row = 0; row < N3; ++row) {
      hessian[static_cast<size_t>(row) * N3 + column] =
        -(force_plus[row] - force_minus[row]) * inv_two_delta;
    }
  }

  // Restore the atom baseline state and ship the result out.
  if (!backup.restore()) return false;

  // If a sparse block pattern was requested, scatter the dense FD Hessian
  // into the per-block sparse layout expected by the public Hessian API
  // (one block of 3x3 doubles per non-zero atom pair, indexed by
  // block * 9 + a * 3 + b in SoA).  This mirrors the storage order of the
  // analytic NEP Hessian so downstream sparse writers see a uniform
  // representation regardless of whether dft3 was enabled.
  if (sparse_pattern != nullptr) {
    const auto& pattern = *sparse_pattern;
    std::vector<double> sparse_values(pattern.value_count(), 0.0);
    for (int atom = 0; atom < N; ++atom) {
      for (int block = pattern.row_offsets[atom];
           block < pattern.row_offsets[atom + 1]; ++block) {
        const int other = pattern.columns[block];
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            const size_t sparse_index =
              static_cast<size_t>(block) * 9 + a * 3 + b;
            const size_t dense_index =
              static_cast<size_t>(a * N + atom) * N3 + b * N + other;
            sparse_values[sparse_index] = hessian[dense_index];
          }
        }
      }
    }
    hessian_out = std::move(sparse_values);
  } else {
    hessian_out = std::move(hessian);
  }
  return true;
}

} // namespace nep_hessian
