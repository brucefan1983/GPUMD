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
Spin step implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com
*/

#include "spin_ensemble.cuh"
#include "glsd.cuh"
#include "integrator.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>

namespace
{
__global__ void midpoint_position(
  int n, Box box, const double* start, const double* endpoint, double* midpoint)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double r[3], s[3];
  for (int d = 0; d < 3; ++d)
    r[d] = 0.5 * (start[d * n + i] + endpoint[d * n + i]);
  for (int d = 0; d < 3; ++d)
    s[d] = box.cpu_h[9 + 3*d] * r[0] + box.cpu_h[10 + 3*d] * r[1] +
           box.cpu_h[11 + 3*d] * r[2];
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  // Force's public PBC pass only removes one image; discard all midpoint images here.
  for (int d = 0; d < 3; ++d) {
    const double image = pbc[d] ? floor(s[d]) : 0.0;
    for (int k = 0; k < 3; ++k)
      r[k] -= box.cpu_h[3*k + d] * image;
  }
  for (int d = 0; d < 3; ++d)
    midpoint[d * n + i] = r[d];
}

// Save each Action increment once at the predictor spin. Strip its Zeeman
// energy here so it can be evaluated at each corrected spin.
__global__ void capture_modifiers(
  int n, const double* spin, const double* force, const double* field,
  const double* energy, const double* virial, double* df, double* db, double* de, double* dv)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  de[i] = energy[i] - de[i];
  for (int d = 0; d < 3; ++d) {
    const int k = d * n + i;
    df[k] = force[k] - df[k];
    db[k] = field[k] - db[k];
    de[i] += db[k] * spin[k];
  }
  for (int d = 0; d < 9; ++d) {
    const size_t k = size_t(d) * n + i;
    dv[k] = virial[k] - dv[k];
  }
}
__global__ void restore_modifiers(
  int n,
  const double* df,
  const double* db,
  const double* de,
  const double* dv,
  const double* spin,
  double* force,
  double* mforce,
  double* potential,
  double* virial)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double zeeman = 0.0;
    for (int d = 0; d < 3; ++d) {
      const int k = i + d * n;
      force[k] += df[k];
      mforce[k] += db[k];
      zeeman -= db[k] * spin[k];
    }
    potential[i] += de[i] + zeeman;
    for (int d = 0; d < 9; ++d) {
      const size_t k = size_t(d) * n + i;
      virial[k] += dv[k];
    }
  }
}
} // namespace

void Spin_Ensemble::refresh_endpoint(
  double dt, Force& force, Box& box, Atom& atom, const std::vector<Group>& group,
  GPU_Vector<double>& thermo)
{
  // Reuse each Action's already sampled increment. Replaying post_force would
  // resample mechanical noise. Magnetic Actions add a constant field on a group
  // or fractional region; the corrector preserves membership under cell scaling.
  const int n = atom.number_of_atoms;
  capture_modifiers<<<(n + 127) / 128, 128>>>(
    n, atom.spin_per_atom.data(), atom.force_per_atom.data(), atom.mforce_per_atom.data(),
    atom.potential_per_atom.data(), atom.virial_per_atom.data(), modifiers_.force.data(),
    modifiers_.mforce.data(), modifiers_.energy.data(), modifiers_.virial.data());
  GPU_CHECK_KERNEL
  if (use_spin_sib)
    spin_integrator->compute2(dt, atom);
  auto evaluate = [&]() {
    force.compute(box, atom, group);
    restore_modifiers<<<(n + 127) / 128, 128>>>(
      n,
      modifiers_.force.data(),
      modifiers_.mforce.data(),
      modifiers_.energy.data(),
      modifiers_.virial.data(),
      atom.spin_per_atom.data(),
      atom.force_per_atom.data(),
      atom.mforce_per_atom.data(),
      atom.potential_per_atom.data(),
      atom.virial_per_atom.data());
    GPU_CHECK_KERNEL
  };
  const bool coupled_glsd = use_spin_glsd && lattice_;
  const Box endpoint_box = box;
  if (coupled_glsd) {
    endpoint_position_.copy_from_device(atom.position_per_atom.data());
    endpoint_images_.copy_from_device(atom.position_image.data());
    for (int i = 0; i < 9; ++i)
      box.cpu_h[i] = 0.5 * (start_box_.cpu_h[i] + endpoint_box.cpu_h[i]);
    box.get_inverse();
    box.set_is_orthogonal();
    // Interpolate continuously, then wrap completely in the midpoint cell before force evaluation.
    midpoint_position<<<(n + 127) / 128, 128>>>(
      n, box, start_position_.data(), atom.unwrapped_position.data(),
      atom.position_per_atom.data());
    GPU_CHECK_KERNEL
    atom.position_image.fill(0);
    evaluate();
  }
  if (use_spin_glsd) {
    // With fixed geometry the predictor field already belongs to the midpoint.
    for (int iteration = 0;; ++iteration) {
      auto& glsd = static_cast<Spin_GLSD&>(*spin_integrator);
      glsd.compute2(dt, atom);
      if (glsd.converged() || iteration + 1 >= spin_glsd_midpoint_max_iter) {
        glsd.finish_step(atom);
        break;
      }
      evaluate();
    }
  }
  if (coupled_glsd) {
    box = endpoint_box;
    atom.position_per_atom.copy_from_device(endpoint_position_.data());
    atom.position_image.copy_from_device(endpoint_images_.data());
  }
  evaluate();
  atom.update_unwrapped_position(box);
  find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
}
