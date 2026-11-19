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

// Part of the REMD-TP implementation proposed and primarily developed by
// Jiuyang Shi, ported into gpumd by Wenhao Luo. The SCR method is due to
// Bernetti and Bussi; see replica_scr.cuh for the reference.
#include "remd_utilities.cuh"
#include "replica_common.cuh"
#include "replica_scr.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <iomanip>
#include <sstream>

namespace replica
{
SCR_Barostat::SCR_Barostat(const int device_id) : host_thermo(5, 0.0), ready(device_id) {}

double SCR_Barostat::update_box(Box& box, const double target_temperature)
{
  const double volume = box.get_volume();
  int number_of_coupled = 0;
  double pressure = 0.0;
  for (int d = 0; d < 3; ++d) {
    if (couple[d]) {
      pressure += host_thermo[2 + d];
      ++number_of_coupled;
    }
  }
  pressure /= number_of_coupled;
  const double d_epsilon = scr_log_volume_increment(
    target_temperature,
    volume,
    target_pressure,
    pressure,
    compressibility,
    interval / tau_p,
    normal(rng));
  const double scale = std::exp(d_epsilon / number_of_coupled);
  if (!std::isfinite(scale) || scale < 0.95 || scale > 1.05) {
    std::ostringstream message;
    message << std::setprecision(10)
            << "SCR barostat changed a box length by more than 5% in one update (replica "
            << replica_id << ", replica MD step " << step_counter << ", length scale " << scale
            << ", volume " << volume << " A^3, target pressure "
            << target_pressure * PRESSURE_UNIT_CONVERSION << " GPa, measured pressure "
            << pressure * PRESSURE_UNIT_CONVERSION << " GPa); increase tau_p or the bulk modulus.";
    PRINT_INPUT_ERROR(message.str().c_str());
  }
  for (int d = 0; d < 3; ++d) {
    if (couple[d])
      box.cpu_h[d * 4] *= scale;
  }
  box.get_inverse();
  return scale;
}

std::string SCR_Barostat::export_state() const
{
  return export_scr_state(step_counter, pending, host_thermo.data(), rng, normal);
}

bool SCR_Barostat::import_state(const std::string& state)
{
  return import_scr_state(state, interval, step_counter, pending, host_thermo.data(), rng, normal);
}

static __global__ void gpu_scr_scale(
  const int number_of_atoms,
  const double scale_x,
  const double scale_y,
  const double scale_z,
  double* x,
  double* y,
  double* z,
  double* vx,
  double* vy,
  double* vz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    x[n] *= scale_x;
    y[n] *= scale_y;
    z[n] *= scale_z;
    vx[n] /= scale_x;
    vy[n] /= scale_y;
    vz[n] /= scale_z;
  }
}

void refresh_barostat_pressure(Replica_Slot& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  slot.thermostat->find_thermo(slot.box.get_volume(), slot.atom, slot.thermo, slot.stream.get());
  slot.thermo.copy_to_host_async(slot.barostat->host_thermo.data(), 5, slot.stream.get());
  slot.barostat->ready.record(slot.stream);
  slot.barostat->pending = true;
}

void apply_barostat(Replica_Slot& slot)
{
  SCR_Barostat& barostat = *slot.barostat;
  barostat.ready.synchronize();
  barostat.pending = false;
  const double scale = barostat.update_box(slot.box, slot.target_temperature);
  const int n = slot.atom.number_of_atoms;
  double* position = slot.atom.position_per_atom.data();
  double* velocity = slot.atom.velocity_per_atom.data();
  gpu_scr_scale<<<(n - 1) / 128 + 1, 128, 0, slot.stream.get()>>>(
    n,
    barostat.couple[0] ? scale : 1.0,
    barostat.couple[1] ? scale : 1.0,
    barostat.couple[2] ? scale : 1.0,
    position,
    position + n,
    position + 2 * n,
    velocity,
    velocity + n,
    velocity + 2 * n);
  GPU_CHECK_KERNEL
}

} // namespace replica
