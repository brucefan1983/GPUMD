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
#include "replica_scr.cuh"
#include "replica_common.cuh"
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
  const double kT = K_B * target_temperature;
  const double dt_over_tau = interval / tau_p;
  const double d_epsilon =
    -compressibility * dt_over_tau * (target_pressure - pressure) +
    std::sqrt(2.0 * kT * compressibility * dt_over_tau / volume) * normal(rng);
  const double scale = std::exp(d_epsilon / number_of_coupled);
  if (!std::isfinite(scale) || scale < 0.95 || scale > 1.05)
    PRINT_INPUT_ERROR(
      "SCR barostat changed a box length by more than 5% in one update; "
      "increase tau_p or the bulk modulus.");
  for (int d = 0; d < 3; ++d) {
    if (couple[d])
      box.cpu_h[d * 4] *= scale;
  }
  box.get_inverse();
  return scale;
}

std::string SCR_Barostat::export_state() const
{
  std::ostringstream output;
  output << 1 << ' ' << step_counter << ' ' << (pending ? 1 : 0) << ' '
         << std::setprecision(17);
  for (int i = 0; i < 5; ++i)
    output << host_thermo[i] << ' ';
  output << rng << ' ' << normal;
  return output.str();
}

bool SCR_Barostat::import_state(const std::string& state)
{
  std::istringstream input(state);
  int version = 0;
  long long counter = 0;
  int restored_pending = 0;
  double thermo[5];
  std::mt19937 restored_rng;
  std::normal_distribution<double> restored_normal;
  if (!(input >> version >> counter >> restored_pending) || version != 1 || counter < 0 ||
      (restored_pending != 0 && restored_pending != 1))
    return false;
  for (int i = 0; i < 5; ++i) {
    if (!(input >> thermo[i]) || !std::isfinite(thermo[i]))
      return false;
  }
  if (!(input >> restored_rng >> restored_normal))
    return false;
  if (counter % interval != 0 && restored_pending)
    return false;
  if (!std::isfinite(restored_normal.mean()) || restored_normal.mean() != 0.0 ||
      !std::isfinite(restored_normal.stddev()) || restored_normal.stddev() != 1.0)
    return false;
  input >> std::ws;
  if (!input.eof())
    return false;
  step_counter = counter;
  pending = restored_pending == 1;
  for (int i = 0; i < 5; ++i)
    host_thermo[i] = thermo[i];
  rng = restored_rng;
  normal = restored_normal;
  return true;
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
  slot.thermostat->find_thermo(
    slot.box.get_volume(), slot.atom, slot.thermo, slot.stream.get());
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

static void zero_linear_momentum(const int N, const double* m, double* vx, double* vy, double* vz)
{
  double center_of_mass_velocity[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < N; ++n) {
    total_mass += m[n];
    center_of_mass_velocity[0] += m[n] * vx[n];
    center_of_mass_velocity[1] += m[n] * vy[n];
    center_of_mass_velocity[2] += m[n] * vz[n];
  }
  center_of_mass_velocity[0] /= total_mass;
  center_of_mass_velocity[1] /= total_mass;
  center_of_mass_velocity[2] /= total_mass;
  for (int n = 0; n < N; ++n) {
    vx[n] -= center_of_mass_velocity[0];
    vy[n] -= center_of_mass_velocity[1];
    vz[n] -= center_of_mass_velocity[2];
  }
}

void generate_canonical_velocities(
  const double temperature,
  const std::vector<double>& mass,
  std::vector<double>& velocity,
  const unsigned int seed)
{
  if (mass.size() < 2 || !std::isfinite(temperature) || temperature <= 0.0)
    PRINT_INPUT_ERROR("Canonical velocity sampling requires at least two atoms and a positive finite temperature.");
  for (const double m : mass) {
    if (!std::isfinite(m) || m <= 0.0)
      PRINT_INPUT_ERROR("Canonical velocity sampling requires positive finite masses.");
  }

  const int N = mass.size();
  velocity.resize(3 * N);
  // Keep the distribution local so no cached Gaussian survives a sampling call.
  std::mt19937 rng(seed);
  std::normal_distribution<double> gaussian(0.0, 1.0);
  for (int n = 0; n < N; ++n) {
    const double sigma = std::sqrt(K_B * temperature / mass[n]);
    for (int d = 0; d < 3; ++d)
      velocity[n + d * N] = sigma * gaussian(rng);
  }
  zero_linear_momentum(
    N, mass.data(), velocity.data(), velocity.data() + N, velocity.data() + 2 * N);
  // Do not remove angular momentum or rescale to a fixed kinetic energy:
  // the constrained canonical distribution must retain its kinetic fluctuations.
}


} // namespace replica
