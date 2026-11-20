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
Isotropic stochastic cell rescaling (SCR) barostat for one replica of NPT
replica-exchange molecular dynamics; see replica_scr.cuh.

Reference: M. Bernetti and G. Bussi, J. Chem. Phys. 153, 114107 (2020).

Major contributors:
  Jiuyang Shi: Initial implementation.
  Wenhao Luo: Integration into the gpumd replica runtime.
------------------------------------------------------------------------------*/

#include "ensemble_bdp_replica.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "remd_utilities.cuh"
#include "replica_scr.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include <cmath>
#include <iomanip>
#include <sstream>

static __global__ void
gpu_scr_scale(const int number_of_atoms, const double scale, double* position, double* velocity)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    for (int d = 0; d < 3; ++d) {
      position[n + d * number_of_atoms] *= scale;
      velocity[n + d * number_of_atoms] /= scale;
    }
  }
}

namespace replica
{
SCR_Barostat::SCR_Barostat(
  const int device_id,
  const int replica_id,
  const double bulk_modulus,
  const double tau_p,
  const int interval,
  const unsigned int seed)
  : device_id_(device_id),
    replica_id_(replica_id),
    compressibility_(PRESSURE_UNIT_CONVERSION / bulk_modulus),
    tau_p_(tau_p),
    interval_(interval),
    rng_(seed),
    host_thermo_(5, 0.0),
    ready_(device_id)
{
}

void SCR_Barostat::set_target_pressure(const double pressure) { target_pressure_ = pressure; }

void SCR_Barostat::apply_pending_update(
  const double temperature, Box& box, Atom& atom, const gpuStream_t stream)
{
  if (!pending_)
    return;
  ready_.synchronize();
  pending_ = false;
  const double volume = box.get_volume();
  const double pressure = (host_thermo_[2] + host_thermo_[3] + host_thermo_[4]) / 3.0;
  const double d_epsilon = scr_log_volume_increment(
    temperature,
    volume,
    target_pressure_,
    pressure,
    compressibility_,
    interval_ / tau_p_,
    normal_(rng_));
  const double scale = std::exp(d_epsilon / 3.0);
  if (!std::isfinite(scale) || scale < 0.95 || scale > 1.05) {
    std::ostringstream message;
    message << std::setprecision(10)
            << "SCR barostat changed a box length by more than 5% in one update (replica "
            << replica_id_ << ", replica MD step " << step_counter_ << ", length scale " << scale
            << ", volume " << volume << " A^3, target pressure "
            << target_pressure_ * PRESSURE_UNIT_CONVERSION << " GPa, measured pressure "
            << pressure * PRESSURE_UNIT_CONVERSION << " GPa); increase tau_p or the bulk modulus.";
    PRINT_INPUT_ERROR(message.str().c_str());
  }
  for (int d = 0; d < 3; ++d)
    box.cpu_h[d * 4] *= scale;
  box.get_inverse();

  const int n = atom.number_of_atoms;
  gpu_scr_scale<<<(n - 1) / 128 + 1, 128, 0, stream>>>(
    n, scale, atom.position_per_atom.data(), atom.velocity_per_atom.data());
  GPU_CHECK_KERNEL
}

void SCR_Barostat::end_step(
  const Box& box,
  Atom& atom,
  Ensemble_BDP_Replica& thermostat,
  GPU_Vector<double>& thermo,
  const GPU_Stream& stream)
{
  if (++step_counter_ % interval_ == 0)
    measure_pressure(box, atom, thermostat, thermo, stream);
}

void SCR_Barostat::refresh_pending_pressure(
  const Box& box,
  Atom& atom,
  Ensemble_BDP_Replica& thermostat,
  GPU_Vector<double>& thermo,
  const GPU_Stream& stream)
{
  if (pending_)
    measure_pressure(box, atom, thermostat, thermo, stream);
}

void SCR_Barostat::measure_pressure(
  const Box& box,
  Atom& atom,
  Ensemble_BDP_Replica& thermostat,
  GPU_Vector<double>& thermo,
  const GPU_Stream& stream)
{
  CHECK(gpuSetDevice(device_id_));
  thermostat.find_thermo(box.get_volume(), atom, thermo, stream.get());
  thermo.copy_to_host_async(host_thermo_.data(), 5, stream.get());
  ready_.record(stream);
  pending_ = true;
}

std::string SCR_Barostat::export_state() const
{
  return export_scr_state(step_counter_, pending_, host_thermo_.data(), rng_, normal_);
}

bool SCR_Barostat::import_state(const std::string& state)
{
  long long counter = 0;
  bool pending = false;
  double thermo[5];
  std::mt19937 rng;
  std::normal_distribution<double> normal;
  // A checkpoint is written between steps, so an update is pending exactly when
  // the last integrated step ended an update interval.
  if (
    !import_scr_state(state, interval_, counter, pending, thermo, rng, normal) ||
    pending != (counter > 0 && counter % interval_ == 0))
    return false;
  step_counter_ = counter;
  pending_ = pending;
  for (int i = 0; i < 5; ++i)
    host_thermo_[i] = thermo[i];
  rng_ = rng;
  normal_ = normal;
  return true;
}
} // namespace replica
