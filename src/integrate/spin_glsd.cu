// SPDX-License-Identifier: LGPL-3.0-or-later
#include "spin_glsd.cuh"
#include "spin_noise.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <limits>

// Paper spin s = -M/(g muB); stored S = M/muB = -g s.
// H_s = -g H_S, so Eq. (19) becomes -g/hbar S x H_S + g^2 gamma_s_prime H_S.
// Eq. (33) transforms to <dS_noise dS_noise^T> = 2 mobility kBT dt I.
// Cartesian GLSD: dS = (gamma H x S + mobility H) dt + sqrt(2 mobility kBT) dW.
// The transverse part of mobility H is -mobility S x (S x H)/|S|^2;
// evaluating the full vector avoids a singular longitudinal/transverse split at zero.
static __global__ void gpu_glsd_heun(
  const int n, const bool correct, const double dt, const double gamma,
  const double mobility, const double sigma, const std::uint64_t seed,
  const std::uint64_t step, const double* field, double* spin,
  double* start, double* drift_start, double* noise)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double sx = spin[i], sy = spin[n+i], sz = spin[2*n+i];
  const double hx = field[i], hy = field[n+i], hz = field[2*n+i];
  const double drift[3] = {gamma * (hy*sz-hz*sy) + mobility*hx,
                           gamma * (hz*sx-hx*sz) + mobility*hy,
                           gamma * (hx*sy-hy*sx) + mobility*hz};
  for (int d = 0; d < 3; ++d) {
    const int k = d*n+i;
    if (!correct) {
      start[k] = spin[k];
      drift_start[k] = drift[d];
      noise[k] = sigma > 0.0 ? sigma * spin_noise::gaussian(seed, i+1, step, d) : 0.0;
      spin[k] = start[k] + dt * drift[d] + noise[k];
    } else {
      spin[k] = start[k] + 0.5 * dt * (drift_start[k] + drift[d]) + noise[k];
    }
  }
}


Spin_GLSD::Spin_GLSD(
  double gamma, double mobility, double spin_temperature, int seed, Atom& atom)
  : gamma_(gamma), mobility_(mobility), spin_temperature_(spin_temperature),
    seed_(static_cast<std::uint64_t>(seed)), step_(atom.spin_glsd_step)
{
  const int n = atom.number_of_atoms;
  if (!atom.has_spin || atom.spin_per_atom.size() != 3 * n ||
      atom.mforce_per_atom.size() != 3 * n) {
    PRINT_INPUT_ERROR("GLSD requires complete spin and mforce arrays.");
  }
  if (!std::isfinite(gamma_) || gamma_ <= 0.0 ||
      !std::isfinite(mobility_) || mobility_ < 0.0 ||
      !std::isfinite(spin_temperature_) ||
      (spin_temperature_ < 0.0 && spin_temperature_ != -1.0) || seed <= 0) {
    PRINT_INPUT_ERROR("GLSD requires gamma > 0, mobility >= 0, stemp -1 or >= 0, and seed > 0.");
  }
  if (atom.spin_glsd_seed != 0 && atom.spin_glsd_seed != seed_) {
    PRINT_INPUT_ERROR("GLSD restart seed differs from ensemble seed.");
  }
  atom.spin_glsd_seed = seed_;
  // First-order Langevin dynamics has no independent inertial spin velocity.
  atom.spin_velocity_initialized = false;
  atom.spin_velocity_per_atom.fill(0.0);
  spin_start_.resize(3 * n);
  drift_start_.resize(3 * n);
  noise_increment_.resize(3 * n);
}

void Spin_GLSD::compute1(const double time_step, Atom& atom)
{
  const double dt_ps = time_step * TIME_UNIT_CONVERSION / 1000.0;
  if (!(dt_ps > 0.0) || !std::isfinite(dt_ps)) {
    PRINT_INPUT_ERROR("GLSD requires a finite positive timestep.");
  }
  if (step_ == std::numeric_limits<std::uint64_t>::max()) {
    PRINT_INPUT_ERROR("GLSD noise counter exhausted.");
  }
  const double noise_temperature = spin_temperature_ < 0.0 ? 0.0 :
    (spin_temperature_ == 0.0 ? temperature : spin_temperature_);
  const double sigma = sqrt(2.0 * mobility_ * K_B * noise_temperature * dt_ps);
  gpu_glsd_heun<<<(atom.number_of_atoms + 127) / 128, 128>>>(
    atom.number_of_atoms, false, dt_ps, gamma_, mobility_, sigma, seed_, step_,
    atom.mforce_per_atom.data(), atom.spin_per_atom.data(), spin_start_.data(),
    drift_start_.data(), noise_increment_.data());
  GPU_CHECK_KERNEL
}

void Spin_GLSD::compute2(const double time_step, Atom& atom)
{
  const double dt_ps = time_step * TIME_UNIT_CONVERSION / 1000.0;
  gpu_glsd_heun<<<(atom.number_of_atoms + 127) / 128, 128>>>(
    atom.number_of_atoms, true, dt_ps, gamma_, mobility_, 0.0, seed_, step_,
    atom.mforce_per_atom.data(), atom.spin_per_atom.data(), spin_start_.data(),
    drift_start_.data(), noise_increment_.data());
  GPU_CHECK_KERNEL
  atom.spin_glsd_step = ++step_;
}
