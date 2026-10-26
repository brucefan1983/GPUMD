/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "spin_sib.cuh"
#include "spin_noise.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdint>

namespace
{
__device__ void cayley_direction(
  const double ex,
  const double ey,
  const double ez,
  const double ox,
  const double oy,
  const double oz,
  double& out_x,
  double& out_y,
  double& out_z)
{
  const double ax = 0.5 * ox;
  const double ay = 0.5 * oy;
  const double az = 0.5 * oz;
  const double a2 = ax * ax + ay * ay + az * az;
  const double adot = ax * ex + ay * ey + az * ez;
  const double denominator = 1.0 + a2;
  out_x =
    ((1.0 - a2) * ex + 2.0 * (ay * ez - az * ey) + 2.0 * ax * adot) /
    denominator;
  out_y =
    ((1.0 - a2) * ey + 2.0 * (az * ex - ax * ez) + 2.0 * ay * adot) /
    denominator;
  out_z =
    ((1.0 - a2) * ez + 2.0 * (ax * ey - ay * ex) + 2.0 * az * adot) /
    denominator;
}
} // namespace

static __global__ void gpu_sib_predictor_midpoint(
  const int number_of_atoms,
  const double drift,
  const double alpha,
  const double noise_prefactor,
  const std::uint64_t seed,
  const std::uint64_t step,
  const double* mforce,
  double* spin,
  double* spin_start,
  double* noise_increment)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= number_of_atoms) {
    return;
  }

  const double sx = spin[atom];
  const double sy = spin[number_of_atoms + atom];
  const double sz = spin[2 * number_of_atoms + atom];
  spin_start[atom] = sx;
  spin_start[number_of_atoms + atom] = sy;
  spin_start[2 * number_of_atoms + atom] = sz;

  const double magnitude = sqrt(sx * sx + sy * sy + sz * sz);
  if (magnitude <= 1.0e-30) {
    noise_increment[atom] = 0.0;
    noise_increment[number_of_atoms + atom] = 0.0;
    noise_increment[2 * number_of_atoms + atom] = 0.0;
    spin[atom] = 0.0;
    spin[number_of_atoms + atom] = 0.0;
    spin[2 * number_of_atoms + atom] = 0.0;
    return;
  }

  const double ex = sx / magnitude;
  const double ey = sy / magnitude;
  const double ez = sz / magnitude;
  const double hx = mforce[atom];
  const double hy = mforce[number_of_atoms + atom];
  const double hz = mforce[2 * number_of_atoms + atom];
  const double sigma = noise_prefactor > 0.0 ? sqrt(noise_prefactor / magnitude) : 0.0;
  const double nx = sigma * spin_noise::gaussian(seed, atom + 1, step, 0);
  const double ny = sigma * spin_noise::gaussian(seed, atom + 1, step, 1);
  const double nz = sigma * spin_noise::gaussian(seed, atom + 1, step, 2);
  noise_increment[atom] = nx;
  noise_increment[number_of_atoms + atom] = ny;
  noise_increment[2 * number_of_atoms + atom] = nz;

  const double cx = ey * hz - ez * hy;
  const double cy = ez * hx - ex * hz;
  const double cz = ex * hy - ey * hx;
  const double ox = drift * (hx + alpha * cx) + nx;
  const double oy = drift * (hy + alpha * cy) + ny;
  const double oz = drift * (hz + alpha * cz) + nz;
  double px, py, pz;
  cayley_direction(ex, ey, ez, ox, oy, oz, px, py, pz);

  // SIB evaluates the second magnetic field on the arithmetic chord midpoint.
  // This midpoint is intentionally not normalized.
  spin[atom] = magnitude * 0.5 * (ex + px);
  spin[number_of_atoms + atom] = magnitude * 0.5 * (ey + py);
  spin[2 * number_of_atoms + atom] = magnitude * 0.5 * (ez + pz);
}

static __global__ void gpu_sib_corrector(
  const int number_of_atoms,
  const double drift,
  const double alpha,
  const double* spin_start,
  const double* noise_increment,
  const double* mforce,
  double* spin)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= number_of_atoms) {
    return;
  }

  const double sx = spin_start[atom];
  const double sy = spin_start[number_of_atoms + atom];
  const double sz = spin_start[2 * number_of_atoms + atom];
  const double magnitude = sqrt(sx * sx + sy * sy + sz * sz);
  if (magnitude <= 1.0e-30) {
    spin[atom] = 0.0;
    spin[number_of_atoms + atom] = 0.0;
    spin[2 * number_of_atoms + atom] = 0.0;
    return;
  }

  const double ex = sx / magnitude;
  const double ey = sy / magnitude;
  const double ez = sz / magnitude;
  const double mx = spin[atom] / magnitude;
  const double my = spin[number_of_atoms + atom] / magnitude;
  const double mz = spin[2 * number_of_atoms + atom] / magnitude;
  const double hx = mforce[atom];
  const double hy = mforce[number_of_atoms + atom];
  const double hz = mforce[2 * number_of_atoms + atom];
  const double cx = my * hz - mz * hy;
  const double cy = mz * hx - mx * hz;
  const double cz = mx * hy - my * hx;
  const double ox =
    drift * (hx + alpha * cx) + noise_increment[atom];
  const double oy =
    drift * (hy + alpha * cy) + noise_increment[number_of_atoms + atom];
  const double oz =
    drift * (hz + alpha * cz) + noise_increment[2 * number_of_atoms + atom];
  double out_x, out_y, out_z;
  cayley_direction(ex, ey, ez, ox, oy, oz, out_x, out_y, out_z);
  spin[atom] = magnitude * out_x;
  spin[number_of_atoms + atom] = magnitude * out_y;
  spin[2 * number_of_atoms + atom] = magnitude * out_z;
}

Spin_SIB::Spin_SIB(
  double alpha,
  double gamma,
  double spin_temperature,
  int seed,
  Atom& atom)
  : alpha_(alpha),
    gamma_(gamma),
    spin_temperature_(spin_temperature),
    seed_(static_cast<std::uint64_t>(seed))
{
  const int number_of_atoms = atom.number_of_atoms;
  if (!atom.has_spin || atom.spin_per_atom.size() != 3 * number_of_atoms ||
      atom.mforce_per_atom.size() != 3 * number_of_atoms) {
    PRINT_INPUT_ERROR("SIB requires complete spin and mforce arrays.");
  }
  if (!std::isfinite(alpha_) || alpha_ < 0.0 ||
      !std::isfinite(gamma_) || gamma_ <= 0.0 ||
      !std::isfinite(spin_temperature_) ||
      (spin_temperature_ < 0.0 && spin_temperature_ != -1.0) ||
      seed <= 0) {
    PRINT_INPUT_ERROR(
      "SIB requires alpha >= 0, gamma > 0, stemp >= -1, and a positive seed.");
  }
  spin_start_.resize(3 * number_of_atoms);
  noise_increment_.resize(3 * number_of_atoms);
}

void Spin_SIB::compute1(const double time_step, Atom& atom)
{
  const double dt_ps = time_step * TIME_UNIT_CONVERSION / 1000.0;
  const double denominator = 1.0 + alpha_ * alpha_;
  const double drift = gamma_ * dt_ps / denominator;
  const double noise_temperature =
    spin_temperature_ < 0.0 ? 0.0 :
    (spin_temperature_ == 0.0 ? temperature : spin_temperature_);
  // The thermal increment enters the rotation vector directly, without the
  // Gilbert cross term, so its variance must match the damping mobility:
  // sigma^2 = 2 alpha gamma k_B T dt / (mu_s (1 + alpha^2)). The drift already
  // carries one 1 / (1 + alpha^2) factor; applying it twice here would make the
  // bath equilibrate at T / (1 + alpha^2).
  const double noise_prefactor =
    (alpha_ > 0.0 && noise_temperature > 0.0) ?
    2.0 * alpha_ * gamma_ * K_B * noise_temperature * dt_ps /
      denominator :
    0.0;
  const int number_of_atoms = atom.number_of_atoms;
  gpu_sib_predictor_midpoint<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    drift,
    alpha_,
    noise_prefactor,
    seed_,
    step_,
    atom.mforce_per_atom.data(),
    atom.spin_per_atom.data(),
    spin_start_.data(),
    noise_increment_.data());
  GPU_CHECK_KERNEL
}

void Spin_SIB::compute2(const double time_step, Atom& atom)
{
  const double dt_ps = time_step * TIME_UNIT_CONVERSION / 1000.0;
  const double drift = gamma_ * dt_ps / (1.0 + alpha_ * alpha_);
  const int number_of_atoms = atom.number_of_atoms;
  gpu_sib_corrector<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    drift,
    alpha_,
    spin_start_.data(),
    noise_increment_.data(),
    atom.mforce_per_atom.data(),
    atom.spin_per_atom.data());
  GPU_CHECK_KERNEL
  ++step_;
}
