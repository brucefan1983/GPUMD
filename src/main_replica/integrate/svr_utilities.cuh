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

/*----------------------------------------------------------------------------80
// The following functions are from Bussi's website
// https://sites.google.com/site/giovannibussi/Research/algorithms
// I have only added "static" in front of the functions, and changed ran1()
// to C++ calls
// Reference:
[1] G. Bussi et al. J. Chem. Phys. 126, 014101 (2007).
------------------------------------------------------------------------------*/

#pragma once
#include "svr_state.cuh"
#include "utilities/gpu_macro.cuh"
#include <random>

static SVR_Gaussian_Cache& default_svr_gaussian_cache()
{
  static SVR_Gaussian_Cache cache;
  return cache;
}

static double gasdev(std::mt19937& rng, SVR_Gaussian_Cache& cache)
{
  std::uniform_real_distribution<double> rand1(0, 1);

  double fac, rsq, v1, v2;

  if (cache.iset == 0) {
    do {
      v1 = 2.0 * rand1(rng) - 1.0;
      v2 = 2.0 * rand1(rng) - 1.0;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    cache.gset = v1 * fac;
    cache.iset = 1;
    return v2 * fac;
  } else {
    cache.iset = 0;
    return cache.gset;
  }
}

inline double gasdev(std::mt19937& rng)
{
  return gasdev(rng, default_svr_gaussian_cache());
}

static double gamdev(const int ia, std::mt19937& rng)
{
  std::uniform_real_distribution<double> rand1(0, 1);

  int j;
  double am, e, s, v1, v2, x, y;

  if (ia < 1) {
  }; // FATAL ERROR
  if (ia < 6) {
    x = 1.0;
    for (j = 1; j <= ia; j++)
      x *= rand1(rng);
    x = -log(x);
  } else {
    do {
      do {
        do {
          v1 = rand1(rng);
          v2 = 2.0 * rand1(rng) - 1.0;
        } while (v1 * v1 + v2 * v2 > 1.0);
        y = v2 / v1;
        am = ia - 1;
        s = sqrt(2.0 * am + 1.0);
        x = s * y + am;
      } while (x <= 0.0);
      e = (1.0 + y * y) * exp(am * log(x / am) - s * y);
    } while (rand1(rng) > e);
  }
  return x;
}

static double
resamplekin_sumnoises(int nn, std::mt19937& rng, SVR_Gaussian_Cache& cache)
{
  /*
    returns the sum of n independent gaussian noises squared
     (i.e. equivalent to summing the square of the return values of nn calls to gasdev)
  */
  double rr;
  if (nn == 0) {
    return 0.0;
  } else if (nn == 1) {
    rr = gasdev(rng, cache);
    return rr * rr;
  } else if (nn % 2 == 0) {
    return 2.0 * gamdev(nn / 2, rng);
  } else {
    rr = gasdev(rng, cache);
    return 2.0 * gamdev((nn - 1) / 2, rng) + rr * rr;
  }
}

struct SVR_Noise {
  double gaussian;
  double sum_of_squares;
};

static SVR_Noise
draw_resamplekin_noise(int ndeg, std::mt19937& rng, SVR_Gaussian_Cache& cache)
{
  SVR_Noise noise;
  noise.gaussian = gasdev(rng, cache);
  noise.sum_of_squares = resamplekin_sumnoises(ndeg - 1, rng, cache);
  return noise;
}

static double resamplekin(
  double kk,
  double sigma,
  int ndeg,
  double taut,
  std::mt19937& rng,
  SVR_Gaussian_Cache& cache)
{
  /*
    kk:    present value of the kinetic energy of the atoms to be thermalized (in arbitrary units)
    sigma: target average value of the kinetic energy (ndeg k_b T/2)  (in the same units as kk)
    ndeg:  number of degrees of freedom of the atoms to be thermalized
    taut:  relaxation time of the thermostat, in units of 'how often this routine is called'
  */
  double factor;
  if (taut > 0.1) {
    factor = exp(-1.0 / taut);
  } else {
    factor = 0.0;
  }
  const SVR_Noise noise = draw_resamplekin_noise(ndeg, rng, cache);
  return kk +
         (1.0 - factor) *
           (sigma * (noise.sum_of_squares + noise.gaussian * noise.gaussian) / ndeg - kk) +
         2.0 * noise.gaussian * sqrt(kk * sigma / ndeg * (1.0 - factor) * factor);
}

inline double resamplekin(double kk, double sigma, int ndeg, double taut, std::mt19937& rng)
{
  return resamplekin(kk, sigma, ndeg, taut, rng, default_svr_gaussian_cache());
}
