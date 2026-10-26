// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "spin_integrator.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdint>

class Atom;

// Ma and Dudarev, PRB 86, 054416 (2012), Eqs. (19), (33), in magnetic-moment units.
class Spin_GLSD : public Spin_Integrator
{
public:
  Spin_GLSD(double gamma, double mobility, double spin_temperature, int seed, Atom& atom);
  void compute1(const double time_step, Atom& atom) override;
  void compute2(const double time_step, Atom& atom) override;

private:
  double gamma_; // numerical magnetic-moment precession factor, g / hbar
  double mobility_; // muB^2 / (eV ps); g^2 times the paper's gamma_s_prime
  double spin_temperature_; // -1: no noise, 0: follow ensemble, >0: fixed K
  std::uint64_t seed_, step_;
  GPU_Vector<double> spin_start_, drift_start_, noise_increment_;
};
