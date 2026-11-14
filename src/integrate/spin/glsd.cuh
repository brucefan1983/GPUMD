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
Spin glsd implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com

References:
  P. W. Ma and S. L. Dudarev, Phys. Rev. B 86, 054416 (2012).
*/

#pragma once
#include "integrator.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdint>
#include <vector>

class Atom;

// Ma and Dudarev, PRB 86, 054416 (2012), Eqs. (19), (33), in magnetic-moment units.
class Spin_GLSD : public Spin_Integrator
{
public:
  Spin_GLSD(
    double gamma,
    double mobility,
    double spin_temperature,
    double midpoint_tolerance,
    int seed,
    const std::vector<int>& spin_types,
    Atom& atom);
  void compute1(const double time_step, Atom& atom) override;
  void compute2(const double time_step, Atom& atom) override;
  bool converged() const { return converged_; }
  void finish_step(Atom& atom);
  void report_iterations() const;

private:
  double gamma_;            // numerical magnetic-moment precession factor, g / hbar
  double mobility_;         // muB^2 / (eV ps); g^2 times the paper's gamma_s_prime
  double spin_temperature_; // -1: no noise, 0: follow ensemble, >0: fixed K
  double midpoint_tolerance_;
  double total_iterations_ = 0.0;
  int iterations_ = 0, max_iterations_ = 0;
  std::uint64_t seed_, step_;
  std::uint64_t unconverged_steps_ = 0;
  bool converged_ = false;
  GPU_Vector<int> active_, iteration_status_;
  GPU_Vector<double> spin_start_, spin_endpoint_, noise_increment_;
};
