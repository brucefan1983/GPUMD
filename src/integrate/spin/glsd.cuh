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
    int seed,
    const std::vector<int>& spin_types,
    Atom& atom);
  void compute1(const double time_step, Atom& atom) override;
  void compute2(const double time_step, Atom& atom) override;

private:
  double gamma_;            // numerical magnetic-moment precession factor, g / hbar
  double mobility_;         // muB^2 / (eV ps); g^2 times the paper's gamma_s_prime
  double spin_temperature_; // -1: no noise, 0: follow ensemble, >0: fixed K
  std::uint64_t seed_, step_;
  GPU_Vector<int> active_;
  GPU_Vector<double> spin_start_, drift_start_, noise_increment_;
};
