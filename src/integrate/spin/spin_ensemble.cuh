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
Spin ensemble implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com
*/

#pragma once
#include "integrate/ensemble.cuh"
#include <memory>
#include <string>

class Spin_Integrator;
class NEP_Spin;
class NEP_Spin_MultiGPU;

class Spin_Ensemble final : public Ensemble
{
public:
  Spin_Ensemble(
    const std::vector<std::string>&, const Atom&, const Box&, const std::vector<Group>&, Force&);
  ~Spin_Ensemble() override;
  double get_temperature1() const { return temperature1; }
  double get_temperature2() const { return temperature2; }
  void initialize_run(double, Atom&, Box&, const std::vector<Group>&) override;
  void initialize_before_first_step(
    double, int, const std::vector<Group>&, Box&, Atom&, GPU_Vector<double>&) override;
  void finalize_run(const Atom&, const Box&) override;
  void
  compute1(double, int, int, const std::vector<Group>&, Box&, Atom&, GPU_Vector<double>&) override;
  void compute2(
    double, int, int, const std::vector<Group>&, Box&, Atom&, GPU_Vector<double>&, Force&) override;

private:
  class Lattice_MTTK;
  Force& force_;
  NEP_Spin* potential_ = nullptr;
  NEP_Spin_MultiGPU* multigpu_potential_ = nullptr;
  std::unique_ptr<Ensemble> lattice_;
  std::unique_ptr<Spin_Integrator> spin_integrator;
  double temperature1 = 0.0, temperature2 = 0.0;
  void set_step_temperature(int, int);
  double temperature_coupling = 100.0;
  std::vector<double> spin_mass_factor_by_type;
  int spin_seed = 12345;
  double spin_sib_alpha = 0.0, spin_dynamics_gamma = 0.0, spin_dynamics_temperature = 0.0;
  int spin_dynamics_seed = 12345;
  double spin_glsd_damping = 0.0;  // dimensionless hbar * gamma_s_prime
  double spin_glsd_mobility = 0.0; // converted muB^2 / (eV ps), internal only
  bool use_spin_tspin = false, use_spin_sib = false, use_spin_glsd = false,
       spin_lattice_enabled = true;
  GPU_Vector<double> base_force_, base_mforce_;
  void refresh_endpoint(Force&, Box&, Atom&, const std::vector<Group>&, GPU_Vector<double>&);
};
